"""Async probe nodes for extracting media metadata.

Provides VideoProbeNode, ImageProbeNode, and AudioProbeNode with:
- Fully async execution (non-blocking in ComfyUI's event loop)
- aiohttp for HTTP operations (connection pooling, DNS cache)
- asyncio subprocess for ffprobe/ffmpeg (proper timeout and cleanup)
- Exponential backoff retry on transient network failures
- Timeout budget to cap total execution time
- Robust temp file cleanup with atexit fallback
"""

from __future__ import annotations

import asyncio
import atexit
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Tuple
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger("ComfyUI-Workflow-Tools")

# ---------------------------------------------------------------------------
# Shared aiohttp session (lazy, auto-recreated when event loop changes)
# ---------------------------------------------------------------------------

_session: aiohttp.ClientSession | None = None
_session_lock: asyncio.Lock | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session, _session_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                connector = aiohttp.TCPConnector(
                    limit=10,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                )
                _session = aiohttp.ClientSession(
                    connector=connector,
                    headers={"User-Agent": "ComfyUI-MetadataProbe/1.0"},
                )
    return _session


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_DEFAULT_RETRYABLE = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


async def _retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Tuple[type, ...] = _DEFAULT_RETRYABLE,
    operation_name: str = "operation",
    **kwargs: Any,
) -> Any:
    """Execute an async callable with exponential backoff retry."""
    delay = base_delay
    last_exception: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt == max_retries:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name, max_retries, exc,
                )
                raise
            logger.warning(
                "%s attempt %d/%d failed: %s — retrying in %.1fs",
                operation_name, attempt, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
            delay *= backoff_factor
    raise last_exception  # unreachable, satisfies type checker


# ---------------------------------------------------------------------------
# Async subprocess runner
# ---------------------------------------------------------------------------

async def _async_run_subprocess(
    cmd: list[str],
    timeout: float = 30.0,
    operation_name: str = "subprocess",
) -> Tuple[str, str]:
    """Run a subprocess asynchronously with escalated kill on timeout.

    Returns (stdout, stderr).  Raises RuntimeError on failure.
    """
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "%s timed out after %.1fs, terminating", operation_name, timeout,
            )
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "%s did not exit after SIGTERM, sending SIGKILL",
                    operation_name,
                )
                process.kill()
                await process.wait()
            raise RuntimeError(f"{operation_name} timed out after {timeout}s")

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            raise RuntimeError(
                f"{operation_name} exited with code {process.returncode}: "
                f"{stderr[:500]}"
            )
        return stdout, stderr

    except RuntimeError:
        raise
    except Exception as exc:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise RuntimeError(f"{operation_name} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Timeout budget
# ---------------------------------------------------------------------------

class TimeoutBudget:
    """Track remaining time budget across multiple sequential operations."""

    def __init__(self, total_seconds: float = 90.0):
        self.total = total_seconds
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed)

    @property
    def is_expired(self) -> bool:
        return self.remaining <= 0

    def allocate(self, requested: float, minimum: float = 5.0) -> float:
        """Return min(requested, remaining).  Raise TimeoutError if < minimum."""
        if self.remaining < minimum:
            raise TimeoutError(
                f"Timeout budget exhausted: {self.elapsed:.1f}s elapsed "
                f"of {self.total:.1f}s total, only {self.remaining:.1f}s left"
            )
        return min(requested, self.remaining)


# ---------------------------------------------------------------------------
# Temp file management
# ---------------------------------------------------------------------------

_active_temp_files: set[str] = set()


def _emergency_cleanup() -> None:
    for path in list(_active_temp_files):
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_emergency_cleanup)


class AsyncTempFile:
    """Async context manager for temp files with atexit fallback cleanup."""

    def __init__(self, suffix: str = "", prefix: str = "comfyui_probe_"):
        self.suffix = suffix
        self.prefix = prefix
        self.path: Path | None = None

    async def __aenter__(self) -> Path:
        fd, name = tempfile.mkstemp(suffix=self.suffix, prefix=self.prefix)
        os.close(fd)
        self.path = Path(name)
        _active_temp_files.add(str(self.path))
        return self.path

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self.path:
            _active_temp_files.discard(str(self.path))
            try:
                self.path.unlink(missing_ok=True)
            except OSError as e:
                logger.debug("Failed to clean up temp file %s: %s", self.path, e)
        return False


# ---------------------------------------------------------------------------
# High-level async operations
# ---------------------------------------------------------------------------

async def _async_ffprobe(url: str, timeout: float = 20.0) -> dict:
    """Run ffprobe on a URL and return parsed JSON, with retry."""
    logger.info("[ffprobe] start: url=%s, timeout=%.1fs", url, timeout)
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams",
        "-user_agent", "ComfyUI-MetadataProbe/1.0",
        url,
    ]

    async def _run() -> dict:
        stdout, _ = await _async_run_subprocess(
            cmd, timeout=timeout, operation_name="ffprobe",
        )
        return json.loads(stdout)

    result = await _retry_async(
        _run,
        max_retries=3,
        base_delay=2.0,
        retryable_exceptions=(RuntimeError, json.JSONDecodeError, OSError),
        operation_name="ffprobe",
    )
    fmt = result.get("format", {})
    streams = result.get("streams", [])
    logger.info(
        "[ffprobe] done: url=%s, format=%s, duration=%s, size=%s, streams=%d",
        url,
        fmt.get("format_name", "?"),
        fmt.get("duration", "?"),
        fmt.get("size", "?"),
        len(streams),
    )
    return result


async def _async_ffmpeg_extract_frame(
    url: str,
    output_path: Path,
    timeout: float = 20.0,
) -> bool:
    """Extract first frame from video URL.  Returns True on success."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-user_agent", "ComfyUI-MetadataProbe/1.0",
        "-i", url,
        "-vframes", "1",
        "-f", "image2",
        str(output_path),
    ]

    async def _run() -> bool:
        await _async_run_subprocess(
            cmd, timeout=timeout, operation_name="ffmpeg-frame",
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced empty or missing output")
        return True

    try:
        return await _retry_async(
            _run,
            max_retries=3,
            base_delay=2.0,
            retryable_exceptions=(RuntimeError, OSError),
            operation_name="ffmpeg-frame-extract",
        )
    except Exception as exc:
        logger.warning("Frame extraction failed after retries: %s", exc)
        return False


async def _async_download_to_temp(
    url: str,
    dest: Path,
    timeout: float = 30.0,
) -> None:
    """Download URL contents to *dest* using aiohttp (streaming)."""
    session = await _get_session()
    async with session.get(
        url,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                f.write(chunk)


async def _async_put_to_presigned_url(
    presign_url: str,
    data: bytes,
    content_type: str = "image/webp",
    timeout: float = 30.0,
) -> None:
    """Upload data to a presigned URL via HTTP PUT."""
    session = await _get_session()
    async with session.put(
        presign_url,
        data=data,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        },
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        if response.status not in (200, 201, 204):
            body = await response.text()
            raise RuntimeError(
                f"PUT failed with status {response.status}: {body[:200]}"
            )


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _get_resolution_label(height: int) -> str:
    if height >= 2160:
        return "4K"
    elif height >= 1080:
        return "1080p"
    elif height >= 720:
        return "720p"
    elif height >= 480:
        return "480p"
    else:
        return f"{height}p"


def _get_aspect_ratio(width: int, height: int) -> str:
    from math import gcd
    if width == 0 or height == 0:
        return "unknown"
    divisor = gcd(width, height)
    w = width // divisor
    h = height // divisor
    if (w, h) == (16, 9) or abs(width / height - 16 / 9) < 0.05:
        return "16:9"
    elif (w, h) == (9, 16) or abs(width / height - 9 / 16) < 0.05:
        return "9:16"
    elif (w, h) == (4, 3) or abs(width / height - 4 / 3) < 0.05:
        return "4:3"
    elif (w, h) == (3, 4) or abs(width / height - 3 / 4) < 0.05:
        return "3:4"
    elif (w, h) == (1, 1) or abs(width / height - 1) < 0.05:
        return "1:1"
    elif (w, h) == (21, 9) or abs(width / height - 21 / 9) < 0.05:
        return "21:9"
    else:
        return f"{w}:{h}"


def _scale_to_resolution(
    width: int, height: int, target_resolution: str,
) -> Tuple[int, int]:
    resolution_heights = {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "4K": 2160,
    }
    target_height = resolution_heights.get(target_resolution, 480)
    if height <= target_height:
        return width, height
    scale = target_height / height
    new_width = int(width * scale)
    new_height = target_height
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    return new_width, new_height


def _normalize_format(format_name: str, url: str = "") -> str:
    if not format_name or format_name == "unknown":
        if url:
            ext = Path(urlparse(url).path).suffix.lstrip(".").lower()
            if ext:
                return ext
        return "unknown"

    format_lower = format_name.lower()

    if any(f in format_lower for f in ["mp4", "mov", "m4a", "3gp", "3g2"]):
        return "mp4"
    if "mp3" in format_lower or "mpeg" in format_lower:
        return "mp3"
    if "wav" in format_lower:
        return "wav"
    if "flac" in format_lower:
        return "flac"
    if "ogg" in format_lower:
        return "ogg"
    if "aac" in format_lower:
        return "aac"
    if "webm" in format_lower:
        return "webm"
    if "avi" in format_lower:
        return "avi"
    if "mkv" in format_lower or "matroska" in format_lower:
        return "mkv"

    return format_name.split(",")[0].strip()


# ---------------------------------------------------------------------------
# Node: VideoProbeNode
# ---------------------------------------------------------------------------

class VideoProbeNode:
    DESCRIPTION = """
Extract video metadata and generate cover images.
- Inputs:
  - url: video URL to probe.
  - presign_info: JSON array of presigned upload info for covers.
    Format: [{"resolution": "480p", "presign_url": "...", "cdn_url": "..."}, ...]
  - webp_quality: WebP image quality (1-100, default 80).
- Outputs:
  - metadata_json: JSON string containing video metadata and cover URLs.
- Behavior:
  - Uses ffprobe/ffmpeg directly on URL (no full download needed).
  - Generates cover images from first frame.
  - Uploads covers to presigned URLs.
  - Returns comprehensive metadata including dimensions, duration, format, and cover URLs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "presign_info": ("STRING", {"default": "[]", "multiline": True}),
                "webp_quality": ("INT", {"default": 80, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata_json",)
    FUNCTION = "probe"
    CATEGORY = "AigcWorkflowTools/Metadata"
    OUTPUT_NODE = True

    async def probe(self, url: str, presign_info: str = "[]", webp_quality: int = 80):
        logger.info("[VideoProbe] input: url=%s, presign_info=%s, webp_quality=%d", url, presign_info[:100], webp_quality)
        if not url:
            raise ValueError("URL cannot be empty.")

        budget = TimeoutBudget(total_seconds=90.0)

        # ── Step 1: ffprobe (critical — failure = node failure) ──
        probe_data = await _async_ffprobe(url, timeout=budget.allocate(20.0))

        video_stream = None
        audio_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        if video_stream is None:
            raise RuntimeError("No video stream found")

        width = video_stream.get("width", 0)
        height = video_stream.get("height", 0)
        duration = float(probe_data.get("format", {}).get("duration", 0))
        size = int(probe_data.get("format", {}).get("size", 0))
        raw_format = probe_data.get("format", {}).get("format_name", "unknown")
        format_name = _normalize_format(raw_format, url)

        # ── Step 2: Cover generation (optional — failure = degraded output) ──
        covers: list[dict] = []
        presigns = json.loads(presign_info) if presign_info else []

        if presigns:
            try:
                from PIL import Image

                async with AsyncTempFile(suffix=".png") as frame_path:
                    ffmpeg_timeout = budget.allocate(20.0, minimum=5.0)
                    success = await _async_ffmpeg_extract_frame(
                        url, frame_path, timeout=ffmpeg_timeout,
                    )
                    if success:
                        img = Image.open(frame_path)
                        try:
                            for presign in presigns:
                                resolution = presign.get("resolution", "480p")
                                presign_url = presign.get("presign_url", "")
                                cdn_url = presign.get("cdn_url", "")
                                if not presign_url or not cdn_url:
                                    continue

                                new_w, new_h = _scale_to_resolution(
                                    width, height, resolution,
                                )
                                resized = img.resize(
                                    (new_w, new_h), Image.Resampling.LANCZOS,
                                )
                                buffer = io.BytesIO()
                                resized.save(
                                    buffer, format="WEBP", quality=webp_quality,
                                )
                                webp_data = buffer.getvalue()

                                try:
                                    upload_timeout = budget.allocate(
                                        15.0, minimum=3.0,
                                    )
                                    await _retry_async(
                                        _async_put_to_presigned_url,
                                        presign_url,
                                        webp_data,
                                        timeout=upload_timeout,
                                        max_retries=3,
                                        base_delay=1.0,
                                        operation_name=f"cover-upload-{resolution}",
                                    )
                                    covers.append({
                                        "url": cdn_url,
                                        "resolution": resolution,
                                        "width": new_w,
                                        "height": new_h,
                                    })
                                except Exception as exc:
                                    logger.warning(
                                        "Cover upload failed for %s: %s",
                                        resolution, exc,
                                    )
                        finally:
                            img.close()
                    else:
                        logger.warning(
                            "Frame extraction returned no usable frame for %s",
                            url,
                        )
            except TimeoutError as exc:
                logger.warning(
                    "Timeout budget exhausted during cover generation: %s", exc,
                )
            except Exception as exc:
                logger.warning(
                    "Cover generation failed for %s: %s", url, exc, exc_info=True,
                )

        metadata = {
            "width": width,
            "height": height,
            "size": size,
            "duration": duration,
            "has_audio": audio_stream is not None,
            "format": format_name,
            "resolution": _get_resolution_label(height),
            "aspect_ratio": _get_aspect_ratio(width, height),
            "covers": covers,
            "extra": {},
        }
        metadata_json = json.dumps(metadata)
        logger.info("[VideoProbe] output: %s", metadata_json)
        return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}


# ---------------------------------------------------------------------------
# Node: ImageProbeNode
# ---------------------------------------------------------------------------

class ImageProbeNode:
    DESCRIPTION = """
Extract image metadata and generate thumbnails.
- Inputs:
  - url: image URL to probe.
  - presign_info: JSON array of presigned upload info for thumbnails.
    Format: [{"resolution": "480p", "presign_url": "...", "cdn_url": "..."}, ...]
  - webp_quality: WebP image quality (1-100, default 85).
- Outputs:
  - metadata_json: JSON string containing image metadata and thumbnail URLs.
- Behavior:
  - Downloads image, extracts metadata using PIL.
  - Generates thumbnails at specified resolutions.
  - Uploads thumbnails to presigned URLs.
  - Returns comprehensive metadata including dimensions, format, and thumbnail URLs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "presign_info": ("STRING", {"default": "[]", "multiline": True}),
                "webp_quality": ("INT", {"default": 85, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata_json",)
    FUNCTION = "probe"
    CATEGORY = "AigcWorkflowTools/Metadata"
    OUTPUT_NODE = True

    async def probe(self, url: str, presign_info: str = "[]", webp_quality: int = 85):
        logger.info("[ImageProbe] input: url=%s, presign_info=%s, webp_quality=%d", url, presign_info[:100], webp_quality)
        if not url:
            raise ValueError("URL cannot be empty.")

        budget = TimeoutBudget(total_seconds=90.0)
        presigns = json.loads(presign_info) if presign_info else []

        if not presigns:
            # ── Fast path: metadata only via ffprobe, no download ──
            probe_data = await _async_ffprobe(url, timeout=budget.allocate(20.0))

            image_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    image_stream = stream
                    break

            if image_stream is None:
                raise RuntimeError("No image stream found")

            width = image_stream.get("width", 0)
            height = image_stream.get("height", 0)
            size = int(probe_data.get("format", {}).get("size", 0))
            raw_format = probe_data.get("format", {}).get("format_name", "unknown")
            format_name = _normalize_format(raw_format, url)

            metadata = {
                "width": width,
                "height": height,
                "size": size,
                "format": format_name,
                "resolution": _get_resolution_label(height),
                "aspect_ratio": _get_aspect_ratio(width, height),
                "thumbnails": [],
                "extra": {},
            }
            metadata_json = json.dumps(metadata)
            logger.info("[ImageProbe] output: %s", metadata_json)
            return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}

        # ── Full path: download needed for thumbnail generation ──
        from PIL import Image

        async with AsyncTempFile(suffix=_guess_suffix(url)) as tmp_path:
            await _retry_async(
                _async_download_to_temp,
                url,
                tmp_path,
                timeout=budget.allocate(30.0),
                max_retries=3,
                base_delay=1.0,
                operation_name="image-download",
            )

            size = os.path.getsize(tmp_path)
            img = Image.open(tmp_path)
            width, height = img.size
            format_name = img.format.lower() if img.format else "unknown"

            thumbnails: list[dict] = []
            for presign in presigns:
                resolution = presign.get("resolution", "480p")
                presign_url = presign.get("presign_url", "")
                cdn_url = presign.get("cdn_url", "")
                if not presign_url or not cdn_url:
                    continue

                try:
                    new_w, new_h = _scale_to_resolution(width, height, resolution)
                    resized = img.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS,
                    )
                    buffer = io.BytesIO()
                    resized.save(buffer, format="WEBP", quality=webp_quality)
                    webp_data = buffer.getvalue()

                    upload_timeout = budget.allocate(15.0, minimum=3.0)
                    await _retry_async(
                        _async_put_to_presigned_url,
                        presign_url,
                        webp_data,
                        timeout=upload_timeout,
                        max_retries=3,
                        base_delay=1.0,
                        operation_name=f"thumbnail-upload-{resolution}",
                    )
                    thumbnails.append({
                        "url": cdn_url,
                        "resolution": resolution,
                        "width": new_w,
                        "height": new_h,
                    })
                except Exception as exc:
                    logger.warning(
                        "Thumbnail generation/upload failed for %s: %s",
                        resolution, exc,
                    )
                    continue

            img.close()

        metadata = {
            "width": width,
            "height": height,
            "size": size,
            "format": format_name,
            "resolution": _get_resolution_label(height),
            "aspect_ratio": _get_aspect_ratio(width, height),
            "thumbnails": thumbnails,
            "extra": {},
        }
        metadata_json = json.dumps(metadata)
        logger.info("[ImageProbe] output: %s", metadata_json)
        return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}


# ---------------------------------------------------------------------------
# Node: AudioProbeNode
# ---------------------------------------------------------------------------

class AudioProbeNode:
    DESCRIPTION = """
Extract audio metadata.
- Inputs:
  - url: audio URL to probe.
- Outputs:
  - metadata_json: JSON string containing audio metadata.
- Behavior:
  - Uses ffprobe directly on URL (no full download needed).
  - Returns metadata including duration, format, and file size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata_json",)
    FUNCTION = "probe"
    CATEGORY = "AigcWorkflowTools/Metadata"
    OUTPUT_NODE = True

    async def probe(self, url: str):
        logger.info("[AudioProbe] input: url=%s", url)
        if not url:
            raise ValueError("URL cannot be empty.")

        budget = TimeoutBudget(total_seconds=60.0)

        probe_data = await _async_ffprobe(url, timeout=budget.allocate(20.0))

        audio_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        duration = float(probe_data.get("format", {}).get("duration", 0))
        size = int(probe_data.get("format", {}).get("size", 0))
        raw_format = probe_data.get("format", {}).get("format_name", "unknown")
        format_name = _normalize_format(raw_format, url)

        metadata = {
            "duration": duration,
            "format": format_name,
            "size": size,
            "word_count": 0,
            "extra": {},
        }
        metadata_json = json.dumps(metadata)
        logger.info("[AudioProbe] output: %s", metadata_json)
        return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _guess_suffix(url: str) -> str:
    """Guess file suffix from URL path."""
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix if suffix else ".img"
