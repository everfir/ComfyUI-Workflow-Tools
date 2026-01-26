import contextlib
import io
import mimetypes
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


def _sanitize_filename(name: str) -> str:
    """Keep only filesystem-friendly characters."""
    cleaned = name.strip().replace("\x00", "")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned or f"file-{uuid.uuid4().hex}"


def _filename_from_headers(content_disposition: str) -> str | None:
    if not content_disposition:
        return None

    match = re.search(r'filename\\*?=\\s*"?([^";]+)"?', content_disposition, re.IGNORECASE)
    if not match:
        return None

    value = match.group(1)
    if "''" in value:
        _, value = value.split("''", 1)
    return _sanitize_filename(unquote(value))


def _infer_filename(url: str, response) -> str:
    from_header = _filename_from_headers(response.headers.get("Content-Disposition"))
    if from_header:
        return from_header

    url_name = Path(unquote(urlparse(url).path)).name
    if url_name:
        sanitized = _sanitize_filename(url_name)
        if sanitized:
            return sanitized

    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip()
    guessed_ext = mimetypes.guess_extension(content_type) if content_type else None
    extension = guessed_ext or ".bin"
    return f"download-{uuid.uuid4().hex}{extension}"


def _unique_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    base = directory / filename
    if not base.exists():
        return base

    stem, suffix = base.stem, base.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class DownloadURL:
    DESCRIPTION = """
Download an HTTP/HTTPS URL to local disk.
- Inputs:
  - url: target link to fetch.
  - save_dir: output directory (default: output, relative to ComfyUI working dir).
- Behavior:
  - Tries filename from Content-Disposition, then URL path; otherwise generates one.
  - Sanitizes filename and avoids overwrite by adding numeric suffixes.
  - Streams download to disk with a custom User-Agent; stdlib only, no extra deps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "save_dir": ("STRING", {"default": "output", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "download"
    CATEGORY = "AigcWorkflowTools"

    def download(self, url: str, save_dir: str):
        if not url:
            raise ValueError("URL cannot be empty.")

        scheme = urlparse(url).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ValueError("Only http/https URLs are supported.")

        target_dir = Path(save_dir).expanduser()
        if not target_dir.is_absolute():
            target_dir = Path.cwd() / target_dir

        request = Request(url, headers={"User-Agent": "ComfyUI-URL-Downloader/1.0"})
        try:
            with contextlib.closing(urlopen(request, timeout=60)) as response:
                filename = _infer_filename(url, response)
                destination = _unique_path(target_dir, filename)
                with open(destination, "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP error {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach URL: {exc.reason}") from exc

        return (str(destination),)


NODE_CLASS_MAPPINGS = {"DownloadURL": DownloadURL}
NODE_DISPLAY_NAME_MAPPINGS = {"DownloadURL": "Download URL"}


class ExtractFileInfo:
    DESCRIPTION = """
Extract file name and type from a file path.
- Inputs:
  - file_path: path to the file (absolute or relative).
- Outputs:
  - file_name: base name of the file.
  - file_type: extension without leading dot (empty if none).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_name", "file_type")
    FUNCTION = "extract"
    CATEGORY = "AigcWorkflowTools"

    def extract(self, file_path: str):
        if not file_path:
            raise ValueError("file_path cannot be empty.")

        path = Path(file_path)
        name = path.name
        suffix = path.suffix.lstrip(".")
        return (name, suffix)


NODE_CLASS_MAPPINGS.update({"ExtractFileInfo": ExtractFileInfo})
NODE_DISPLAY_NAME_MAPPINGS.update({"ExtractFileInfo": "Extract File Info"})


class APITextOutput:
    DESCRIPTION = """
Output node to send plain text back to ComfyUI UI/API responses.
- Inputs:
  - text: string content to return.
- Outputs:
  - text: same string, also attached to UI history (ui.text).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "notify_api"
    CATEGORY = "AigcWorkflowTools"
    OUTPUT_NODE = True

    def notify_api(self, text: str):
        # Keep text wrapped in a list to match ComfyUI's expected UI payload shape.
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS.update({"APITextOutput": APITextOutput})
NODE_DISPLAY_NAME_MAPPINGS.update({"APITextOutput": "API Text Output"})


def _tos_client(ak: str, sk: str, region: str, endpoint: str | None = None):
    try:
        from tos import TosClientV2
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: tos. Install with `pip install tos`.") from exc

    ep = endpoint or f"https://tos-{region}.volces.com"
    # Use keyword args to avoid positional/keyword conflicts across SDK versions
    return TosClientV2(ak, sk, region=region, endpoint=ep), ep


def _public_url(endpoint: str, bucket: str, object_key: str) -> str:
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "https"
    host = parsed.netloc or endpoint.replace("https://", "").replace("http://", "")
    return f"{scheme}://{bucket}.{host}/{object_key.lstrip('/')}"


def _put_object_compat(client, bucket: str, object_key: str, data: bytes, content_type: str | None = None):
    """Try multiple call signatures to adapt to different tos SDK versions."""
    attempts = [
        {"args": (bucket, object_key), "kwargs": {"data": data, "content_type": content_type}},
        {"args": (bucket, object_key), "kwargs": {"content": data, "content_type": content_type}},
        {"args": (bucket, object_key, data), "kwargs": {"content_type": content_type}},
        {"args": (bucket, object_key), "kwargs": {"body": data, "content_type": content_type}},
        {"args": (bucket, object_key, data), "kwargs": {}},
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return client.put_object(*attempt["args"], **{k: v for k, v in attempt["kwargs"].items() if v is not None})
        except TypeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("put_object failed without TypeError.")


class LoadImageFromPath:
    DESCRIPTION = """
Load an image file from disk and output an IMAGE tensor (B,H,W,C) in 0-1 float.
- Inputs:
  - file_path: image path (absolute or relative).
- Outputs:
  - image: IMAGE tensor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "AigcWorkflowTools"

    def load(self, file_path: str):
        if not file_path:
            raise ValueError("file_path cannot be empty.")

        from PIL import Image
        import numpy as np
        import torch

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        return (tensor,)


class LoadVideoFromPath:
    DESCRIPTION = """
Load a video file from disk and output frames + fps.
- Inputs:
  - file_path: video path.
- Outputs:
  - video: dict with frames tensor (T,H,W,C) in 0-1 float and fps.
  - fps: float frames per second.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("VIDEO", "FLOAT")
    RETURN_NAMES = ("video", "fps")
    FUNCTION = "load"
    CATEGORY = "AigcWorkflowTools"

    def load(self, file_path: str):
        if not file_path:
            raise ValueError("file_path cannot be empty.")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        try:
            import imageio.v2 as imageio
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("LoadVideoFromPath requires imageio and torch.") from exc

        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 24.0))
        frames = []
        for frame in reader:
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        if not frames:
            raise RuntimeError("No frames read from video.")
        stacked = torch.stack(frames, dim=0)
        video = {"frames": stacked, "fps": fps}
        return (video, fps)


class LoadAudioFromPath:
    DESCRIPTION = """
Load an audio file from disk and output waveform + sample rate.
- Inputs:
  - file_path: audio path.
- Outputs:
  - audio: waveform tensor (channels, samples).
  - sample_rate: integer sample rate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "sample_rate")
    FUNCTION = "load"
    CATEGORY = "AigcWorkflowTools"

    def load(self, file_path: str):
        if not file_path:
            raise ValueError("file_path cannot be empty.")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            import torchaudio
        except ModuleNotFoundError as exc:
            raise RuntimeError("LoadAudioFromPath requires torchaudio.") from exc

        waveform, sample_rate = torchaudio.load(path)
        return (waveform, int(sample_rate))


NODE_CLASS_MAPPINGS.update(
    {
        "LoadImageFromPath": LoadImageFromPath,
        "LoadVideoFromPath": LoadVideoFromPath,
        "LoadAudioFromPath": LoadAudioFromPath,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "LoadImageFromPath": "Load Image From Path",
        "LoadVideoFromPath": "Load Video From Path",
        "LoadAudioFromPath": "Load Audio From Path",
    }
)


class UploadImageToTOS:
    DESCRIPTION = """
Upload an IMAGE tensor to Volcengine TOS and return the public/object URL.
- Inputs:
  - image: IMAGE tensor (B,H,W,C in 0-1 float).
  - ak / sk / region / bucket: TOS credentials.
  - upload_dir: object key prefix (e.g., uploads/images); default creates a UUID name.
  - endpoint: optional custom endpoint (default: https://tos-{region}.volces.com).
- Outputs:
  - url: public-style URL (requires bucket/object ACL to be readable).
  - object_key: key used in the bucket.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ak": ("STRING", {"default": "", "multiline": False}),
                "sk": ("STRING", {"default": "", "multiline": False}),
                "region": ("STRING", {"default": "", "multiline": False}),
                "bucket": ("STRING", {"default": "", "multiline": False}),
                "upload_dir": ("STRING", {"default": "uploads/images", "multiline": False}),
            },
            "optional": {"endpoint": ("STRING", {"default": "", "multiline": False})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "object_key")
    FUNCTION = "upload"
    CATEGORY = "AigcWorkflowTools"

    def upload(self, image, ak: str, sk: str, region: str, bucket: str, upload_dir: str, endpoint: str = ""):
        if image is None:
            raise ValueError("image cannot be empty.")
        if not all([ak, sk, region, bucket]):
            raise ValueError("ak, sk, region, bucket are required.")

        try:
            import torch
            from PIL import Image
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("UploadImageToTOS requires torch and pillow.") from exc

        arr = image
        if isinstance(image, torch.Tensor):
            arr = image
        else:
            raise TypeError("image must be a torch.Tensor.")

        arr = arr[0] if arr.dim() == 4 else arr
        arr = arr.clamp(0, 1).cpu()
        np_img = (arr.numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        data = buffer.read()

        object_key = f"{upload_dir.strip('/')}/{uuid.uuid4().hex}.png"
        client, ep = _tos_client(ak, sk, region, endpoint or None)
        _put_object_compat(client, bucket, object_key, data=data, content_type="image/png")
        url = _public_url(ep, bucket, object_key)
        return (url, object_key)


class UploadVideoToTOS:
    DESCRIPTION = """
Upload a VIDEO dict (frames + fps) to Volcengine TOS as MP4 and return the URL.
- Inputs:
  - video: dict with frames tensor (T,H,W,C, 0-1 float) and fps.
  - ak / sk / region / bucket: TOS credentials.
  - upload_dir: object key prefix; default creates UUID name.
  - endpoint: optional custom endpoint.
- Outputs:
  - url: public-style URL (requires bucket/object ACL to be readable).
  - object_key: key used in the bucket.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "ak": ("STRING", {"default": "", "multiline": False}),
                "sk": ("STRING", {"default": "", "multiline": False}),
                "region": ("STRING", {"default": "", "multiline": False}),
                "bucket": ("STRING", {"default": "", "multiline": False}),
                "upload_dir": ("STRING", {"default": "uploads/videos", "multiline": False}),
            },
            "optional": {"endpoint": ("STRING", {"default": "", "multiline": False})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "object_key")
    FUNCTION = "upload"
    CATEGORY = "AigcWorkflowTools"

    def upload(self, video, ak: str, sk: str, region: str, bucket: str, upload_dir: str, endpoint: str = ""):
        if video is None:
            raise ValueError("video cannot be empty.")
        if not all([ak, sk, region, bucket]):
            raise ValueError("ak, sk, region, bucket are required.")
        try:
            import torch
            import numpy as np
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise RuntimeError("UploadVideoToTOS requires torch, numpy, and imageio (with ffmpeg).") from exc

        frames = video.get("frames")
        fps = float(video.get("fps", 24.0))
        if frames is None:
            raise ValueError("video.frames is missing.")
        if not isinstance(frames, torch.Tensor):
            raise TypeError("video.frames must be a torch.Tensor.")

        arr = frames.clamp(0, 1).cpu().numpy()
        arr_uint8 = (arr * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with imageio.get_writer(tmp_path, fps=fps) as writer:
                for frame in arr_uint8:
                    writer.append_data(frame)

            with open(tmp_path, "rb") as f:
                data = f.read()
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

        object_key = f"{upload_dir.strip('/')}/{uuid.uuid4().hex}.mp4"
        client, ep = _tos_client(ak, sk, region, endpoint or None)
        _put_object_compat(client, bucket, object_key, data=data, content_type="video/mp4")
        url = _public_url(ep, bucket, object_key)
        return (url, object_key)


class UploadAudioToTOS:
    DESCRIPTION = """
Upload an AUDIO tensor to Volcengine TOS as WAV and return the URL.
- Inputs:
  - audio: waveform tensor (channels, samples).
  - sample_rate: integer sample rate.
  - ak / sk / region / bucket: TOS credentials.
  - upload_dir: object key prefix; default creates UUID name.
  - endpoint: optional custom endpoint.
- Outputs:
  - url: public-style URL (requires bucket/object ACL to be readable).
  - object_key: key used in the bucket.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "min": 1, "max": 96000}),
                "ak": ("STRING", {"default": "", "multiline": False}),
                "sk": ("STRING", {"default": "", "multiline": False}),
                "region": ("STRING", {"default": "", "multiline": False}),
                "bucket": ("STRING", {"default": "", "multiline": False}),
                "upload_dir": ("STRING", {"default": "uploads/audios", "multiline": False}),
            },
            "optional": {"endpoint": ("STRING", {"default": "", "multiline": False})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "object_key")
    FUNCTION = "upload"
    CATEGORY = "AigcWorkflowTools"

    def upload(self, audio, sample_rate: int, ak: str, sk: str, region: str, bucket: str, upload_dir: str, endpoint: str = ""):
        if audio is None:
            raise ValueError("audio cannot be empty.")
        if not all([ak, sk, region, bucket]):
            raise ValueError("ak, sk, region, bucket are required.")

        try:
            import torch
            import torchaudio
        except ModuleNotFoundError as exc:
            raise RuntimeError("UploadAudioToTOS requires torch and torchaudio.") from exc

        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torchaudio.save(tmp_path, audio.cpu(), sample_rate=sample_rate)
            with open(tmp_path, "rb") as f:
                data = f.read()
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

        object_key = f"{upload_dir.strip('/')}/{uuid.uuid4().hex}.wav"
        client, ep = _tos_client(ak, sk, region, endpoint or None)
        _put_object_compat(client, bucket, object_key, data=data, content_type="audio/wav")
        url = _public_url(ep, bucket, object_key)
        return (url, object_key)


NODE_CLASS_MAPPINGS.update(
    {
        "UploadImageToTOS": UploadImageToTOS,
        "UploadVideoToTOS": UploadVideoToTOS,
        "UploadAudioToTOS": UploadAudioToTOS,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "UploadImageToTOS": "Upload Image To TOS",
        "UploadVideoToTOS": "Upload Video To TOS",
        "UploadAudioToTOS": "Upload Audio To TOS",
    }
)


class FFmpegExecutor:
    DESCRIPTION = """
Execute FFmpeg command and return the output file path.
- Inputs:
  - command: FFmpeg command WITHOUT output file (e.g., "ffmpeg -i input.mp4 -vf scale=1280:720").
  - output_dir: directory for output files (default: output, relative to ComfyUI working dir).
  - output_filename: output filename without extension (default: UUID).
  - output_extension: output file extension (default: .mp4).
- Outputs:
  - file_path: absolute path to the generated output file.
- Behavior:
  - Appends the full output path to the FFmpeg command.
  - Creates output directory if it doesn't exist.
  - Executes FFmpeg with 10-minute timeout.
  - Verifies output file exists and returns absolute path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "command": ("STRING", {"default": "", "multiline": True}),
                "output_dir": ("STRING", {"default": "output", "multiline": False}),
                "output_extension": ("STRING", {"default": ".mp4", "multiline": False}),
            },
            "optional": {
                "output_filename": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "execute"
    CATEGORY = "AigcWorkflowTools"

    def execute(self, command: str, output_dir: str, output_extension: str, output_filename: str = ""):
        if not command:
            raise ValueError("FFmpeg command cannot be empty.")

        import subprocess
        import shlex

        # Generate output filename if not provided
        if not output_filename:
            output_filename = uuid.uuid4().hex

        # Ensure extension starts with dot
        if output_extension and not output_extension.startswith('.'):
            output_extension = '.' + output_extension

        # Build full output path
        target_dir = Path(output_dir).expanduser()
        if not target_dir.is_absolute():
            target_dir = Path.cwd() / target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        output_path = target_dir / f"{output_filename}{output_extension}"

        # Append output path to command
        full_command = f"{command} {shlex.quote(str(output_path))}"

        # Execute the FFmpeg command
        try:
            result = subprocess.run(
                full_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr if exc.stderr else exc.stdout
            raise RuntimeError(f"FFmpeg command failed with code {exc.returncode}: {error_msg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FFmpeg command timed out after 10 minutes.") from exc

        # Verify output file exists
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not found after FFmpeg execution: {output_path}")

        return (str(output_path),)


NODE_CLASS_MAPPINGS.update({"FFmpegExecutor": FFmpegExecutor})
NODE_DISPLAY_NAME_MAPPINGS.update({"FFmpegExecutor": "FFmpeg Executor"})


class UploadFileToTOS:
    DESCRIPTION = """
Upload various file types to Volcengine TOS and return the public URL.
Supports direct file upload from FFmpegExecutor output or ComfyUI media types.

- Inputs (required):
  - ak / sk / region / bucket: TOS credentials.
  - upload_dir: object key prefix (e.g., "uploads/videos").

- Inputs (optional, at least one):
  - file_path: STRING path to local file (can connect from FFmpegExecutor.file_path output).
  - image: IMAGE tensor (B,H,W,C in 0-1 float).
  - video: VIDEO dict with frames and fps.
  - audio: AUDIO waveform tensor (channels, samples).
  - sample_rate: INT sample rate (required when audio is provided).

- Inputs (optional config):
  - endpoint: custom endpoint (default: https://tos-{region}.volces.com).
  - acl: access control (public-read/private/public-read-write).
  - storage_class: storage type (STANDARD/IA/ARCHIVE_FR).
  - content_type: custom Content-Type (auto-detected if empty).
  - custom_filename: custom filename without extension (uses UUID if empty).

- Outputs:
  - url: public-style URL (requires bucket/object ACL to be readable).
  - object_key: key used in the bucket.

- Behavior:
  - Priority: file_path > image > video > audio
  - Auto-detects content type and extension based on input type
  - Supports ACL and storage class configuration
  - Returns both public URL and object key

- Usage Example:
  FFmpegExecutor.file_path -> UploadFileToTOS.file_path -> url
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ak": ("STRING", {"default": "", "multiline": False}),
                "sk": ("STRING", {"default": "", "multiline": False}),
                "region": ("STRING", {"default": "", "multiline": False}),
                "bucket": ("STRING", {"default": "", "multiline": False}),
                "upload_dir": ("STRING", {"default": "uploads", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "min": 1, "max": 96000}),
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "endpoint": ("STRING", {"default": "", "multiline": False}),
                "acl": (["", "public-read", "private", "public-read-write"], {"default": ""}),
                "storage_class": (["", "STANDARD", "IA", "ARCHIVE_FR"], {"default": ""}),
                "content_type": ("STRING", {"default": "", "multiline": False}),
                "custom_filename": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "object_key")
    FUNCTION = "upload"
    CATEGORY = "AigcWorkflowTools"

    def upload(self, ak: str, sk: str, region: str, bucket: str, upload_dir: str,
               image=None, video=None, audio=None, sample_rate: int = 16000,
               file_path: str = "", endpoint: str = "", acl: str = "",
               storage_class: str = "", content_type: str = "", custom_filename: str = ""):

        if not all([ak, sk, region, bucket]):
            raise ValueError("ak, sk, region, bucket are required.")

        # Determine input type and prepare data
        data, detected_content_type, extension = self._prepare_data(
            image, video, audio, sample_rate, file_path
        )

        # Use custom content_type if provided, otherwise use detected
        final_content_type = content_type or detected_content_type

        # Generate object key
        filename = custom_filename or uuid.uuid4().hex
        object_key = f"{upload_dir.strip('/')}/{filename}{extension}"

        # Create TOS client
        client, ep = _tos_client(ak, sk, region, endpoint or None)

        # Prepare upload kwargs
        upload_kwargs = {"data": data, "content_type": final_content_type}

        # Add optional parameters if provided
        if acl:
            upload_kwargs["acl"] = acl
        if storage_class:
            upload_kwargs["storage_class"] = storage_class

        # Upload to TOS using compatibility wrapper
        try:
            _put_object_compat(client, bucket, object_key, **upload_kwargs)
        except Exception as exc:
            # Try setting ACL separately if it failed during upload
            if acl:
                try:
                    _put_object_compat(client, bucket, object_key,
                                     data=data, content_type=final_content_type)
                    # Set ACL separately (if SDK supports it)
                    with contextlib.suppress(Exception):
                        client.put_object_acl(bucket, object_key, acl=acl)
                except Exception:
                    raise exc
            else:
                raise exc

        # Generate public URL
        url = _public_url(ep, bucket, object_key)
        return (url, object_key)

    def _prepare_data(self, image, video, audio, sample_rate: int, file_path: str):
        """Prepare upload data based on input type. Returns (data, content_type, extension)."""

        # Priority 1: file_path
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "rb") as f:
                data = f.read()

            # Auto-detect content type
            content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            extension = path.suffix or ".bin"
            return data, content_type, extension

        # Priority 2: image
        if image is not None:
            try:
                import torch
                from PIL import Image
                import numpy as np
            except ModuleNotFoundError as exc:
                raise RuntimeError("Image upload requires torch and pillow.") from exc

            arr = image
            if isinstance(image, torch.Tensor):
                arr = image
            else:
                raise TypeError("image must be a torch.Tensor.")

            arr = arr[0] if arr.dim() == 4 else arr
            arr = arr.clamp(0, 1).cpu()
            np_img = (arr.numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            data = buffer.read()

            return data, "image/png", ".png"

        # Priority 3: video
        if video is not None:
            try:
                import torch
                import numpy as np
                import imageio.v2 as imageio
            except ModuleNotFoundError as exc:
                raise RuntimeError("Video upload requires torch, numpy, and imageio (with ffmpeg).") from exc

            frames = video.get("frames")
            fps = float(video.get("fps", 24.0))
            if frames is None:
                raise ValueError("video.frames is missing.")
            if not isinstance(frames, torch.Tensor):
                raise TypeError("video.frames must be a torch.Tensor.")

            arr = frames.clamp(0, 1).cpu().numpy()
            arr_uint8 = (arr * 255).astype(np.uint8)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with imageio.get_writer(tmp_path, fps=fps) as writer:
                    for frame in arr_uint8:
                        writer.append_data(frame)

                with open(tmp_path, "rb") as f:
                    data = f.read()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()

            return data, "video/mp4", ".mp4"

        # Priority 4: audio
        if audio is not None:
            try:
                import torch
                import torchaudio
            except ModuleNotFoundError as exc:
                raise RuntimeError("Audio upload requires torch and torchaudio.") from exc

            if not isinstance(audio, torch.Tensor):
                raise TypeError("audio must be a torch.Tensor.")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                torchaudio.save(tmp_path, audio.cpu(), sample_rate=sample_rate)
                with open(tmp_path, "rb") as f:
                    data = f.read()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()

            return data, "audio/wav", ".wav"

        raise ValueError("At least one input (image, video, audio, or file_path) must be provided.")


NODE_CLASS_MAPPINGS.update({"UploadFileToTOS": UploadFileToTOS})
NODE_DISPLAY_NAME_MAPPINGS.update({"UploadFileToTOS": "Upload File To TOS"})


# =============================================================================
# Metadata Probe Nodes
# =============================================================================

def _download_to_temp(url: str, suffix: str = "") -> Path:
    """Download URL to a temporary file and return the path."""
    request = Request(url, headers={"User-Agent": "ComfyUI-MetadataProbe/1.0"})
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    with contextlib.closing(urlopen(request, timeout=120)) as response:
        with open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    return tmp_path


def _get_resolution_label(height: int) -> str:
    """Get resolution label based on height."""
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
    """Get aspect ratio string."""
    from math import gcd
    if width == 0 or height == 0:
        return "unknown"
    divisor = gcd(width, height)
    w = width // divisor
    h = height // divisor
    # Common ratios
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


def _put_to_presigned_url(presign_url: str, data: bytes, content_type: str = "image/webp"):
    """Upload data to a presigned URL using HTTP PUT."""
    from urllib.request import Request, urlopen
    request = Request(presign_url, data=data, method="PUT")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(data)))
    with urlopen(request, timeout=60) as response:
        if response.status not in (200, 201, 204):
            raise RuntimeError(f"PUT failed with status {response.status}")


def _scale_to_resolution(width: int, height: int, target_resolution: str) -> tuple[int, int]:
    """Scale dimensions to target resolution while maintaining aspect ratio."""
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
    # Ensure even dimensions for video encoding
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    return new_width, new_height


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
  - Downloads video, extracts metadata using ffprobe.
  - Generates cover images from middle frame.
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

    def probe(self, url: str, presign_info: str = "[]", webp_quality: int = 80):
        import json
        import subprocess
        import os

        if not url:
            raise ValueError("URL cannot be empty.")

        # Download video to temp file
        tmp_path = _download_to_temp(url, suffix=".mp4")
        try:
            # Extract metadata using ffprobe
            ffprobe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(tmp_path)
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            probe_data = json.loads(result.stdout)

            # Extract video stream info
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
            format_name = probe_data.get("format", {}).get("format_name", "unknown")

            # Generate covers if presign_info is provided
            covers = []
            presigns = json.loads(presign_info) if presign_info else []

            if presigns:
                try:
                    from PIL import Image

                    # Extract frame from middle of video
                    middle_time = duration / 2 if duration > 0 else 0
                    frame_path = tmp_path.with_suffix(".png")

                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-ss", str(middle_time),
                        "-i", str(tmp_path), "-vframes", "1",
                        "-f", "image2", str(frame_path)
                    ]
                    subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)

                    if frame_path.exists():
                        img = Image.open(frame_path)

                        for presign in presigns:
                            resolution = presign.get("resolution", "480p")
                            presign_url = presign.get("presign_url", "")
                            cdn_url = presign.get("cdn_url", "")

                            if not presign_url or not cdn_url:
                                continue

                            # Scale image
                            new_w, new_h = _scale_to_resolution(width, height, resolution)
                            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                            # Convert to WebP
                            buffer = io.BytesIO()
                            resized.save(buffer, format="WEBP", quality=webp_quality)
                            buffer.seek(0)
                            webp_data = buffer.read()

                            # Upload to presigned URL
                            _put_to_presigned_url(presign_url, webp_data)

                            covers.append({
                                "url": cdn_url,
                                "resolution": resolution,
                                "width": new_w,
                                "height": new_h,
                            })

                        frame_path.unlink(missing_ok=True)
                except Exception as e:
                    # Cover generation failed, but we still have metadata
                    pass

            # Build metadata
            metadata = {
                "width": width,
                "height": height,
                "size": size,
                "duration": duration,
                "has_audio": audio_stream is not None,
                "format": format_name.split(",")[0] if format_name else "unknown",
                "resolution": _get_resolution_label(height),
                "aspect_ratio": _get_aspect_ratio(width, height),
                "covers": covers,
                "extra": {},
            }

            metadata_json = json.dumps(metadata)
            return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}

        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()


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

    def probe(self, url: str, presign_info: str = "[]", webp_quality: int = 85):
        import json
        import os

        if not url:
            raise ValueError("URL cannot be empty.")

        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("ImageProbeNode requires pillow.") from exc

        # Download image to temp file
        tmp_path = _download_to_temp(url)
        try:
            # Get file size
            size = os.path.getsize(tmp_path)

            # Open image and extract metadata
            img = Image.open(tmp_path)
            width, height = img.size
            format_name = img.format.lower() if img.format else "unknown"

            # Generate thumbnails if presign_info is provided
            thumbnails = []
            presigns = json.loads(presign_info) if presign_info else []

            for presign in presigns:
                resolution = presign.get("resolution", "480p")
                presign_url = presign.get("presign_url", "")
                cdn_url = presign.get("cdn_url", "")

                if not presign_url or not cdn_url:
                    continue

                try:
                    # Scale image
                    new_w, new_h = _scale_to_resolution(width, height, resolution)
                    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # Convert to WebP
                    buffer = io.BytesIO()
                    resized.save(buffer, format="WEBP", quality=webp_quality)
                    buffer.seek(0)
                    webp_data = buffer.read()

                    # Upload to presigned URL
                    _put_to_presigned_url(presign_url, webp_data)

                    thumbnails.append({
                        "url": cdn_url,
                        "resolution": resolution,
                        "width": new_w,
                        "height": new_h,
                    })
                except Exception:
                    # Thumbnail generation failed for this resolution, skip
                    continue

            # Build metadata
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
            return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}

        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()


class AudioProbeNode:
    DESCRIPTION = """
Extract audio metadata.
- Inputs:
  - url: audio URL to probe.
- Outputs:
  - metadata_json: JSON string containing audio metadata.
- Behavior:
  - Downloads audio, extracts metadata using ffprobe.
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

    def probe(self, url: str):
        import json
        import subprocess
        import os

        if not url:
            raise ValueError("URL cannot be empty.")

        # Download audio to temp file
        tmp_path = _download_to_temp(url)
        try:
            # Get file size
            size = os.path.getsize(tmp_path)

            # Extract metadata using ffprobe
            ffprobe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(tmp_path)
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            probe_data = json.loads(result.stdout)

            # Extract audio stream info
            audio_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            duration = float(probe_data.get("format", {}).get("duration", 0))
            format_name = probe_data.get("format", {}).get("format_name", "unknown")

            # Build metadata
            metadata = {
                "duration": duration,
                "format": format_name.split(",")[0] if format_name else "unknown",
                "size": size,
                "word_count": 0,  # Placeholder, TTS scenarios may populate this
                "extra": {},
            }

            metadata_json = json.dumps(metadata)
            return {"ui": {"text": [metadata_json]}, "result": (metadata_json,)}

        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()


NODE_CLASS_MAPPINGS.update({
    "VideoProbeNode": VideoProbeNode,
    "ImageProbeNode": ImageProbeNode,
    "AudioProbeNode": AudioProbeNode,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "VideoProbeNode": "Video Probe",
    "ImageProbeNode": "Image Probe",
    "AudioProbeNode": "Audio Probe",
})
