import contextlib
import mimetypes
import re
import shutil
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
