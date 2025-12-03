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
        # Return plain string in UI payload instead of a single-item list.
        return {"ui": {"text": text}, "result": (text,)}


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
