# ComfyUI Workflow Tools

把整个 `ComfyUI-Workflow-Tools` 放入 `ComfyUI/custom_nodes`，重启后即可使用其中的节点。

## 节点列表

### 基础工具（分类 `AigcWorkflowTools`）

- **API Text Output**
  输入：`text`（STRING，forceInput，需连接其他节点）。
  输出：`text`（STRING）。
  输出节点，将字符串写入 UI/API 返回的 history `ui.text`，便于 API 直接取文本。

- **Download URL**
  输入：`url`（HTTP/HTTPS），`save_dir`（默认 `output`，相对路径基于 ComfyUI 工作目录）。
  输出：`file_path`（绝对路径）。
  从 `Content-Disposition` 或 URL 推断文件名，自动清洗非法字符；如重名会添加序号，不覆盖原文件；仅依赖标准库。

- **Extract File Info**
  输入：`file_path`。
  输出：`file_name`（文件名），`file_type`（扩展名，不含点；若无扩展名则为空）。
  从给定路径提取文件名与扩展名，不依赖外部库。

- **FFmpeg Executor**
  输入：`command`（FFmpeg 命令，不含输出文件路径），`output_dir`（默认 `output`），`output_extension`（默认 `.mp4`），可选 `output_filename`（默认 UUID）。
  输出：`file_path`（绝对路径）。
  自动拼接输出路径到命令末尾执行 FFmpeg，10 分钟超时；可与 `UploadFileToTOS` 串联使用。

### 文件加载（分类 `AigcWorkflowTools`）

- **Load Image From Path**
  输入：`file_path`。
  输出：`image`（IMAGE，B,H,W,C，0-1 float）。
  用 Pillow 读取图片，转 RGB，归一化并转为 torch 张量。

- **Load Video From Path**
  输入：`file_path`。
  输出：`video`（包含 `frames` 张量 T,H,W,C 与 `fps`），`fps`（FLOAT）。
  用 imageio 读取视频为帧张量（0-1 float），附带 fps；需 `imageio` 与 `torch`。

- **Load Audio From Path**
  输入：`file_path`。
  输出：`audio`（waveform，channels × samples），`sample_rate`（INT）。
  用 torchaudio 读取音频，返回波形与采样率；需 `torchaudio`。

### 文件上传（分类 `AigcWorkflowTools`）

- **Upload Image To TOS**
  输入：`image`，`ak`，`sk`，`region`，`bucket`，`upload_dir`（默认 `uploads/images`），可选 `endpoint`。
  输出：`url`，`object_key`。
  将 IMAGE 张量转 PNG 上传至火山引擎 TOS，默认 endpoint `https://tos-{region}.volces.com`，生成 UUID 文件名；需 `tos` SDK、`torch`、`Pillow`。

- **Upload Video To TOS**
  输入：`video`（含 frames 与 fps），`ak`，`sk`，`region`，`bucket`，`upload_dir`（默认 `uploads/videos`），可选 `endpoint`。
  输出：`url`，`object_key`。
  将 VIDEO 帧编码为 MP4 上传 TOS；需 `tos` SDK、`torch`、`numpy`、`imageio`（含 ffmpeg）。

- **Upload Audio To TOS**
  输入：`audio`（waveform），`sample_rate`，`ak`，`sk`，`region`，`bucket`，`upload_dir`（默认 `uploads/audios`），可选 `endpoint`。
  输出：`url`，`object_key`。
  将 AUDIO 保存为 WAV 上传 TOS；需 `tos` SDK、`torch`、`torchaudio`。

- **Upload File To TOS**
  输入（必填）：`ak`，`sk`，`region`，`bucket`，`upload_dir`。
  输入（可选，至少提供一个）：`file_path`、`image`、`video`、`audio`（+ `sample_rate`）。
  输入（可选配置）：`endpoint`、`acl`、`storage_class`、`content_type`、`custom_filename`。
  输出：`url`，`object_key`。
  通用上传节点，优先级 file_path > image > video > audio，自动检测 Content-Type 和扩展名。支持 ACL 和存储类型配置。可直接接收 FFmpegExecutor 的输出。

### 媒体元信息探测（分类 `AigcWorkflowTools/Metadata`）

三个 Probe 节点均为**全异步实现**，使用 aiohttp + asyncio subprocess，不阻塞 ComfyUI 事件循环，适合集群部署。内置指数退避重试（最多 3 次）、90 秒超时预算、临时文件自动清理。

- **Video Probe**
  输入：`url`（视频 URL），可选 `presign_info`（JSON 数组，预签名上传信息），`webp_quality`（1-100，默认 80）。
  输出：`metadata_json`（JSON 字符串，含 width、height、duration、size、has_audio、format、resolution、aspect_ratio、covers）。
  通过 ffprobe 直接探测远程 URL 获取元信息（无需完整下载）。如提供 presign_info，则用 ffmpeg 提取首帧生成多分辨率 WebP 封面并上传。封面生成失败不影响元信息返回。

- **Image Probe**
  输入：`url`（图片 URL），可选 `presign_info`（JSON 数组），`webp_quality`（1-100，默认 85）。
  输出：`metadata_json`（JSON 字符串，含 width、height、size、format、resolution、aspect_ratio、thumbnails）。
  双路径优化：无缩略图需求时通过 ffprobe 直接探测（不下载文件）；需要缩略图时异步下载后用 PIL 处理并上传。

- **Audio Probe**
  输入：`url`（音频 URL）。
  输出：`metadata_json`（JSON 字符串，含 duration、format、size、word_count）。
  通过 ffprobe 直接探测远程 URL，轻量级元信息提取，无需下载文件。

## 依赖

```
tos
pillow
numpy
imageio[ffmpeg]
torchaudio
```

`aiohttp` 为 ComfyUI 核心依赖，无需额外安装。`torch` 由 ComfyUI 环境提供。

如缺少 `tos`（TOS Python SDK）、`imageio`、`torchaudio` 等，请在 ComfyUI 环境安装对应库后使用。
