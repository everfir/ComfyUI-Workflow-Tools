# ComfyUI Workflow Tools

把整个 `ComfyUI-Workflow-Tools` 放入 `ComfyUI/custom_nodes`，重启后即可使用其中的节点。

目前包含：
- **API Text Output**（分类 `AigcWorkflowTools`）  
  输入：`text`（STRING，forceInput，需连接其他节点）。  
  输出：`text`（STRING）。  
  节点 Info 摘要：输出节点，将字符串写入 UI/API 返回的 history `ui.text`，便于 API 直接取文本。
- **Download URL**（分类 `AigcWorkflowTools`）  
  输入：`url`（HTTP/HTTPS），`save_dir`（默认 `output`，相对路径基于 ComfyUI 工作目录）。  
  输出：`file_path`（绝对路径）。  
  从 `Content-Disposition` 或 URL 推断文件名，自动清洗非法字符；如重名会添加序号，不覆盖原文件；仅依赖标准库。

节点 Info 摘要（UI 中可见）：
- 说明：下载 HTTP/HTTPS 链接到本地。
- 行为：尝试从响应头或 URL 获取文件名，重名时添加序号，使用自定义 UA，流式写盘，使用标准库。

- **Extract File Info**（分类 `AigcWorkflowTools`）  
  输入：`file_path`。  
  输出：`file_name`（文件名），`file_type`（扩展名，不含点；若无扩展名则为空）。
  节点 Info 摘要：从给定路径提取文件名与扩展名，不依赖外部库。

- **Load Image From Path**（分类 `AigcWorkflowTools`）  
  输入：`file_path`。  
  输出：`image`（IMAGE，B,H,W,C，0-1 float）。  
  节点 Info 摘要：用 Pillow 读取图片，转 RGB，归一化并转为 torch 张量。

- **Load Video From Path**（分类 `AigcWorkflowTools`）  
  输入：`file_path`。  
  输出：`video`（包含 `frames` 张量 T,H,W,C 与 `fps`），`fps`（FLOAT）。  
  节点 Info 摘要：用 imageio 读取视频为帧张量（0-1 float），附带 fps；需 `imageio` 与 `torch`。

- **Load Audio From Path**（分类 `AigcWorkflowTools`）  
  输入：`file_path`。  
  输出：`audio`（waveform，channels x samples），`sample_rate`（INT）。  
  节点 Info 摘要：用 torchaudio 读取音频，返回波形与采样率；需 `torchaudio`。

- **Upload Image To TOS**（分类 `AigcWorkflowTools`）  
  输入：`image`，`ak`，`sk`，`region`，`bucket`，`upload_dir`（默认 `uploads/images`），可选 `endpoint`。  
  输出：`url`，`object_key`。  
  节点 Info 摘要：将 IMAGE 张量转 PNG 上传至火山引擎 TOS，默认 endpoint `https://tos-{region}.volces.com`，生成 UUID 文件名；需 `tos` SDK、`torch`、`Pillow`。

- **Upload Video To TOS**（分类 `AigcWorkflowTools`）  
  输入：`video`（含 frames 与 fps），`ak`，`sk`，`region`，`bucket`，`upload_dir`（默认 `uploads/videos`），可选 `endpoint`。  
  输出：`url`，`object_key`。  
  节点 Info 摘要：将 VIDEO 帧编码为 MP4 上传 TOS；需 `tos` SDK、`torch`、`numpy`、`imageio`（含 ffmpeg）。

- **Upload Audio To TOS**（分类 `AigcWorkflowTools`）  
  输入：`audio`（waveform），`sample_rate`，`ak`，`sk`，`region`，`bucket`，`upload_dir`（默认 `uploads/audios`），可选 `endpoint`。  
  输出：`url`，`object_key`。  
  节点 Info 摘要：将 AUDIO 保存为 WAV 上传 TOS；需 `tos` SDK、`torch`、`torchaudio`。

依赖提示：如缺少 `tos`（TOS Python SDK）、`imageio`、`torchaudio` 等，请在 ComfyUI 环境安装对应库后使用。
