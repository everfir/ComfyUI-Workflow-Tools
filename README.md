# ComfyUI Workflow Tools

把整个 `ComfyUI-Workflow-Tools` 放入 `ComfyUI/custom_nodes`，重启后即可使用其中的节点。

目前包含：
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
