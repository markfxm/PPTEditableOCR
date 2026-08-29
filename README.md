# PPT 图片转可编辑文字

这个项目用于把“每页是一张图片”的 PDF/PPT 转成可编辑文字版 PPT。程序会先提取页面图片，用户选择 OCR 方式后识别文字区域，并允许手动检查和修正识别框；导出时擦除原图文字，可选择清晰化底图，再重建可编辑文本框。

## 项目组成

- `run_gui.py`：桌面版，适合本机处理 PDF/PPT、检查 OCR 框并导出结果。
- `make_editable_ppt.py`：命令行版，适合批处理或快速转换。
- `web_deploy/`：Web 版，包含 FastAPI 后端、静态前端、Redis/RQ worker 和 Docker Compose 部署文件。

## 运行桌面版

首次在新电脑或新 clone 上开发时，先安装本地依赖：

```powershell
.\setup_dev.ps1
```

启动 GUI：

```powershell
python run_gui.py
```

## 运行命令行版

命令行版支持 `.pptx` 和 `.pdf` 输入。如果输入 PDF，会先自动转成图片型 PPT。

```powershell
python make_editable_ppt.py "C:\path\to\source.pptx" "C:\path\to\output.pptx"
```

如果不传输出路径，默认生成 `<源文件名>-editable-clean.pptx`。

## 运行 Web 版

本地开发可直接运行：

```powershell
.\web_deploy\run_local_dev.ps1
```

脚本会启动：

- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`

Docker 部署入口在 `web_deploy/docker-compose.yml`，包含 `api`、`worker`、`redis`、`nginx` 四个服务。Web 版支持上传 PDF/PPTX、查看页面和识别框、编辑框内容、导出后下载结果。同步开发模式下可通过 `WEB_SYNC_JOBS=1` 让后端直接在请求进程中处理任务；Docker 模式下使用 Redis/RQ worker。

## 桌面版功能

- 打开图片型 `.pptx`
- 打开 `.pdf` 并转换成每页一张整页图片的 `.pptx`
- 自动抽取每页图片；由用户选择本地或远端 PaddleOCR 后手动开始识别
- 支持本地 PaddleOCR 和远端 PaddleOCR
- 可保存、删除远端 OCR 令牌
- 远端 OCR 令牌无效时会提示用户设置令牌或切换为本地 OCR
- 在右侧 PPT 列表中切换本次打开或转换生成的 PPT
- 在页面预览中拖动、缩放、删除、新增 OCR 框
- 自动检测图片区候选，可拖动、缩放、新增或删除候选框
- 可按需下载 SAM 2.1 Tiny，对确认的候选框执行本地 AI 精细分割；失败时回退 OpenCV
- 将确认后的图片区导出为透明 PNG，并作为可移动、缩放的独立 PPT 图片重新插入
- 导出时优先使用 OCR 文字轮廓生成精细擦除蒙版，减少相邻插图被矩形擦除框误伤；无法可靠分析的框会标记为矩形回退
- 修改选中框的导出文本
- 对当前页全部框或选中框按边距重算擦除范围
- 可选清除当前页右下角水印区域
- 支持竖向文字框，导出时可重建为旋转文本
- 支持撤销框编辑操作，快捷键 `Ctrl+Z`
- 保存和自动加载识别框缓存，避免重复 OCR
- 导出时调用 IOPaint/LaMa 擦除底图文字
- 可选择在擦除后调用 RealESRGAN 对底图做 2x 清晰化
- 重建可编辑文本框并导出新的 `.pptx`
- 内置“说明书”窗口，可在工具栏打开

## 基本流程

桌面版 PPT 预览区上方有一条流程图，会显示从导入到导出的完整步骤。

1. 如果源文件是 PDF，点击“打开 PDF”，程序会直接把 PDF 页面载入内部工作区，不再生成 `<PDF名>-from-pdf.pptx` 中间 PPT。
2. 如果源文件已经是图片型 PPT，点击“打开 PPT”直接导入。
3. 程序提取每页图片，但不会自动 OCR。
4. 在右侧“OCR 设置”中选择“本地 PaddleOCR”或“远端 PaddleOCR”，然后点击“开始 OCR（按当前选择）”。
5. OCR 完成后，在中间画布检查蓝色识别框，必要时拖动、缩放、删除或新增框。
6. 在右侧修改文本、边距、旋转；不需要处理的文字区域可以直接删除对应框。
7. 点击“保存识别框”可手动保存当前编辑状态。
8. 如需拆出插图，在“图片拆分”中调整紫色候选框并点击“AI 精细分割”；也可选择“恢复 OpenCV 蒙版”直接确认候选。
9. 按需要勾选“导出时清晰化底图（RealESRGAN）”。
10. 点击右侧“继续：导出可编辑 PPT”或工具栏“导出可编辑 PPT”，程序会自动保存识别框、擦除底图文字，并按勾选状态决定是否清晰化底图，最后重建文本框和已确认图片。

## AI 图片分割

- AI 图片分割使用视觉分割模型，不需要 LLM、Ollama 或在线推理 API。
- 第一次点击“AI 精细分割”时，程序会询问是否下载约 156 MB 的官方 `sam2.1_hiera_tiny` 权重。
- 模型保存在 `%LOCALAPPDATA%\PPTEditableOCR\models\sam2`，下载完成前使用临时文件，并通过固定 SHA-256 校验。
- 紫色框是未确认候选，绿色框是已确认候选；只有已确认候选参与底图擦除和独立图片导出。
- 调整已分割候选框会使旧蒙版失效，需要再次 AI 分割或选择“恢复 OpenCV 蒙版”。
- AI 推理优先使用 CUDA，失败时自动重试 CPU；仍失败则将当前候选回退为 OpenCV，不阻塞 PPT 导出。
- 第一版仅在 Windows 桌面端提供，Web 版暂不包含 AI 图片分割。

## OCR 设置

- 默认使用“本地 PaddleOCR”，不需要网络和令牌。
- 如果电脑配置较低，可以在右侧“OCR 设置”中切换为“远端 PaddleOCR”。
- 桌面版导入 PPT/PDF 后不会自动开始 OCR，需要用户确认识别方式后点击“开始 OCR（按当前选择）”。
- 远端 OCR 需要 40 位 PaddleOCR 访问令牌。
- 令牌会保存在本机应用设置中；切回本地 OCR 后不会使用该令牌。
- 可以点击“删除远端 OCR 令牌”清除已保存令牌，并自动切回本地 OCR。
- 如果选择远端 OCR 但令牌缺失或长度不正确，点击“开始 OCR（按当前选择）”时会提示先设置令牌或切换为本地 OCR。
- 如果当前 `paddleocr` 包不包含远端 SDK，远端识别会提示升级；开发依赖已要求 `paddleocr>=3.6.0`。

## 识别框和导出

- 蓝色框表示 OCR 识别到或手动新增的文字区域。
- 所有识别框默认都会参与擦除和重建；不需要处理的区域请删除对应框。
- 预览中绿色框表示可使用精细文字蒙版，红色框表示将回退到矩形擦除；后者应重点检查是否与插图重叠。手动移动、缩放或新增的框会使用矩形回退。
- 手动修正过的文本即使 OCR 置信度较低，也会参与导出重建。
- `NotebookLM` 这类右下角水印文字不会生成识别框，右下角水印区域可用独立开关额外擦除。
- 导出底图会先经过 IOPaint/LaMa 擦除；如果勾选“导出时清晰化底图（RealESRGAN）”，再经过 RealESRGAN 2x 清晰化。
- 清晰化会改变底图像素尺寸，但文本框坐标仍按原始页面图片尺寸映射，避免文字位置缩偏。

## 识别框缓存

- 点击“保存识别框”会保存当前 PPT 的 OCR 框、手动新增框、文本修正、旋转信息和水印开关。
- 点击“导出可编辑 PPT”时，程序会先自动保存一次识别框，再开始导出。
- 缓存优先保存到 PPT 同目录，文件名是 `<PPT名>.ppttoedit.json`。
- 如果 PPT 同目录不可写，会保存到本机用户目录下的 `PPTEditableOCR\ocr_caches`。
- 再次打开同一个 PPT 时，缓存页数和页面尺寸匹配就会自动加载并跳过 OCR。

## Web 版说明

- 前端可上传 `.pdf` 或 `.pptx`。
- PDF 会先在后端转成图片型 PPT，再进入 OCR。
- 左侧显示页列表和每页识别框数量。
- 中间画布显示页面预览和识别框。
- 右侧可编辑边距、文本、旋转角度和水印开关。
- 修改会自动同步到后端，导出前会使用最新识别框。
- 导出结果保存在服务端任务目录中，再通过“下载结果”下载到本机。
- 任务数据默认写入 `web_deploy/data` 或 `DATA_DIR` 指定目录。
- 后端包含旧任务清理逻辑，默认按 24 小时阈值清理。

## 打包 Windows 安装版

先构建可分发应用目录：

```powershell
.\build_windows.ps1 -SkipInstaller
```

如果机器上已经安装了 Inno Setup 6，并且 `ISCC.exe` 在 `PATH` 里，可以直接生成安装包：

```powershell
.\build_windows.ps1
```

输出位置都在 `release_artifacts/` 下：

- 应用目录：`release_artifacts\dist\PPTEditableOCR`
- 安装包：`release_artifacts\installer_output\PPTEditableOCR-Setup.exe`

打包脚本会尽量把本机已缓存的模型一起放进安装包：

- IOPaint LaMa：`big-lama.pt`
- RealESRGAN：`realesr-general-x4v3.pth`
- PaddleOCR 检测/识别模型：`PP-OCRv5_server_det`、`PP-OCRv5_server_rec`

SAM 2.1 权重不会放进安装包，由桌面程序在用户首次确认使用时下载。SAM 2 代码和模型由 Meta 发布，使用前请同时遵守其 Apache 2.0/BSD 许可文件。

如果本机尚未缓存某个模型，安装包不会包含它，运行时可能需要首次下载。模型越完整，离线可用性越好，但安装包体积也会更大。

## 注意事项

- 当前主要面向“每页是一张图片”的 PPT/PDF。
- AI 清晰化会补细节，不等于还原真实原始细节。
- RealESRGAN 使用 CPU 时可能较慢，页数多或图片大时导出会明显变久。
- Web 版 Docker 部署需要 Redis 和 worker 正常运行，否则异步任务不会处理。

## 下一步建议

- 增加“框选新增区域”，替代固定位置新增框
- 增加项目保存/继续编辑
- 增加批量同步同类页设置
- 给 RealESRGAN 增加界面开关和倍率选项
- 精简打包体积和首次模型下载流程
