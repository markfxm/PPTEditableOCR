# PPT 图片转可编辑文字 MVP

这个项目现在包含两部分：

- `make_editable_ppt.py`
  纯命令行转换脚本。
- `run_gui.py`
  桌面版 MVP，支持 PDF 转图片型 PPT、加载 PPT、查看 OCR 框、手动拖拽/缩放擦除框、修正文案，再导出新的可编辑 PPT。

## 运行 GUI

首次在新电脑/新 clone 上开发时，先安装本地依赖：

```powershell
.\setup_dev.ps1
```

```powershell
python run_gui.py
```

## 运行命令行版

```powershell
python make_editable_ppt.py "C:\path\to\source.pptx" "C:\path\to\output.pptx"
```

## 当前 GUI 能做什么

- 将横向 PDF 转成每页一张整页图片的 `pptx`
- 在右侧 PPT 列表中切换本次打开或转换生成的 PPT
- 打开图片型 `pptx`
- 自动抽取每页图片
- 默认使用本地 `PaddleOCR` 识别文字框，也可在界面中切换为远端 PaddleOCR
- 可视化每页擦除框
- 手动拖动和缩放擦除框
- 按统一边距重算当前页/选中框
- 删除误识别框
- 新增手动框
- 直接修改框里的文本内容
- 保存和自动加载识别框缓存，避免重复 OCR
- 可选清除右下角水印区域
- 调用 `IOPaint` 清底图并导出可编辑版 `pptx`

PDF 流程：

1. 点击“PDF 转 PPT”，选择 `.pdf` 文件。
2. 程序会在 PDF 同目录生成 `<PDF名>-from-pdf.pptx`，如果文件已存在会自动追加序号。
3. 生成的 PPT 会自动加入右侧“PPT 列表”并打开。
4. 检查和调整 OCR 框后，点击“导出可编辑 PPT”生成最终可编辑版。

PPT 列表：

- 右侧上方会显示本次运行期间打开过的 PPT，以及 PDF 转换后生成的 PPT。
- 点击列表中的一个 PPT，会自动打开它并刷新左侧页列表和中间预览区。
- 列表只在本次运行期间保留，关闭软件后会清空。
- 当前仍然一次只编辑一个 PPT；列表用于切换当前正在编辑的 PPT。

OCR 设置：

- 默认使用“本地 PaddleOCR”，不需要网络和令牌。
- 如果电脑配置较低，可以在右侧“OCR 设置”中选择“远端 PaddleOCR”。
- 首次使用远端 OCR 前，点击“设置远端 OCR 令牌”，输入自己申请的 40 位访问令牌。
- 远端 OCR 令牌会保存在本机应用设置中；切回本地 OCR 后不会使用该令牌。
- 如果当前安装的 `paddleocr` 包不包含远端 SDK，远端识别会提示升级；开发依赖已要求 `paddleocr>=3.6.0`。

识别框缓存：

- 点击“保存识别框”会保存当前 PPT 的识别框、文本修正、参与擦除开关、旋转信息和水印开关。
- 点击“导出可编辑 PPT”时，程序会先自动保存一次识别框，再开始导出。
- 缓存优先保存到 PPT 同目录，文件名是 `<PPT名>.ppttoedit.json`。
- 如果 PPT 同目录不可写，会保存到本机用户目录下的 `PPTEditableOCR\ocr_caches`。
- 再次打开同一个 PPT 时，缓存页数和页面尺寸匹配就会自动加载并跳过 OCR。

## 打包 Windows 安装版

先构建可分发应用目录：

```powershell
.\build_windows.ps1 -SkipInstaller
```

如果机器上已经安装了 Inno Setup 6，并且 `ISCC.exe` 在 `PATH` 里，可以直接生成安装包：

```powershell
.\build_windows.ps1
```

输出位置都在 `release_artifacts/` 下，方便和源码分开：

- 应用目录：`release_artifacts\dist\PPTEditableOCR`
- 安装包：`release_artifacts\installer_output\PPTEditableOCR-Setup.exe`

说明：

- 当前打包脚本会尽量把本机已缓存的 `IOPaint lama` 模型和 `PaddleOCR` 检测/识别模型一起打进安装包。
- 这样安装后的软件更接近离线可用，但安装包体积会明显变大。

## 下一步建议

- 增加“框选新增区域”而不是固定位置新增框
- 增加项目保存/继续编辑
- 增加批量同步同类页设置
- 精简打包体积和首次模型下载流程
