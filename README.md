# PPT 图片转可编辑文字 MVP

这个项目现在包含两部分：

- `make_editable_ppt.py`
  纯命令行转换脚本。
- `run_gui.py`
  桌面版 MVP，支持加载 PPT、查看 OCR 框、手动拖拽/缩放擦除框、修正文案，再导出新的可编辑 PPT。

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

- 打开图片型 `pptx`
- 自动抽取每页图片
- 使用 `PaddleOCR` 识别文字框
- 可视化每页擦除框
- 手动拖动和缩放擦除框
- 按统一边距重算当前页/选中框
- 删除误识别框
- 新增手动框
- 直接修改框里的文本内容
- 可选清除右下角水印区域
- 调用 `IOPaint` 清底图并导出可编辑版 `pptx`

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
