from __future__ import annotations

import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
for deps_name in [".py310gui", ".py310iopaint", ".py310deps"]:
    deps = BASE / deps_name
    if deps.exists() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .core import (
    OCRBox,
    PPTProject,
    PPTSlide,
    convert_pdf_to_pptx,
    export_editable_ppt,
    prepare_project,
    save_project_cache,
    PAGE_READY_PREFIX,
    PROGRESS_PREFIX,
)

BoxSnapshot = list[tuple[str, float, tuple[int, int, int, int], tuple[int, int, int, int], bool, bool, bool, int]]


def page_list_text(index: int, box_count: int, status: str = "ok") -> str:
    if status == "failed":
        return f"第{index}页 - OCR失败"
    return f"第{index}页 - {box_count} 个框"


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, progress=self.progress.emit, **self.kwargs)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


class EditableRectItem(QGraphicsRectItem):
    HANDLE_SIZE = 8.0

    def __init__(self, box: OCRBox, changed_cb):
        left, top, right, bottom = box.erase_rect
        super().__init__(0, 0, right - left, bottom - top)
        self.box = box
        self.changed_cb = changed_cb
        self.setPos(left, top)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._active_handle = None
        self._press_scene_pos = QPointF()
        self._press_rect = QRectF()
        self._press_pos = QPointF()
        self._press_erase_rect = box.erase_rect
        self._update_pen()

    def _update_pen(self):
        if not self.box.enabled:
            color = QColor(150, 150, 150, 190)
        elif self.isSelected():
            color = QColor(255, 140, 0, 220)
        else:
            color = QColor(0, 170, 255, 200)
        self.setPen(QPen(color, 2))

    def _handles(self):
        r = self.rect()
        s = self.HANDLE_SIZE
        return {
            "tl": QRectF(r.left() - s / 2, r.top() - s / 2, s, s),
            "tr": QRectF(r.right() - s / 2, r.top() - s / 2, s, s),
            "bl": QRectF(r.left() - s / 2, r.bottom() - s / 2, s, s),
            "br": QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
        }

    def _handle_at(self, pos):
        for name, rect in self._handles().items():
            if rect.contains(pos):
                return name
        return None

    def hoverMoveEvent(self, event):
        handle = self._handle_at(event.pos())
        if handle in {"tl", "br"}:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in {"tr", "bl"}:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor if self.isSelected() else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        self._active_handle = self._handle_at(event.pos())
        self._press_scene_pos = event.scenePos()
        self._press_rect = QRectF(self.rect())
        self._press_pos = QPointF(self.pos())
        self._press_erase_rect = self.box.erase_rect
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._active_handle:
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self._press_scene_pos
        min_size = 12.0
        scene_rect = self.scene().sceneRect() if self.scene() else QRectF()
        left = self._press_pos.x()
        top = self._press_pos.y()
        right = self._press_pos.x() + self._press_rect.width()
        bottom = self._press_pos.y() + self._press_rect.height()

        if "l" in self._active_handle:
            left = min(left + delta.x(), right - min_size)
            if not scene_rect.isNull():
                left = max(scene_rect.left(), left)
        if "r" in self._active_handle:
            right = max(right + delta.x(), left + min_size)
            if not scene_rect.isNull():
                right = min(scene_rect.right(), right)
        if "t" in self._active_handle:
            top = min(top + delta.y(), bottom - min_size)
            if not scene_rect.isNull():
                top = max(scene_rect.top(), top)
        if "b" in self._active_handle:
            bottom = max(bottom + delta.y(), top + min_size)
            if not scene_rect.isNull():
                bottom = min(scene_rect.bottom(), bottom)

        self.setPos(QPointF(left, top))
        self.setRect(0, 0, max(min_size, right - left), max(min_size, bottom - top))
        event.accept()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._active_handle = None
        self.sync_to_box(record_undo=True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_pen()
        return super().itemChange(change, value)

    def sync_to_box(self, record_undo: bool = False):
        rect = self.sceneBoundingRect()
        new_rect = (round(rect.left()), round(rect.top()), round(rect.right()), round(rect.bottom()))
        if record_undo and new_rect != self._press_erase_rect:
            self.changed_cb(self, before_change=True)
        self.box.set_erase_rect(new_rect)
        if self.box.manual:
            self.box.set_bbox_from_rect(new_rect)
            left, top, right, bottom = self.box.erase_rect
            self.box.rotation = 270 if (bottom - top) > (right - left) * 1.45 else 0
        self._update_pen()
        self.changed_cb(self, before_change=False)


class SlideScene(QGraphicsScene):
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selectionChanged.connect(self.selection_changed.emit)


class PPTGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and self.scene().items():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ManualDialog(QDialog):
    SECTIONS = [
        (
            "基本流程",
            """1. 如果源文件是 PDF，先点击“PDF 转 PPT”，生成每页一张整页图片的 PPT。
2. 点击“打开 PPT”，导入图片型 PPT。
3. 程序会提取每页图片，并用 OCR 自动识别文字区域。
4. 在中间画布检查蓝色文字框，必要时调整、删除或新增框。
5. 点击“导出可编辑 PPT”，程序会先擦除原图文字，再重建可编辑文本框。""",
        ),
        (
            "PDF 转 PPT",
            """PDF 转 PPT：
点击工具栏“PDF 转 PPT”，选择 .pdf 文件。程序会在 PDF 同目录生成 <PDF名>-from-pdf.pptx。

输出规则：
如果同名 PPT 已存在，程序会自动追加 -2、-3 等序号，避免覆盖已有文件。

继续编辑：
转换完成后，生成的 PPT 会自动加入右侧“PPT 列表”并打开。检查 OCR 框后，点击“导出可编辑 PPT”即可继续。""",
        ),
        (
            "PPT 列表",
            """PPT 列表：
右侧上方会显示本次运行期间打开过的 PPT，以及 PDF 转换后生成的 PPT。

切换 PPT：
点击列表中的一个 PPT，程序会自动打开它并刷新左侧页列表和中间预览区。当前仍然一次只编辑一个 PPT。

自动加入：
点击“打开 PPT”选择的文件会加入列表。PDF 转 PPT 完成后，生成的 PPT 也会自动加入列表并打开。

列表范围：
这个列表只在本次运行期间保留，关闭软件后会清空。""",
        ),
        (
            "页面与框",
            """页面边界：
灰色边框表示当前 PPT 页面范围，导出的页面大小以这个范围为准。

蓝色框：
表示 OCR 识别到或手动新增的文字区域。导出时，参与处理的框会用于擦除原图文字并重建可编辑文本。

拖拽/缩放：
选中框后可以拖动位置，也可以拖动四角调整大小。竖向文字可以用窄高框框住，输入正常横向文本后导出为旋转文本。""",
        ),
        (
            "右侧功能",
            """横向边距 / 纵向边距：
控制识别框向外扩大的像素数。边距越大，擦除范围越宽，能减少文字残留，但过大可能擦到附近线条或图片。

当前页所有框按边距重算：
用当前横向/纵向边距，重新计算当前页全部框的擦除范围。适合整页框普遍太紧或太松时使用。

选中框按边距重算：
只重新计算当前选中框的擦除范围，不影响其他框。

删除选中框：
删除当前选中的框。删除后该区域不会被擦除，也不会重建文本。

新增框：
手动添加一个框。新增框会先显示在当前页右上角区域，方便找到；之后可以拖动、缩放到目标文字位置，并在下方文本框输入正确内容。适合 OCR 漏识别的文字，尤其是竖向坐标轴文字或低置信度文字。

选中框参与擦除和重建：
勾选后，导出时会擦除这个区域的原图文字，并重建为可编辑文本。取消勾选后，该框不会参与导出处理。

清除当前页右下角水印区域：
导出时额外擦除右下角预设水印区域，比如 NotebookLM 标记。这个选项不依赖选中的文字框。

文本输入框：
显示并修改选中框导出后的文本内容。手动修正过的文本即使 OCR 置信度较低，也会参与导出重建。""",
        ),
        (
            "保存识别框",
            """保存识别框：
点击工具栏“保存识别框”，会把当前 PPT 的 OCR 框、手动新增框、文本修正、是否参与擦除、旋转信息、水印开关等保存到缓存文件。

导出时保存：
点击“导出可编辑 PPT”时，程序会先自动保存一次识别框，再开始导出。这样可以先放心调整框，确认后再把最终状态写入缓存。

读取规则：
再次打开同一个 PPT 时，如果缓存页数和页面图片尺寸匹配，程序会直接加载识别框并跳过 OCR。

保存位置：
程序优先保存到 PPT 同目录，文件名是 <PPT名>.ppttoedit.json。这样把 PPT 和 JSON 一起移动到其他电脑或文件夹时也能继续使用。

备用位置：
如果 PPT 同目录无法写入，程序会保存到本机用户目录下的 PPTEditableOCR/ocr_caches。这个备用缓存只适合同一台电脑、同一个 PPT 路径继续使用。""",
        ),
        (
            "撤销",
            """点击“撤销”或按 Ctrl+Z，可以撤销上一步框编辑操作。

支持撤销：
拖拽、缩放、新增、删除、重算边距、修改文本、切换“参与擦除和重建”。

不撤销：
水印区域开关。""",
        ),
        (
            "常见问题",
            """竖向文字识别不了：
这通常是 OCR 对局部旋转 90 度文字识别不稳定。可以手动新增窄高框，输入正确文本，再导出。

文字导出后不见：
请确认该框已勾选“参与擦除和重建”，并且文本输入框里有正确内容。

擦除后有残留：
适当增大横向/纵向边距，然后对当前页或选中框按边距重算。

擦到了不该擦的内容：
减小边距，或手动缩小对应框。""",
        ),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("说明书")
        self.resize(820, 560)

        layout = QHBoxLayout(self)
        self.section_list = QListWidget()
        self.section_list.setFixedWidth(180)
        self.content = QPlainTextEdit()
        self.content.setReadOnly(True)

        for title, _text in self.SECTIONS:
            self.section_list.addItem(title)
        self.section_list.currentRowChanged.connect(self.show_section)

        layout.addWidget(self.section_list)
        layout.addWidget(self.content, 1)
        self.section_list.setCurrentRow(0)

    def show_section(self, row: int):
        if row < 0 or row >= len(self.SECTIONS):
            return
        title, text = self.SECTIONS[row]
        self.content.setPlainText(f"{title}\n\n{text}")


class MainWindow(QMainWindow):
    MAX_UNDO_STEPS = 50

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPT 图片转可编辑文字 - MVP")
        self.resize(1500, 900)
        self.project: PPTProject | None = None
        self.current_slide: PPTSlide | None = None
        self.current_items: list[EditableRectItem] = []
        self.worker_thread: QThread | None = None
        self.worker: Worker | None = None
        self.selected_item: EditableRectItem | None = None
        self.undo_stacks: dict[int, list[BoxSnapshot]] = {}
        self.pending_select_index: int | None = None
        self.ppt_paths: list[Path] = []
        self.selecting_ppt_list_item = False
        self.pending_autoload_ppt: Path | None = None
        self.pending_pdf_loaded_notice: Path | None = None

        self._build_ui()

    def _build_ui(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        open_action = QAction("打开 PPT", self)
        open_action.triggered.connect(self.open_ppt)
        pdf_to_ppt_action = QAction("PDF 转 PPT", self)
        pdf_to_ppt_action.triggered.connect(self.convert_pdf)
        export_action = QAction("导出可编辑 PPT", self)
        export_action.triggered.connect(self.export_ppt)
        self.save_cache_action = QAction("保存识别框", self)
        self.save_cache_action.triggered.connect(self.save_current_cache)
        self.save_cache_action.setEnabled(False)
        self.undo_action = QAction("撤销", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        manual_action = QAction("说明书", self)
        manual_action.triggered.connect(self.show_manual)
        toolbar.addAction(open_action)
        toolbar.addAction(pdf_to_ppt_action)
        toolbar.addAction(export_action)
        toolbar.addAction(self.save_cache_action)
        toolbar.addAction(self.undo_action)
        toolbar.addAction(manual_action)

        root = QSplitter()
        self.setCentralWidget(root)

        self.slide_list = QListWidget()
        self.slide_list.currentRowChanged.connect(self.on_slide_changed)
        root.addWidget(self.slide_list)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.scene = SlideScene()
        self.scene.selection_changed.connect(self.on_scene_selection_changed)
        self.view = PPTGraphicsView()
        self.view.setScene(self.scene)
        center_layout.addWidget(self.view)
        root.addWidget(center)

        side = QWidget()
        side_layout = QVBoxLayout(side)

        ppt_group = QGroupBox("PPT 列表")
        ppt_group_layout = QVBoxLayout(ppt_group)
        self.ppt_file_list = QListWidget()
        self.ppt_file_list.setMaximumHeight(130)
        self.ppt_file_list.currentRowChanged.connect(self.on_ppt_file_selected)
        ppt_group_layout.addWidget(self.ppt_file_list)
        side_layout.addWidget(ppt_group)

        form = QFormLayout()
        self.pad_x = QSpinBox()
        self.pad_x.setRange(0, 200)
        self.pad_x.setValue(16)
        self.pad_y = QSpinBox()
        self.pad_y.setRange(0, 200)
        self.pad_y.setValue(12)
        form.addRow("横向边距", self.pad_x)
        form.addRow("纵向边距", self.pad_y)
        side_layout.addLayout(form)

        self.reset_slide_btn = QPushButton("当前页所有框按边距重算")
        self.reset_slide_btn.clicked.connect(self.reset_current_slide_boxes)
        side_layout.addWidget(self.reset_slide_btn)

        self.reset_box_btn = QPushButton("选中框按边距重算")
        self.reset_box_btn.clicked.connect(self.reset_selected_box)
        side_layout.addWidget(self.reset_box_btn)

        self.delete_box_btn = QPushButton("删除选中框")
        self.delete_box_btn.clicked.connect(self.delete_selected_box)
        side_layout.addWidget(self.delete_box_btn)

        self.add_box_btn = QPushButton("新增框")
        self.add_box_btn.clicked.connect(self.add_box)
        side_layout.addWidget(self.add_box_btn)

        self.toggle_box_cb = QCheckBox("选中框参与擦除和重建")
        self.toggle_box_cb.stateChanged.connect(self.on_selected_box_toggle)
        side_layout.addWidget(self.toggle_box_cb)

        self.watermark_cb = QCheckBox("清除当前页右下角水印区域")
        self.watermark_cb.setChecked(True)
        self.watermark_cb.stateChanged.connect(self.on_watermark_toggle)
        side_layout.addWidget(self.watermark_cb)

        self.selected_info = QLabel("未选择任何框")
        self.selected_info.setWordWrap(True)
        side_layout.addWidget(self.selected_info)

        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("选中框的文本内容")
        self.text_edit.editingFinished.connect(self.on_text_edited)
        side_layout.addWidget(self.text_edit)

        self.progress_label = QLabel("空闲")
        self.progress_label.setWordWrap(True)
        side_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        side_layout.addWidget(self.progress_bar)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        side_layout.addWidget(self.log, 1)
        root.addWidget(side)
        root.setStretchFactor(1, 1)

    def append_log(self, message: str):
        if message.startswith(PROGRESS_PREFIX):
            _prefix, percent, text = message.split("|", 2)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(percent))
            self.progress_label.setText(text)
            return
        if message.startswith(PAGE_READY_PREFIX):
            parts = message.split("|", 3)
            _prefix, index, box_count = parts[:3]
            status = parts[3] if len(parts) > 3 else "ok"
            self.upsert_slide_list_item(int(index), int(box_count), status)
            return
        self.log.appendPlainText(message)

    def upsert_slide_list_item(self, slide_index: int, box_count: int, status: str = "ok"):
        row = slide_index - 1
        text = page_list_text(slide_index, box_count, status)
        while self.slide_list.count() <= row:
            next_index = self.slide_list.count() + 1
            self.slide_list.addItem(QListWidgetItem(page_list_text(next_index, 0)))
        item = self.slide_list.item(row)
        if item:
            item.setText(text)

    def set_busy(self, busy: bool):
        self.ppt_file_list.setEnabled(not busy)
        self.slide_list.setEnabled(not busy)
        self.reset_slide_btn.setEnabled(not busy)
        self.reset_box_btn.setEnabled(not busy)
        self.delete_box_btn.setEnabled(not busy)
        self.add_box_btn.setEnabled(not busy)
        self.toggle_box_cb.setEnabled(not busy)
        self.watermark_cb.setEnabled(not busy)
        self.text_edit.setEnabled(not busy)
        self.save_cache_action.setEnabled((not busy) and self.project is not None)
        if busy:
            self.progress_label.setText("运行中...")
            self.progress_bar.setRange(0, 0)
        self.update_undo_action()

    def run_worker(self, fn, finished_cb, *args, **kwargs):
        if self.worker_thread:
            return
        self.worker_thread = QThread(self)
        self.worker = Worker(fn, *args, **kwargs)
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(finished_cb)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.cleanup_worker)
        self.worker.failed.connect(lambda _msg: self.cleanup_worker())
        self.worker_thread.started.connect(self.worker.run)
        self.set_busy(True)
        self.worker_thread.start()

    def cleanup_worker(self):
        self.set_busy(False)
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
        if self.pending_autoload_ppt:
            source = self.pending_autoload_ppt
            self.pending_autoload_ppt = None
            self.load_ppt_path(source, add_to_list=True, select_in_list=True)
            return
        self.update_undo_action()

    def on_worker_failed(self, error_text: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("运行失败")
        self.append_log(error_text)
        QMessageBox.critical(self, "运行失败", error_text)

    def show_manual(self):
        dialog = ManualDialog(self)
        dialog.exec()

    def convert_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PDF 文件", "", "PDF (*.pdf)")
        if not path:
            return
        source = Path(path)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备转换 PDF")
        self.append_log(f"开始 PDF 转 PPT：{source}")
        self.run_worker(convert_pdf_to_pptx, self.on_pdf_converted, source)

    def on_pdf_converted(self, output_pptx: Path):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("PDF 转 PPT 完成")
        self.append_log(f"PDF 转 PPT 完成：{output_pptx}")
        self.add_ppt_to_recent_list(output_pptx, select=True)
        self.pending_autoload_ppt = output_pptx
        self.pending_pdf_loaded_notice = output_pptx

    def open_ppt(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PPT 文件", "", "PowerPoint (*.pptx)")
        if not path:
            return
        self.load_ppt_path(Path(path), add_to_list=True, select_in_list=True)

    def add_ppt_to_recent_list(self, path: Path, select: bool = False) -> int:
        source = path.expanduser().resolve()
        try:
            row = self.ppt_paths.index(source)
        except ValueError:
            row = len(self.ppt_paths)
            self.ppt_paths.append(source)
            item = QListWidgetItem(source.name)
            item.setToolTip(str(source))
            self.ppt_file_list.addItem(item)
        if select:
            self.selecting_ppt_list_item = True
            self.ppt_file_list.setCurrentRow(row)
            self.selecting_ppt_list_item = False
        return row

    def load_ppt_path(self, path: Path, add_to_list: bool = True, select_in_list: bool = True):
        source = path.expanduser().resolve()
        if self.worker_thread:
            return
        if add_to_list:
            self.add_ppt_to_recent_list(source, select=select_in_list)
        self.project = None
        self.current_slide = None
        self.selected_item = None
        self.current_items.clear()
        self.undo_stacks.clear()
        self.pending_select_index = None
        self.slide_list.clear()
        self.scene.clear()
        self.save_cache_action.setEnabled(False)
        self.append_log(f"开始加载：{source}")
        self.run_worker(prepare_project, self.on_project_loaded, source)

    def on_ppt_file_selected(self, row: int):
        if self.selecting_ppt_list_item or self.worker_thread:
            return
        if row < 0 or row >= len(self.ppt_paths):
            return
        source = self.ppt_paths[row]
        if self.project and self.project.source_pptx == source:
            return
        self.load_ppt_path(source, add_to_list=False, select_in_list=False)

    def on_project_loaded(self, project: PPTProject):
        self.project = project
        self.undo_stacks.clear()
        self.slide_list.clear()
        for slide in project.slides:
            item = QListWidgetItem(page_list_text(slide.index, len(slide.boxes)))
            self.slide_list.addItem(item)
        if project.slides:
            self.slide_list.setCurrentRow(0)
        self.append_log("项目加载完成")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("项目加载完成")
        self.save_cache_action.setEnabled(True)
        self.update_undo_action()
        if self.pending_pdf_loaded_notice and self.pending_pdf_loaded_notice.resolve() == project.source_pptx:
            output_pptx = self.pending_pdf_loaded_notice
            self.pending_pdf_loaded_notice = None
            QMessageBox.information(
                self,
                "完成",
                f"PDF 已转换为 PPT，并已加入右侧 PPT 列表：\n{output_pptx}\n\n现在可以检查 OCR 框并导出可编辑 PPT。",
            )

    def save_current_cache(self, show_message: bool = True):
        if not self.project:
            if show_message:
                QMessageBox.information(self, "提示", "请先打开一个 PPT。")
            return
        try:
            cache_path = save_project_cache(self.project)
        except Exception as exc:
            self.append_log(f"识别框保存失败：{exc}")
            if show_message:
                QMessageBox.critical(self, "保存失败", f"识别框保存失败：{exc}")
            return
        self.append_log(f"识别框已保存：{cache_path}")
        if show_message:
            QMessageBox.information(self, "完成", f"识别框已保存：\n{cache_path}")

    def on_slide_changed(self, row: int):
        if not self.project or row < 0 or row >= len(self.project.slides):
            return
        self.current_slide = self.project.slides[row]
        self.render_current_slide()
        self.update_undo_action()

    def render_current_slide(self):
        slide = self.current_slide
        if not slide:
            return
        self.scene.clear()
        self.current_items.clear()
        pixmap = QPixmap(str(slide.image_path))
        pixmap_item = QGraphicsPixmapItem(pixmap)
        pixmap_item.setZValue(-10)
        self.scene.addItem(pixmap_item)
        self.scene.setSceneRect(pixmap.rect())
        page_border = QGraphicsRectItem(QRectF(pixmap.rect()))
        border_pen = QPen(QColor(120, 120, 120, 230), 2)
        border_pen.setCosmetic(True)
        page_border.setPen(border_pen)
        page_border.setZValue(5)
        self.scene.addItem(page_border)
        for box in slide.boxes:
            item = EditableRectItem(box, self.on_item_changed)
            item.setZValue(10)
            self.scene.addItem(item)
            self.current_items.append(item)
        if self.pending_select_index is not None:
            if 0 <= self.pending_select_index < len(self.current_items):
                self.current_items[self.pending_select_index].setSelected(True)
            self.pending_select_index = None
        self.watermark_cb.blockSignals(True)
        self.watermark_cb.setChecked(slide.remove_watermark)
        self.watermark_cb.blockSignals(False)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.on_scene_selection_changed()
        self.update_undo_action()

    def slide_key(self, slide: PPTSlide | None = None) -> int | None:
        slide = slide or self.current_slide
        return id(slide) if slide else None

    def snapshot_boxes(self, slide: PPTSlide) -> BoxSnapshot:
        return [
            (box.text, box.score, box.bbox, box.erase_rect, box.enabled, box.manual, box.edited, box.rotation)
            for box in slide.boxes
        ]

    def restore_boxes(self, slide: PPTSlide, snapshot: BoxSnapshot):
        slide.boxes = [
            OCRBox(
                text=text,
                score=score,
                bbox=bbox,
                erase_rect=erase_rect,
                enabled=enabled,
                manual=manual,
                edited=edited,
                rotation=rotation,
            )
            for text, score, bbox, erase_rect, enabled, manual, edited, rotation in snapshot
        ]

    def push_undo_state(self):
        if not self.current_slide:
            return
        key = self.slide_key(self.current_slide)
        if key is None:
            return
        stack = self.undo_stacks.setdefault(key, [])
        stack.append(self.snapshot_boxes(self.current_slide))
        if len(stack) > self.MAX_UNDO_STEPS:
            del stack[0]
        self.update_undo_action()

    def update_undo_action(self):
        if not hasattr(self, "undo_action"):
            return
        key = self.slide_key()
        has_undo = bool(key is not None and self.undo_stacks.get(key))
        self.undo_action.setEnabled(has_undo and self.worker_thread is None)

    def update_slide_list_item(self, slide: PPTSlide):
        if not self.project:
            return
        try:
            row = self.project.slides.index(slide)
        except ValueError:
            return
        item = self.slide_list.item(row)
        if item:
            item.setText(page_list_text(slide.index, len(slide.boxes)))

    def undo(self):
        if not self.current_slide:
            return
        key = self.slide_key(self.current_slide)
        stack = self.undo_stacks.get(key) if key is not None else None
        if not stack:
            self.update_undo_action()
            return
        selected_index = None
        if self.selected_item and self.selected_item.box in self.current_slide.boxes:
            selected_index = self.current_slide.boxes.index(self.selected_item.box)
        snapshot = stack.pop()
        self.restore_boxes(self.current_slide, snapshot)
        self.pending_select_index = selected_index
        self.update_slide_list_item(self.current_slide)
        self.render_current_slide()

    def on_item_changed(self, _item: EditableRectItem, before_change: bool = False):
        if before_change:
            self.push_undo_state()
        self.on_scene_selection_changed()

    def on_scene_selection_changed(self):
        items = [item for item in self.scene.selectedItems() if isinstance(item, EditableRectItem)]
        self.selected_item = items[0] if items else None
        if not self.selected_item:
            self.selected_info.setText("未选择任何框")
            self.toggle_box_cb.blockSignals(True)
            self.toggle_box_cb.setChecked(False)
            self.toggle_box_cb.blockSignals(False)
            self.text_edit.blockSignals(True)
            self.text_edit.clear()
            self.text_edit.blockSignals(False)
            return
        box = self.selected_item.box
        self.selected_info.setText(
            f"文本：{box.text}\n"
            f"置信度：{box.score:.3f}\n"
            f"原始 bbox：{box.bbox}\n"
            f"擦除框：{box.erase_rect}"
        )
        self.toggle_box_cb.blockSignals(True)
        self.toggle_box_cb.setChecked(box.enabled)
        self.toggle_box_cb.blockSignals(False)
        self.text_edit.blockSignals(True)
        self.text_edit.setText(box.text)
        self.text_edit.blockSignals(False)

    def reset_current_slide_boxes(self):
        if not self.current_slide:
            return
        self.push_undo_state()
        self.current_slide.reset_boxes(self.pad_x.value(), self.pad_y.value())
        self.render_current_slide()

    def reset_selected_box(self):
        if not self.current_slide or not self.selected_item:
            return
        self.push_undo_state()
        self.selected_item.box.reset_from_bbox(
            self.pad_x.value(),
            self.pad_y.value(),
            self.current_slide.image_width,
            self.current_slide.image_height,
        )
        self.render_current_slide()

    def delete_selected_box(self):
        if not self.current_slide or not self.selected_item:
            return
        self.push_undo_state()
        self.current_slide.boxes.remove(self.selected_item.box)
        self.update_slide_list_item(self.current_slide)
        self.render_current_slide()

    def add_box(self):
        if not self.current_slide:
            return
        self.push_undo_state()
        width = min(280, self.current_slide.image_width - 40)
        height = 60
        left = max(20, self.current_slide.image_width - width - 40)
        top = min(max(20, 40), max(20, self.current_slide.image_height - height - 20))
        box = OCRBox(
            text="新文本",
            score=1.0,
            bbox=(left, top, width, height),
            erase_rect=(left, top, left + width, top + height),
            enabled=True,
            manual=True,
            edited=True,
        )
        self.current_slide.boxes.append(box)
        self.pending_select_index = len(self.current_slide.boxes) - 1
        self.update_slide_list_item(self.current_slide)
        self.render_current_slide()

    def on_text_edited(self):
        if not self.selected_item:
            return
        new_text = self.text_edit.text().strip() or "新文本"
        if new_text == self.selected_item.box.text:
            return
        self.push_undo_state()
        self.selected_item.box.text = new_text
        self.selected_item.box.edited = True
        self.on_scene_selection_changed()

    def on_selected_box_toggle(self, state: int):
        if not self.selected_item:
            return
        enabled = state == Qt.CheckState.Checked.value
        if enabled == self.selected_item.box.enabled:
            return
        self.push_undo_state()
        self.selected_item.box.enabled = enabled
        self.selected_item._update_pen()
        self.on_scene_selection_changed()

    def on_watermark_toggle(self, state: int):
        if self.current_slide:
            self.current_slide.remove_watermark = state == Qt.CheckState.Checked.value

    def export_ppt(self):
        if not self.project:
            QMessageBox.information(self, "提示", "请先打开一个 PPT。")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出可编辑 PPT",
            str(self.project.source_pptx.with_name(f"{self.project.source_pptx.stem}-editable-clean.pptx")),
            "PowerPoint (*.pptx)",
        )
        if not out_path:
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备导出")
        self.save_current_cache(show_message=False)
        self.append_log("开始导出可编辑 PPT")
        self.run_worker(export_editable_ppt, self.on_export_finished, self.project, Path(out_path))

    def on_export_finished(self, _result):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("导出完成")
        self.append_log("导出完成")
        QMessageBox.information(self, "完成", "可编辑 PPT 已导出完成。")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
