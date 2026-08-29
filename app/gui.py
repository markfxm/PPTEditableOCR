from __future__ import annotations

import sys
import traceback
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
for deps_name in [".py310gui", ".py310iopaint", ".py310deps"]:
    deps = BASE / deps_name
    if deps.exists() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .core import (
    OCRBox,
    OCRPageProcessError,
    PPTProject,
    PPTSlide,
    cache_path_candidates,
    prepare_project,
    prepare_pdf_project,
    run_export_editable_ppt_subprocess,
    run_ocr_page_subprocess,
    save_project_cache,
    OCR_BACKEND_LOCAL,
    OCR_BACKEND_REMOTE,
    PAGE_READY_PREFIX,
    PROGRESS_PREFIX,
    REMOTE_OCR_TOKEN_LENGTH,
    VisualAsset,
    detect_visual_assets,
    store_visual_asset_mask,
    visual_asset_from_data,
    visual_asset_to_data,
)
from .ocr_config import resolve_ocr_config
from .sam_segmentation import (
    MODEL_FILENAME,
    MODEL_ID,
    SamSegmentationEngine,
    download_model,
    model_path,
    preferred_device,
    verify_model,
)

FLOW_TEXT = (
    "1. 打开 PPT/PDF --> 2. 选择 OCR 方式并识别 --> "
    "3. 检查识别框 --> 4. 图片拆分 --> 5. IOPaint 擦除 --> "
    "6. 选择是否清晰化 --> 7. 导出可编辑 PPT"
)

BoxSnapshot = list[
    tuple[
        str,
        float,
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        bool,
        bool,
        bool,
        int,
        int | None,
        tuple[tuple[tuple[int, int], ...], ...],
        str,
        str | None,
    ]
]


def page_list_text(index: int, box_count: int, status: str = "ok") -> str:
    if status == "failed":
        return f"第{index}页 - OCR失败"
    if status == "skipped":
        return f"第{index}页 - 已跳过"
    if status == "pending":
        return f"第{index}页 - 待OCR"
    return f"第{index}页 - {box_count} 个框"


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str, object)
    progress = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, progress=self.progress.emit, **self.kwargs)
        except Exception as exc:
            self.failed.emit(traceback.format_exc(), exc)
        else:
            self.finished.emit(result)


class PptListItemWidget(QWidget):
    def __init__(self, path: Path, remove_cb, parent=None):
        super().__init__(parent)
        self.path = path
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 2, 0)
        layout.setSpacing(4)
        remove_btn = QPushButton("×")
        remove_btn.setToolTip("从列表中删除")
        remove_btn.setFixedSize(22, 22)
        remove_btn.clicked.connect(remove_cb)
        layout.addWidget(remove_btn)
        label = QLabel(path.name)
        label.setToolTip(str(path))
        layout.addWidget(label, 1)


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
        elif self.box.mask_mode == "rectangle_fallback" or not self.box.text_regions:
            color = QColor(220, 70, 70, 220)
        elif self.box.text_regions:
            color = QColor(40, 180, 100, 220)
        else:
            color = QColor(0, 170, 255, 200)
        self.setPen(QPen(color, 2))

    def _handles(self):
        r = self.rect()
        s = self.HANDLE_SIZE
        return {
            "tl": QRectF(r.left() - s / 2, r.top() - s / 2, s, s),
            "t": QRectF(r.center().x() - s / 2, r.top() - s / 2, s, s),
            "tr": QRectF(r.right() - s / 2, r.top() - s / 2, s, s),
            "r": QRectF(r.right() - s / 2, r.center().y() - s / 2, s, s),
            "bl": QRectF(r.left() - s / 2, r.bottom() - s / 2, s, s),
            "b": QRectF(r.center().x() - s / 2, r.bottom() - s / 2, s, s),
            "br": QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
            "l": QRectF(r.left() - s / 2, r.center().y() - s / 2, s, s),
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
        elif handle in {"l", "r"}:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif handle in {"t", "b"}:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
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
        self.box.text_regions = ()
        self.box.mask_mode = "pending"
        self.box.mask_reason = "擦除框已手动调整"
        if self.box.manual:
            self.box.set_bbox_from_rect(new_rect)
            left, top, right, bottom = self.box.erase_rect
            self.box.rotation = 270 if (bottom - top) > (right - left) * 1.45 else 0
        self._update_pen()
        self.changed_cb(self, before_change=False)


class EditableAssetItem(QGraphicsRectItem):
    HANDLE_SIZE = 10.0

    def __init__(self, asset: VisualAsset, changed_cb):
        x, y, width, height = asset.bbox
        super().__init__(0, 0, width, height)
        self.asset = asset
        self.changed_cb = changed_cb
        self.setPos(x, y)
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
        self._press_bbox = asset.bbox
        self._update_pen()

    def _update_pen(self):
        if self.isSelected():
            color = QColor(255, 140, 0, 230)
        elif self.asset.confirmed:
            color = QColor(91, 33, 182, 230)
        else:
            color = QColor(124, 58, 237, 220)
        self.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        self.setBrush(QColor(color.red(), color.green(), color.blue(), 18))

    def paint(self, painter: QPainter, option, widget=None):
        super().paint(painter, option, widget)
        rect = self.rect()
        marker_size = min(18.0, max(10.0, min(rect.width(), rect.height()) - 2.0))
        marker_rect = QRectF(
            rect.right() - marker_size - 1.0,
            rect.top() + 1.0,
            marker_size,
            marker_size,
        )
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(91, 33, 182, 235))
        painter.drawRoundedRect(marker_rect, 3.0, 3.0)
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(max(8, int(marker_size * 0.7)))
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 245))
        painter.drawText(marker_rect, Qt.AlignmentFlag.AlignCenter, "P")
        painter.restore()

    def _handles(self):
        rect, size = self.rect(), self.HANDLE_SIZE
        return {
            "tl": QRectF(rect.left() - size / 2, rect.top() - size / 2, size, size),
            "t": QRectF(rect.center().x() - size / 2, rect.top() - size / 2, size, size),
            "tr": QRectF(rect.right() - size / 2, rect.top() - size / 2, size, size),
            "r": QRectF(rect.right() - size / 2, rect.center().y() - size / 2, size, size),
            "bl": QRectF(rect.left() - size / 2, rect.bottom() - size / 2, size, size),
            "b": QRectF(rect.center().x() - size / 2, rect.bottom() - size / 2, size, size),
            "br": QRectF(rect.right() - size / 2, rect.bottom() - size / 2, size, size),
            "l": QRectF(rect.left() - size / 2, rect.center().y() - size / 2, size, size),
        }

    def _handle_at(self, pos):
        return next((name for name, rect in self._handles().items() if rect.contains(pos)), None)

    def shape(self):
        path = super().shape()
        for handle_rect in self._handles().values():
            path.addRect(handle_rect)
        return path

    def hoverMoveEvent(self, event):
        handle = self._handle_at(event.pos())
        if handle in {"tl", "br"}:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in {"tr", "bl"}:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in {"l", "r"}:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif handle in {"t", "b"}:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor if self.isSelected() else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        self._active_handle = self._handle_at(event.pos())
        self._press_scene_pos = event.scenePos()
        self._press_rect = QRectF(self.rect())
        self._press_pos = QPointF(self.pos())
        self._press_bbox = self.asset.bbox
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._active_handle:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._press_scene_pos
        left, top = self._press_pos.x(), self._press_pos.y()
        right = left + self._press_rect.width()
        bottom = top + self._press_rect.height()
        minimum = 12.0
        if "l" in self._active_handle:
            left = min(left + delta.x(), right - minimum)
        if "r" in self._active_handle:
            right = max(right + delta.x(), left + minimum)
        if "t" in self._active_handle:
            top = min(top + delta.y(), bottom - minimum)
        if "b" in self._active_handle:
            bottom = max(bottom + delta.y(), top + minimum)
        bounds = self.scene().sceneRect() if self.scene() else QRectF()
        if not bounds.isNull():
            left, top = max(bounds.left(), left), max(bounds.top(), top)
            right, bottom = min(bounds.right(), right), min(bounds.bottom(), bottom)
        self.setPos(left, top)
        self.setRect(0, 0, max(minimum, right - left), max(minimum, bottom - top))
        event.accept()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._active_handle = None
        self.sync_to_asset(record_undo=True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_pen()
        return super().itemChange(change, value)

    def sync_to_asset(self, record_undo=False):
        rect = self.mapRectToScene(self.rect())
        bbox = (round(rect.left()), round(rect.top()), max(1, round(rect.width())), max(1, round(rect.height())))
        if record_undo and bbox != self._press_bbox:
            self.changed_cb(self, before_change=True)
        self.asset.bbox = bbox
        self.asset.confirmed = False
        self.asset.status = "candidate"
        self.asset.segmentation_mode = "opencv"
        self.asset.confidence = None
        self.asset.model_id = None
        self.asset.image_path = None
        self.asset.mask_path = None
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
            """PPT 预览区上方会显示完整流程，提示每一步该做什么。

1. 如果源文件是 PDF，先点击“打开 PDF”，程序会直接载入页面图片，不生成中间 PPT。
2. 点击“打开 PPT”，导入图片型 PPT。
3. 程序会提取每页图片；请先在右侧选择本地或远端 OCR，再点击“开始 OCR（按当前选择）”识别文字区域。
4. OCR 完成后，在中间画布检查蓝色文字框，必要时调整、删除或新增框。
5. 如需把页面底图中的插图拆成独立图片，在右侧“图片拆分（AI 可选）”中调整候选框并确认图片。
6. 检查完成后，选择是否勾选“导出时清晰化底图（RealESRGAN）”，再点击右侧“继续：导出可编辑 PPT”或工具栏“导出可编辑 PPT”。""",
        ),
        (
            "PDF 导入",
            """PDF 导入：
点击工具栏“打开 PDF”，选择 .pdf 文件。程序会直接把 PDF 页面载入内部工作区，不会在 PDF 同目录生成 <PDF名>-from-pdf.pptx。

输出规则：
只有点击“导出可编辑 PPT”时，才会让你选择最终 PPT 的保存位置。

继续编辑：
PDF 载入完成后，源 PDF 会自动加入右侧“PPT 列表”并打开。检查 OCR 框后，点击“导出可编辑 PPT”即可继续。""",
        ),
        (
            "PPT 列表",
            """PPT 列表：
右侧上方会显示本次运行期间打开过的 PPT 和 PDF。

切换 PPT：
点击列表中的一个文件，程序会自动打开它并刷新左侧页列表和中间预览区。当前仍然一次只编辑一个文件。

自动加入：
点击“打开 PPT”选择的文件会加入列表。PDF 载入完成后，源 PDF 也会自动加入列表并打开。

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
            "图片拆分",
            """功能用途：
“图片拆分”用于把已经合成在 PPT 页面底图中的插图识别出来，导出后作为可以单独移动、缩放的 PPT 图片。它不是 OCR，也不是把任意人物照片自动处理成商业级透明素材；边缘效果取决于原图质量和候选框范围。

开始操作：
打开 PPT 或 PDF 并载入页面后，右侧会显示“图片拆分（AI 可选）”。点击“识别此页图片”后，程序会自动检测当前页的图片区候选，并在画布上显示紫色虚线框。颜色更深的紫色表示已确认图片，浅紫色表示尚未确认的候选；橙色框表示当前选中项。OCR 文字框使用其他颜色显示。

调整候选框：
先点击目标候选框。拖动框内区域可以移动位置；拖动四个角或上下左右边中间的控制点可以调整大小。候选框应覆盖完整插图，尽量不要包含旁边的图片、表格或大块背景。误检时点击“删除候选”；漏检时点击“添加图片区”，再手动移动和缩放新候选框。

AI 精细分割：
候选框调整好后，点击“AI 精细分割”。第一次使用需要下载约 156 MB 的 SAM 2.1 Tiny 模型，按提示选择保存目录并等待下载完成；模型准备好后不需要重启。程序会优先使用 CUDA，无法使用时尝试 CPU。分割成功后候选框会变成绿色，并记录 SAM 蒙版。

OpenCV 蒙版：
如果不想下载模型，或 AI 分割失败，可以点击“恢复 OpenCV 蒙版”。这会使用较快的本地颜色蒙版确认候选，适合边缘简单、颜色对比明显的插图；复杂边缘、相邻图片或背景颜色接近时，效果通常不如 SAM。

确认与导出：
只有深紫色的已确认候选才会参与导出，浅紫色的未确认候选会被忽略。点击“导出可编辑 PPT”后，程序会把已确认插图保存为透明 PNG，并作为独立图片重新放回 PPT；原底图中的对应区域会同时参与擦除和修复。图片上的文字如果已经被 OCR 识别，默认会保留文字在图片上方的层级，并尝试修复图片中的文字区域。

重新检测注意事项：
点击“识别此页图片”会根据当前页面重新生成候选框，可能替换当前页面已有的手动候选和调整结果。已经手动调好的候选不要随意重复识别；需要重新识别时，建议先确认页面内容再重新调整候选框。""",
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

清除当前页右下角水印区域：
导出时额外擦除右下角预设水印区域，比如 NotebookLM 标记。这个选项不依赖选中的文字框。

文本输入框：
显示并修改选中框导出后的文本内容。手动修正过的文本即使 OCR 置信度较低，也会参与导出重建。""",
        ),
        (
            "OCR 设置",
            """本地 OCR：
默认使用“本地 PaddleOCR”，不需要网络和令牌，适合离线使用。

远端 OCR：
如果本机配置较低，可以在右侧“OCR 设置”中切换为“远端 PaddleOCR”。远端 OCR 需要 40 位 PaddleOCR 访问令牌。

令牌管理：
点击“设置远端 OCR 令牌”可以保存令牌；点击“删除远端 OCR 令牌”会清除令牌，并自动切回本地 OCR。

开始 OCR：
打开 PPT 或 PDF 载入完成后，程序只提取页面图片，不会自动 OCR。请确认识别方式后点击“开始 OCR（按当前选择）”。如果选择远端 OCR 但令牌缺失或长度不正确，会提示先设置令牌或切换为本地 OCR。""",
        ),
        (
            "导出与清晰化",
            """导出顺序：
点击“导出可编辑 PPT”后，程序会先自动保存识别框，然后生成擦除蒙版，用 IOPaint/LaMa 擦除原图文字。若勾选“导出时清晰化底图（RealESRGAN）”，会再对清底图做 2x 清晰化，最后重建可编辑文本框并保存 PPT。

清晰化说明：
RealESRGAN 会提升底图分辨率和观感，但属于 AI 补细节，不等于还原真实原始细节。页数多或图片较大时，导出会更慢；不需要时可以取消勾选。

文字位置：
底图清晰化后像素尺寸会变大，但文本框仍按原始页面坐标映射，避免导出的文字位置缩偏。""",
        ),
        (
            "保存识别框",
            """保存识别框：
点击工具栏“保存识别框”，会把当前 PPT 的 OCR 框、手动新增框、文本修正、旋转信息、水印开关等保存到缓存文件。

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
拖拽、缩放、新增、删除、重算边距、修改文本。

不撤销：
水印区域开关。""",
        ),
        (
            "常见问题",
            """竖向文字识别不了：
这通常是 OCR 对局部旋转 90 度文字识别不稳定。可以手动新增窄高框，输入正确文本，再导出。

文字导出后不见：
请确认文本输入框里有正确内容，并且该框没有被删除。

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
    SETTINGS_ORG = "PPTtoEdit"
    SETTINGS_APP = "PPTEditableOCR"
    SAM_MODEL_DIRECTORY_KEY = "sam/model_directory"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPT 图片转可编辑文字 - MVP")
        self.resize(1500, 900)
        self.project: PPTProject | None = None
        self.current_slide: PPTSlide | None = None
        self.current_items: list[EditableRectItem] = []
        self.current_asset_items: list[EditableAssetItem] = []
        self.worker_thread: QThread | None = None
        self.worker: Worker | None = None
        self.selected_item: EditableRectItem | None = None
        self.selected_asset_item: EditableAssetItem | None = None
        self.undo_stacks: dict[int, list[BoxSnapshot]] = {}
        self.pending_select_index: int | None = None
        self.ppt_paths: list[Path] = []
        self.selecting_ppt_list_item = False
        self.pending_autoload_ppt: Path | None = None
        self.current_export_output: Path | None = None
        self.ocr_next_index = 0
        self.ocr_backend: str = OCR_BACKEND_LOCAL
        self.ocr_token: str | None = None
        self.pending_ocr_error: tuple[str, object | None] | None = None
        self.sam_engine: SamSegmentationEngine | None = None
        self.pending_sam_asset: VisualAsset | None = None
        self.sam_download_cancel: threading.Event | None = None
        self.sam_download_dialog: QProgressDialog | None = None
        self.sam_download_destination: Path | None = None
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)

        self._build_ui()
        self.load_ocr_settings()

    def _build_ui(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        open_action = QAction("打开 PPT", self)
        open_action.triggered.connect(self.open_ppt)
        pdf_to_ppt_action = QAction("打开 PDF", self)
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

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(8, 6, 8, 8)
        main_layout.setSpacing(6)
        root = QSplitter()
        main_layout.addWidget(root, 1)
        self.setCentralWidget(main)

        self.slide_list = QListWidget()
        self.slide_list.currentRowChanged.connect(self.on_slide_changed)
        root.addWidget(self.slide_list)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.flow_label = QLabel()
        self.flow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flow_label.setWordWrap(True)
        self.flow_label.setStyleSheet(
            "QLabel { color: #24292f; background: #f6f8fa; border: 1px solid #d0d7de; "
            "border-radius: 6px; padding: 8px; }"
        )
        center_layout.addWidget(self.flow_label)
        self.set_flow_text()
        self.scene = SlideScene()
        self.scene.selection_changed.connect(self.on_scene_selection_changed)
        self.view = PPTGraphicsView()
        self.view.setScene(self.scene)
        center_layout.addWidget(self.view)
        root.addWidget(center)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        ppt_group = QGroupBox("PPT 列表")
        ppt_group_layout = QVBoxLayout(ppt_group)
        self.ppt_file_list = QListWidget()
        self.ppt_file_list.setMaximumHeight(130)
        self.ppt_file_list.currentRowChanged.connect(self.on_ppt_file_selected)
        ppt_group_layout.addWidget(self.ppt_file_list)
        side_layout.addWidget(ppt_group)

        ocr_group = QGroupBox("OCR 设置")
        ocr_layout = QFormLayout(ocr_group)
        self.ocr_backend_combo = QComboBox()
        self.ocr_backend_combo.addItem("本地 PaddleOCR", OCR_BACKEND_LOCAL)
        self.ocr_backend_combo.addItem("远端 PaddleOCR", OCR_BACKEND_REMOTE)
        self.ocr_backend_combo.currentIndexChanged.connect(self.on_ocr_backend_changed)
        ocr_layout.addRow("识别方式", self.ocr_backend_combo)
        token_buttons = QWidget()
        token_buttons_layout = QHBoxLayout(token_buttons)
        token_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.ocr_token_btn = QPushButton("设置远端 OCR 令牌")
        self.ocr_token_btn.clicked.connect(self.configure_remote_ocr_token)
        token_buttons_layout.addWidget(self.ocr_token_btn)
        self.delete_ocr_token_btn = QPushButton("删除远端 OCR 令牌")
        self.delete_ocr_token_btn.clicked.connect(self.delete_remote_ocr_token)
        token_buttons_layout.addWidget(self.delete_ocr_token_btn)
        ocr_layout.addRow(token_buttons)
        self.start_ocr_btn = QPushButton("开始 OCR（按当前选择）")
        self.start_ocr_btn.clicked.connect(self.start_ocr)
        self.start_ocr_btn.setEnabled(False)
        ocr_layout.addRow(self.start_ocr_btn)
        self.ocr_token_status = QLabel("远端令牌未设置")
        self.ocr_token_status.setWordWrap(True)
        ocr_layout.addRow(self.ocr_token_status)
        side_layout.addWidget(ocr_group)

        self.asset_group = QGroupBox("图片拆分（AI 可选）")
        asset_layout = QVBoxLayout(self.asset_group)
        self.sam_status = QLabel()
        self.sam_status.setWordWrap(True)
        asset_layout.addWidget(self.sam_status)
        self.redetect_assets_btn = QPushButton("识别此页图片")
        self.redetect_assets_btn.setToolTip("自动识别当前页的图片区，并用紫色候选框标出")
        self.redetect_assets_btn.clicked.connect(self.redetect_visual_assets)
        asset_layout.addWidget(self.redetect_assets_btn)
        asset_row = QWidget()
        asset_row_layout = QHBoxLayout(asset_row)
        asset_row_layout.setContentsMargins(0, 0, 0, 0)
        self.add_asset_btn = QPushButton("添加图片区")
        self.add_asset_btn.clicked.connect(self.add_visual_asset)
        self.delete_asset_btn = QPushButton("删除候选")
        self.delete_asset_btn.clicked.connect(self.delete_selected_asset)
        asset_row_layout.addWidget(self.add_asset_btn)
        asset_row_layout.addWidget(self.delete_asset_btn)
        asset_layout.addWidget(asset_row)
        self.segment_asset_btn = QPushButton("AI 精细分割")
        self.segment_asset_btn.clicked.connect(self.segment_selected_asset)
        asset_layout.addWidget(self.segment_asset_btn)
        self.restore_asset_btn = QPushButton("恢复 OpenCV 蒙版")
        self.restore_asset_btn.clicked.connect(self.restore_selected_asset_opencv)
        asset_layout.addWidget(self.restore_asset_btn)
        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.redownload_model_btn = QPushButton("下载 / 重新下载模型")
        self.redownload_model_btn.clicked.connect(lambda: self.request_model_download(force=True))
        self.open_model_dir_btn = QPushButton("打开模型目录")
        self.open_model_dir_btn.clicked.connect(self.open_sam_model_dir)
        model_row_layout.addWidget(self.redownload_model_btn)
        model_row_layout.addWidget(self.open_model_dir_btn)
        asset_layout.addWidget(model_row)
        side_layout.addWidget(self.asset_group)
        self.update_sam_status()

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

        self.enhance_images_cb = QCheckBox("导出时清晰化底图（RealESRGAN）（可选）")
        self.enhance_images_cb.setChecked(False)
        side_layout.addWidget(self.enhance_images_cb)

        self.continue_export_btn = QPushButton("继续：导出可编辑 PPT")
        self.continue_export_btn.clicked.connect(self.export_ppt)
        self.continue_export_btn.setEnabled(False)
        side_layout.addWidget(self.continue_export_btn)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        side_layout.addWidget(self.log, 1)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side_scroll.setWidget(side)
        root.addWidget(self.side_scroll)
        root.setStretchFactor(1, 1)

    def set_flow_text(self):
        self.flow_label.setText(FLOW_TEXT)

    def load_ocr_settings(self):
        backend = self.settings.value("ocr/backend", OCR_BACKEND_LOCAL)
        if backend not in {OCR_BACKEND_LOCAL, OCR_BACKEND_REMOTE}:
            backend = OCR_BACKEND_LOCAL
        index = self.ocr_backend_combo.findData(backend)
        self.ocr_backend_combo.setCurrentIndex(max(0, index))
        self.update_ocr_token_status()

    def current_ocr_config(self) -> tuple[str, str | None]:
        backend = self.ocr_backend_combo.currentData() or OCR_BACKEND_LOCAL
        token = str(self.settings.value("ocr/remote_token", "") or "").strip()
        return backend, token if token else None

    def resolved_ocr_config(self) -> tuple[str, str | None, str | None]:
        backend, token = self.current_ocr_config()
        return resolve_ocr_config(
            selected_backend=backend,
            token=token,
            local_backend=OCR_BACKEND_LOCAL,
            remote_backend=OCR_BACKEND_REMOTE,
            token_length=REMOTE_OCR_TOKEN_LENGTH,
        )

    def on_ocr_backend_changed(self, _index: int = 0):
        backend = self.ocr_backend_combo.currentData() or OCR_BACKEND_LOCAL
        self.settings.setValue("ocr/backend", backend)
        self.update_ocr_token_status()

    def update_ocr_token_status(self):
        backend, token = self.current_ocr_config()
        has_token = bool(token and len(token) == REMOTE_OCR_TOKEN_LENGTH)
        self.ocr_token_btn.setEnabled(backend == OCR_BACKEND_REMOTE)
        self.delete_ocr_token_btn.setEnabled(bool(token))
        if backend == OCR_BACKEND_LOCAL:
            self.ocr_token_status.setText("当前使用本地 OCR" if not token else "已保存令牌，当前使用本地 OCR")
        elif has_token:
            self.ocr_token_status.setText("已设置 40 位令牌")
        else:
            self.ocr_token_status.setText("远端令牌未设置或长度不是 40 位")

    def configure_remote_ocr_token(self):
        current = str(self.settings.value("ocr/remote_token", "") or "").strip()
        token, accepted = QInputDialog.getText(
            self,
            "远端 OCR 令牌",
            "请输入 40 位 PaddleOCR 访问令牌：",
            QLineEdit.EchoMode.Password,
            current,
        )
        if not accepted:
            return
        token = token.strip()
        if len(token) != REMOTE_OCR_TOKEN_LENGTH:
            QMessageBox.warning(self, "令牌无效", "远端 OCR 访问令牌长度必须是 40 位。")
            return
        self.settings.setValue("ocr/remote_token", token)
        self.update_ocr_token_status()
        QMessageBox.information(self, "完成", "远端 OCR 令牌已保存。")

    def delete_remote_ocr_token(self):
        self.settings.remove("ocr/remote_token")
        local_index = self.ocr_backend_combo.findData(OCR_BACKEND_LOCAL)
        self.ocr_backend_combo.setCurrentIndex(max(0, local_index))
        self.settings.setValue("ocr/backend", OCR_BACKEND_LOCAL)
        self.update_ocr_token_status()
        QMessageBox.information(self, "完成", "远端 OCR 令牌已删除，后续 OCR 会使用本地 PaddleOCR。")

    def append_log(self, message: str):
        if message.startswith(PROGRESS_PREFIX):
            _prefix, percent, text = message.split("|", 2)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(percent))
            self.progress_label.setText(text)
            QApplication.processEvents()
            return
        if message.startswith(PAGE_READY_PREFIX):
            parts = message.split("|", 3)
            _prefix, index, box_count = parts[:3]
            status = parts[3] if len(parts) > 3 else "ok"
            self.upsert_slide_list_item(int(index), int(box_count), status)
            QApplication.processEvents()
            return
        self.log.appendPlainText(message)
        self.log.ensureCursorVisible()
        QApplication.processEvents()

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
        for button in (
            self.redetect_assets_btn, self.add_asset_btn, self.delete_asset_btn,
            self.segment_asset_btn, self.restore_asset_btn, self.redownload_model_btn,
            self.open_model_dir_btn,
        ):
            button.setEnabled(not busy)
        self.watermark_cb.setEnabled(not busy)
        self.text_edit.setEnabled(not busy)
        self.start_ocr_btn.setEnabled((not busy) and self.project is not None)
        self.continue_export_btn.setEnabled((not busy) and self.project is not None)
        self.save_cache_action.setEnabled((not busy) and self.project is not None)
        if busy:
            self.progress_label.setText("运行中...")
            self.progress_bar.setRange(0, 0)
        self.update_undo_action()

    def run_worker(self, fn, finished_cb, *args, failed_cb=None, **kwargs):
        if self.worker_thread:
            return
        self.worker_thread = QThread(self)
        self.worker = Worker(fn, *args, **kwargs)
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(finished_cb)
        self.worker.finished.connect(self.cleanup_worker)
        self.worker.failed.connect(lambda _msg, _exc: self.cleanup_worker())
        if failed_cb is None:
            self.worker.failed.connect(lambda msg, _exc: self.on_worker_failed(msg))
        else:
            self.worker.failed.connect(failed_cb)
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
        self.run_pending_work()
        self.update_undo_action()

    def run_pending_work(self):
        if self.worker_thread:
            return
        if self.pending_autoload_ppt:
            source = self.pending_autoload_ppt
            self.pending_autoload_ppt = None
            self.load_ppt_path(source, add_to_list=True, select_in_list=True)
            return

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
        self.progress_label.setText("准备载入 PDF")
        self.append_log(f"开始载入 PDF：{source}")
        self.add_ppt_to_recent_list(source, select=True)
        self.run_worker(
            prepare_pdf_project,
            self.on_pdf_loaded,
            source,
            auto_ocr=False,
        )

    def on_pdf_loaded(self, project: PPTProject):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("PDF 载入完成")
        self.append_log(f"PDF 载入完成：{project.source_pptx}")
        self.on_project_loaded(project)

    def open_ppt(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PPT 文件", "", "PowerPoint (*.pptx)")
        if not path:
            return
        self.load_source_path(Path(path), add_to_list=True, select_in_list=True)

    def add_ppt_to_recent_list(self, path: Path, select: bool = False) -> int:
        source = path.expanduser().resolve()
        try:
            row = self.ppt_paths.index(source)
        except ValueError:
            row = len(self.ppt_paths)
            self.ppt_paths.append(source)
            item = QListWidgetItem()
            item.setToolTip(str(source))
            self.ppt_file_list.addItem(item)
            widget = PptListItemWidget(
                source,
                lambda _checked=False, path=source: self.remove_ppt_from_recent_list_by_path(path),
                self.ppt_file_list,
            )
            item.setSizeHint(widget.sizeHint())
            self.ppt_file_list.setItemWidget(item, widget)
        if select:
            self.selecting_ppt_list_item = True
            self.ppt_file_list.setCurrentRow(row)
            self.selecting_ppt_list_item = False
        return row

    def remove_ppt_from_recent_list_by_path(self, path: Path):
        source = path.expanduser().resolve()
        try:
            row = self.ppt_paths.index(source)
        except ValueError:
            return
        self.remove_ppt_from_recent_list(row)

    def remove_ppt_from_recent_list(self, row: int):
        if self.worker_thread or row < 0 or row >= len(self.ppt_paths):
            return
        source = self.ppt_paths.pop(row)
        self.selecting_ppt_list_item = True
        self.ppt_file_list.takeItem(row)
        self.selecting_ppt_list_item = False
        if self.project and self.project.source_pptx == source:
            self.clear_current_project("已从 PPT 列表删除当前文件")
        elif self.ppt_file_list.count():
            next_row = min(row, self.ppt_file_list.count() - 1)
            self.ppt_file_list.setCurrentRow(next_row)

    def clear_current_project(self, progress_text: str | None = None):
        self.project = None
        self.current_slide = None
        self.selected_item = None
        self.selected_asset_item = None
        self.current_items.clear()
        getattr(self, "current_asset_items", []).clear()
        self.undo_stacks.clear()
        self.pending_select_index = None
        self.slide_list.clear()
        self.scene.clear()
        self.save_cache_action.setEnabled(False)
        self.start_ocr_btn.setEnabled(False)
        self.continue_export_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        if progress_text:
            self.progress_label.setText(progress_text)
        self.update_undo_action()

    def load_source_path(
        self,
        path: Path,
        add_to_list: bool = True,
        select_in_list: bool = True,
    ):
        source = path.expanduser().resolve()
        if source.suffix.lower() == ".pdf":
            if self.worker_thread:
                return
            if add_to_list:
                self.add_ppt_to_recent_list(source, select=select_in_list)
            self.clear_current_project()
            self.append_log(f"开始载入 PDF：{source}")
            self.run_worker(
                prepare_pdf_project,
                self.on_project_loaded,
                source,
                auto_ocr=False,
            )
            return
        self.load_ppt_path(source, add_to_list=add_to_list, select_in_list=select_in_list)

    def load_ppt_path(
        self,
        path: Path,
        add_to_list: bool = True,
        select_in_list: bool = True,
    ):
        source = path.expanduser().resolve()
        if self.worker_thread:
            return
        if add_to_list:
            self.add_ppt_to_recent_list(source, select=select_in_list)
        self.clear_current_project()
        self.append_log(f"开始加载：{source}")
        self.run_worker(
            prepare_project,
            self.on_project_loaded,
            source,
            auto_ocr=False,
        )

    def start_ocr(self):
        if not self.project:
            QMessageBox.information(self, "提示", "请先打开一个 PPT。")
            return
        if self.has_existing_ocr_result() and not self.confirm_restart_ocr():
            return
        ocr_backend, ocr_token, fallback_message = self.resolved_ocr_config()
        if fallback_message:
            QMessageBox.information(
                self,
                "OCR 未开始",
                f"{fallback_message}\n\n请设置远端 OCR 令牌，或切换为本地 PaddleOCR 后重试。",
            )
            return
        if ocr_backend == OCR_BACKEND_REMOTE and not (ocr_token and len(ocr_token) == REMOTE_OCR_TOKEN_LENGTH):
            QMessageBox.information(
                self,
                "OCR 未开始",
                "远端 OCR 令牌未设置或长度不正确。\n\n请设置远端 OCR 令牌，或切换为本地 PaddleOCR 后重试。",
            )
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.append_log("开始 OCR 识别")
        self.ocr_backend = ocr_backend
        self.ocr_token = ocr_token
        self.ocr_next_index = 0
        self.pending_ocr_error = None
        for slide in self.project.slides:
            slide.boxes = []
            slide.ocr_status = "pending"
        self.refresh_slide_list()
        if self.project.slides:
            self.slide_list.setCurrentRow(0)
        self.run_next_ocr_page()

    def has_existing_ocr_result(self) -> bool:
        if not self.project:
            return False
        return any(slide.boxes or slide.ocr_status in {"ok", "failed", "skipped"} for slide in self.project.slides)

    def current_cache_display_path(self) -> Path | None:
        if not self.project:
            return None
        candidates = cache_path_candidates(self.project.source_pptx)
        return next((path for path in candidates if path.exists()), candidates[0] if candidates else None)

    def confirm_restart_ocr(self) -> bool:
        cache_path = self.current_cache_display_path()
        location = str(cache_path) if cache_path else "当前项目缓存"
        answer = QMessageBox.question(
            self,
            "重新 OCR？",
            (
                "已经识别完成，识别结果文件在：\n"
                f"{location}\n\n"
                "你想重新识别吗？\n"
                "再次识别会替换现有 JSON 源文件。"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def refresh_slide_list(self):
        if not self.project:
            return
        self.slide_list.clear()
        for slide in self.project.slides:
            item = QListWidgetItem(page_list_text(slide.index, len(slide.boxes), slide.ocr_status))
            self.slide_list.addItem(item)

    def run_next_ocr_page(self):
        if not self.project:
            return
        if self.ocr_next_index >= len(self.project.slides):
            self.finish_ocr_flow()
            return
        slide = self.project.slides[self.ocr_next_index]
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(self.ocr_next_index * 100 / max(1, len(self.project.slides))))
        self.progress_label.setText(f"正在 OCR：第 {slide.index} 页")
        self.run_worker(
            run_ocr_page_subprocess,
            self.on_ocr_page_finished,
            slide,
            failed_cb=self.on_ocr_page_failed,
            ocr_backend=self.ocr_backend,
            ocr_token=self.ocr_token,
        )

    def on_ocr_page_finished(self, boxes: list[OCRBox]):
        if not self.project:
            return
        slide = self.project.slides[self.ocr_next_index]
        slide.boxes = boxes
        slide.ocr_status = "ok"
        self.update_slide_list_item(slide)
        if self.current_slide is slide:
            self.render_current_slide()
        self.append_log(f"第 {slide.index} 页 OCR 完成，共 {len(boxes)} 个框")
        self.save_current_cache(show_message=False, log_success=False)
        self.ocr_next_index += 1
        QTimer.singleShot(0, self.run_next_ocr_page)

    def on_ocr_page_failed(self, error_text: str, exc: object):
        self.pending_ocr_error = (error_text, exc)
        QTimer.singleShot(0, self.prompt_ocr_page_failure)

    def prompt_ocr_page_failure(self):
        if not self.project or not self.pending_ocr_error:
            return
        error_text, exc = self.pending_ocr_error
        self.pending_ocr_error = None
        slide = self.project.slides[self.ocr_next_index]
        slide.boxes = []
        slide.ocr_status = "failed"
        self.update_slide_list_item(slide)
        self.slide_list.setCurrentRow(self.ocr_next_index)
        self.current_slide = slide
        self.render_current_slide()
        output = getattr(exc, "output", "") if isinstance(exc, OCRPageProcessError) else ""
        self.append_log(f"第 {slide.index} 页 OCR 子进程异常，已暂停。")
        if output:
            self.append_log(output)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("OCR 失败")
        dialog.setText(f"第 {slide.index} 页 OCR 子进程异常退出。")
        dialog.setInformativeText("可以重试此页、跳过此页继续后续页面，或取消本次 OCR。")
        if output or error_text:
            dialog.setDetailedText(output or error_text)
        retry_btn = dialog.addButton("重试此页", QMessageBox.ButtonRole.AcceptRole)
        skip_btn = dialog.addButton("跳过此页", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = dialog.addButton("取消 OCR", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is retry_btn:
            slide.ocr_status = "pending"
            self.update_slide_list_item(slide)
            self.run_next_ocr_page()
        elif clicked is skip_btn:
            slide.ocr_status = "skipped"
            slide.boxes = []
            self.update_slide_list_item(slide)
            self.save_current_cache(show_message=False, log_success=False)
            self.ocr_next_index += 1
            QTimer.singleShot(0, self.run_next_ocr_page)
        elif clicked is cancel_btn:
            self.finish_ocr_flow(cancelled=True)

    def finish_ocr_flow(self, cancelled: bool = False):
        if not self.project:
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        if cancelled:
            self.progress_label.setText("OCR 已取消，已保留当前已完成页面。")
            self.append_log("OCR 已取消")
        else:
            self.progress_label.setText("OCR 完成，请检查识别框，然后点击“继续：导出可编辑 PPT”。")
            self.append_log("OCR 完成")
        self.start_ocr_btn.setEnabled(True)
        self.save_current_cache(show_message=False, success_prefix="识别结果已保存")
        self.update_undo_action()

    def on_ppt_file_selected(self, row: int):
        if self.selecting_ppt_list_item or self.worker_thread:
            return
        if row < 0 or row >= len(self.ppt_paths):
            return
        source = self.ppt_paths[row]
        if self.project and self.project.source_pptx == source:
            return
        self.load_source_path(source, add_to_list=False, select_in_list=False)

    def on_project_loaded(self, project: PPTProject):
        self.project = project
        self.undo_stacks.clear()
        self.slide_list.clear()
        for slide in project.slides:
            if slide.boxes and slide.ocr_status == "pending":
                slide.ocr_status = "ok"
            item = QListWidgetItem(page_list_text(slide.index, len(slide.boxes), slide.ocr_status))
            self.slide_list.addItem(item)
        if project.slides:
            self.slide_list.setCurrentRow(0)
        self.append_log("项目加载完成")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        if any(slide.boxes for slide in project.slides):
            self.progress_label.setText("项目加载完成，已加载识别框。请检查后点击“继续：导出可编辑 PPT”。")
        else:
            self.progress_label.setText("项目加载完成，请选择 OCR 方式后点击“开始 OCR（按当前选择）”。")
        self.save_cache_action.setEnabled(True)
        self.start_ocr_btn.setEnabled(True)
        self.continue_export_btn.setEnabled(True)
        self.update_undo_action()

    def save_current_cache(
        self,
        show_message: bool = True,
        log_success: bool = True,
        success_prefix: str = "识别框已保存",
    ):
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
        if log_success:
            self.append_log(f"{success_prefix}：{cache_path}")
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
        self.current_asset_items.clear()
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
        for asset in slide.visual_assets:
            if not asset.enabled:
                continue
            x, y, _width, _height = asset.bbox
            if asset.confirmed and asset.image_path and asset.image_path.is_file():
                preview = QGraphicsPixmapItem(QPixmap(str(asset.image_path)))
                preview.setPos(x, y)
                preview.setOpacity(0.55)
                preview.setZValue(6)
                self.scene.addItem(preview)
            asset_item = EditableAssetItem(asset, self.on_asset_item_changed)
            asset_item.setToolTip(
                f"图片区候选：{asset.status}\n分割：{asset.segmentation_mode}\n层级：{'图片在上' if asset.layer == 'above_text' else '文字在上'}"
            )
            asset_item.setZValue(7)
            self.scene.addItem(asset_item)
            self.current_asset_items.append(asset_item)
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
            (
                box.text,
                box.score,
                box.bbox,
                box.erase_rect,
                box.enabled,
                box.manual,
                box.edited,
                box.rotation,
                box.line_height,
                box.text_regions,
                box.mask_mode,
                box.mask_reason,
            )
            for box in slide.boxes
        ]

    def snapshot_assets(self, slide: PPTSlide) -> list[dict]:
        return [visual_asset_to_data(asset) for asset in slide.visual_assets]

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
                line_height=line_height,
                text_regions=text_regions,
                mask_mode=mask_mode,
                mask_reason=mask_reason,
            )
            for text, score, bbox, erase_rect, enabled, manual, edited, rotation, line_height, text_regions, mask_mode, mask_reason in snapshot
        ]

    def push_undo_state(self):
        if not self.current_slide:
            return
        key = self.slide_key(self.current_slide)
        if key is None:
            return
        stack = self.undo_stacks.setdefault(key, [])
        stack.append((self.snapshot_boxes(self.current_slide), self.snapshot_assets(self.current_slide)))
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
            item.setText(page_list_text(slide.index, len(slide.boxes), slide.ocr_status))

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
        if isinstance(snapshot, tuple) and len(snapshot) == 2:
            box_snapshot, asset_snapshot = snapshot
            self.restore_boxes(self.current_slide, box_snapshot)
            self.current_slide.visual_assets = [visual_asset_from_data(item) for item in asset_snapshot]
        else:
            self.restore_boxes(self.current_slide, snapshot)
        self.pending_select_index = selected_index
        self.update_slide_list_item(self.current_slide)
        self.render_current_slide()

    def on_item_changed(self, _item: EditableRectItem, before_change: bool = False):
        if before_change:
            self.push_undo_state()
        self.on_scene_selection_changed()

    def on_asset_item_changed(self, _item: EditableAssetItem, before_change: bool = False):
        if before_change:
            self.push_undo_state()
        self.on_scene_selection_changed()

    def on_scene_selection_changed(self):
        selected = self.scene.selectedItems()
        items = [item for item in selected if isinstance(item, EditableRectItem)]
        asset_items = [item for item in selected if isinstance(item, EditableAssetItem)]
        self.selected_item = items[0] if items else None
        self.selected_asset_item = asset_items[0] if asset_items else None
        if self.selected_asset_item:
            asset = self.selected_asset_item.asset
            confidence = "--" if asset.confidence is None else f"{asset.confidence:.3f}"
            self.selected_info.setText(
                f"图片区：{asset.asset_id}\n范围：{asset.bbox}\n状态：{asset.status}\n"
                f"分割：{asset.segmentation_mode}\n置信度：{confidence}"
            )
            self.text_edit.blockSignals(True)
            self.text_edit.clear()
            self.text_edit.blockSignals(False)
            return
        if not self.selected_item:
            self.selected_info.setText("未选择任何框")
            self.text_edit.blockSignals(True)
            self.text_edit.clear()
            self.text_edit.blockSignals(False)
            return
        box = self.selected_item.box
        self.selected_info.setText(
            f"文本：{box.text}\n"
            f"置信度：{box.score:.3f}\n"
            f"原始 bbox：{box.bbox}\n"
            f"擦除框：{box.erase_rect}\n"
            f"擦除模式：{'精细文字蒙版' if box.mask_mode == 'text_stroke' or box.text_regions else '矩形回退'}"
            + (f"\n原因：{box.mask_reason or '缺少 OCR 文字轮廓'}" if box.mask_reason or not box.text_regions else "")
        )
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

    def update_sam_status(self):
        path = self.current_sam_model_path()
        if verify_model(path):
            device = preferred_device().upper()
            self.sam_status.setText(f"{MODEL_ID}：已就绪（无需重启）\n推理设备：{device}\n{path}")
        else:
            self.sam_status.setText(
                f"{MODEL_ID}：未下载（首次使用约 156 MB）\n"
                f"当前目录：{path.parent}\n"
                "点击“下载 / 重新下载模型”选择目录并开始下载。"
            )

    def current_sam_model_path(self) -> Path:
        configured = str(self.settings.value(self.SAM_MODEL_DIRECTORY_KEY, "") or "").strip()
        return Path(configured) / MODEL_FILENAME if configured else model_path()

    def open_sam_model_dir(self):
        directory = self.current_sam_model_path().parent
        directory.mkdir(parents=True, exist_ok=True)
        os.startfile(str(directory))

    def request_model_download(self, force: bool = False):
        destination = self.current_sam_model_path()
        if not force and verify_model(destination):
            self.on_sam_model_ready(destination)
            return
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "选择 SAM 2.1 模型保存目录",
            str(destination.parent),
        )
        if not selected_directory:
            self.pending_sam_asset = None
            return
        destination = Path(selected_directory) / MODEL_FILENAME
        answer = QMessageBox.question(
            self,
            "下载 SAM 2.1 模型",
            f"AI 精细分割需要下载约 156 MB 的 {MODEL_ID} 模型。\n\n保存位置：\n{destination}\n\n是否开始下载？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.pending_sam_asset = None
            return
        previous_path = self.current_sam_model_path()
        self.settings.setValue(self.SAM_MODEL_DIRECTORY_KEY, str(destination.parent))
        self.sam_download_destination = destination
        if previous_path != destination:
            self.sam_engine = None
        self.update_sam_status()
        if force and destination.exists():
            destination.unlink()
        self.sam_download_cancel = threading.Event()
        self.sam_download_dialog = QProgressDialog("正在下载 SAM 2.1 模型...", "取消", 0, 0, self)
        self.sam_download_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.sam_download_dialog.canceled.connect(self.sam_download_cancel.set)
        self.sam_download_dialog.show()
        self.run_worker(
            self._download_sam_model,
            self.on_sam_model_ready,
            failed_cb=self.on_sam_download_failed,
        )

    def _download_sam_model(self, progress=None):
        def report(done: int, total: int):
            if progress:
                percent = int(done * 100 / total) if total else 0
                progress(f"SAM 2.1 模型下载：{percent}% ({done // 1048576} MB)")

        cancelled = self.sam_download_cancel.is_set if self.sam_download_cancel else None
        destination = self.sam_download_destination or self.current_sam_model_path()
        return download_model(destination=destination, progress=report, cancelled=cancelled)

    def on_sam_model_ready(self, _path):
        if self.sam_download_dialog:
            self.sam_download_dialog.close()
        self.sam_download_dialog = None
        self.sam_download_cancel = None
        self.sam_download_destination = None
        self.update_sam_status()
        self.append_log("SAM 2.1 模型已就绪，无需重启，可直接使用")
        pending = self.pending_sam_asset
        self.pending_sam_asset = None
        if pending and self.current_slide and pending in self.current_slide.visual_assets:
            self._start_asset_segmentation(pending)

    def on_sam_download_failed(self, message: str, exc):
        if self.sam_download_dialog:
            self.sam_download_dialog.close()
        self.sam_download_dialog = None
        self.sam_download_cancel = None
        self.sam_download_destination = None
        self.pending_sam_asset = None
        self.append_log(message)
        self.update_sam_status()
        if "取消" not in str(exc):
            QMessageBox.warning(self, "模型下载失败", str(exc))

    def redetect_visual_assets(self):
        if not self.current_slide:
            return
        self.push_undo_state()
        with Image.open(self.current_slide.image_path) as source:
            image = np.asarray(source.convert("RGB"))
        self.current_slide.visual_assets = detect_visual_assets(image, self.current_slide)
        self.append_log(f"第 {self.current_slide.index} 页已识别出 {len(self.current_slide.visual_assets)} 个图片区候选")
        self.render_current_slide()

    def add_visual_asset(self):
        if not self.current_slide:
            return
        self.push_undo_state()
        width = min(360, max(40, self.current_slide.image_width // 3))
        height = min(260, max(40, self.current_slide.image_height // 3))
        left = max(0, (self.current_slide.image_width - width) // 2)
        top = max(0, (self.current_slide.image_height - height) // 2)
        asset = VisualAsset(
            asset_id=f"slide-{self.current_slide.index}-visual-{len(self.current_slide.visual_assets) + 1}",
            bbox=(left, top, width, height),
            source="manual",
        )
        self.current_slide.visual_assets.append(asset)
        self.render_current_slide()

    def delete_selected_asset(self):
        if not self.current_slide or not self.selected_asset_item:
            return
        self.push_undo_state()
        self.current_slide.visual_assets.remove(self.selected_asset_item.asset)
        self.render_current_slide()

    def restore_selected_asset_opencv(self):
        if not self.selected_asset_item:
            return
        self.push_undo_state()
        asset = self.selected_asset_item.asset
        asset.segmentation_mode = "opencv"
        asset.confidence = None
        asset.model_id = None
        asset.mask_version = 0
        asset.image_path = None
        asset.mask_path = None
        asset.confirmed = True
        asset.status = "confirmed"
        self.render_current_slide()

    def segment_selected_asset(self):
        if not self.current_slide or not self.selected_asset_item:
            QMessageBox.information(self, "提示", "请先选择一个紫色图片区候选框。")
            return
        asset = self.selected_asset_item.asset
        if not verify_model(self.current_sam_model_path()):
            self.pending_sam_asset = asset
            self.request_model_download()
            return
        self._start_asset_segmentation(asset)

    def _start_asset_segmentation(self, asset: VisualAsset):
        if not self.current_slide or not self.project:
            return
        self.push_undo_state()
        asset.status = "segmenting"
        slide = self.current_slide
        project = self.project
        self.render_current_slide()
        self.run_worker(
            self._run_asset_segmentation,
            self.on_asset_segmented,
            project,
            slide,
            asset,
            failed_cb=lambda message, exc: self.on_asset_segmentation_failed(asset, message, exc),
        )

    def _run_asset_segmentation(self, project, slide, asset, progress=None):
        if progress:
            progress(f"第 {slide.index} 页正在执行 SAM 2.1 精细分割...")
        with Image.open(slide.image_path) as source:
            image = np.asarray(source.convert("RGB"))
        if self.sam_engine is None:
            self.sam_engine = SamSegmentationEngine(self.current_sam_model_path(), device="auto")
        x, y, width, height = asset.bbox
        result = self.sam_engine.segment_with_box(
            image,
            (x, y, x + width, y + height),
            image_key=str(slide.image_path),
        )
        store_visual_asset_mask(project, slide, asset, result)
        return asset

    def on_asset_segmented(self, asset):
        self.append_log(
            f"AI 精细分割完成：{asset.asset_id}，置信度 {asset.confidence:.3f}，设备 {self.sam_engine.device}"
        )
        self.save_current_cache(show_message=False, log_success=False)
        self.render_current_slide()

    def on_asset_segmentation_failed(self, asset, message: str, exc):
        asset.status = "fallback"
        asset.segmentation_mode = "opencv"
        asset.confirmed = True
        asset.confidence = None
        asset.model_id = None
        self.append_log(message)
        self.render_current_slide()
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            detail = (
                "SAM 2.1 运行组件未完整加载。请重新安装或修复程序依赖，"
                "然后重启本程序再试。"
            )
        else:
            detail = str(exc)
        QMessageBox.warning(self, "AI 分割失败", f"已回退 OpenCV 蒙版，可继续导出。\n\n{detail}")

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
        self.start_export_to_path(Path(out_path))

    def start_export_to_path(self, out_path: Path):
        if not self.project:
            return
        self.current_export_output = out_path
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备导出")
        self.save_current_cache(show_message=False, log_success=False)
        self.append_log("开始导出可编辑 PPT")
        self.run_worker(
            run_export_editable_ppt_subprocess,
            self.on_export_finished,
            self.project,
            out_path,
            enhance_images=self.enhance_images_cb.isChecked(),
        )

    def on_export_finished(self, _result):
        output = self.current_export_output
        if output:
            self.append_log(f"可编辑 PPT 已导出：{output}")
        self.current_export_output = None
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
