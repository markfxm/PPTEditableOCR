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
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .core import OCRBox, PPTProject, PPTSlide, export_editable_ppt, prepare_project


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
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._active_handle:
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self._press_scene_pos
        rect = QRectF(self._press_rect)
        pos = QPointF(self._press_pos)

        if "l" in self._active_handle:
            new_left = rect.left() + delta.x()
            max_left = rect.right() - 12
            pos.setX(pos.x() + min(new_left - rect.left(), max_left - rect.left()))
            rect.setLeft(max(0, min(new_left, max_left)))
        if "r" in self._active_handle:
            rect.setRight(max(rect.left() + 12, rect.right() + delta.x()))
        if "t" in self._active_handle:
            new_top = rect.top() + delta.y()
            max_top = rect.bottom() - 12
            pos.setY(pos.y() + min(new_top - rect.top(), max_top - rect.top()))
            rect.setTop(max(0, min(new_top, max_top)))
        if "b" in self._active_handle:
            rect.setBottom(max(rect.top() + 12, rect.bottom() + delta.y()))

        self.setPos(pos)
        self.setRect(0, 0, rect.width(), rect.height())
        event.accept()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._active_handle = None
        self.sync_to_box()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_pen()
        return super().itemChange(change, value)

    def sync_to_box(self):
        rect = self.sceneBoundingRect()
        self.box.set_erase_rect((round(rect.left()), round(rect.top()), round(rect.right()), round(rect.bottom())))
        self._update_pen()
        self.changed_cb(self)


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


class MainWindow(QMainWindow):
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

        self._build_ui()

    def _build_ui(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        open_action = QAction("打开 PPT", self)
        open_action.triggered.connect(self.open_ppt)
        export_action = QAction("导出可编辑 PPT", self)
        export_action.triggered.connect(self.export_ppt)
        toolbar.addAction(open_action)
        toolbar.addAction(export_action)

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

        self.reset_slide_btn = QPushButton("当前页按边距重算")
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

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        side_layout.addWidget(self.log, 1)
        root.addWidget(side)
        root.setStretchFactor(1, 1)

    def append_log(self, message: str):
        self.log.appendPlainText(message)

    def set_busy(self, busy: bool):
        self.slide_list.setEnabled(not busy)
        self.reset_slide_btn.setEnabled(not busy)
        self.reset_box_btn.setEnabled(not busy)
        self.delete_box_btn.setEnabled(not busy)
        self.add_box_btn.setEnabled(not busy)
        self.toggle_box_cb.setEnabled(not busy)
        self.watermark_cb.setEnabled(not busy)
        self.text_edit.setEnabled(not busy)

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

    def on_worker_failed(self, error_text: str):
        self.append_log(error_text)
        QMessageBox.critical(self, "运行失败", error_text)

    def open_ppt(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PPT 文件", "", "PowerPoint (*.pptx *.ppt)")
        if not path:
            return
        source = Path(path)
        self.append_log(f"开始加载：{source}")
        self.run_worker(prepare_project, self.on_project_loaded, source)

    def on_project_loaded(self, project: PPTProject):
        self.project = project
        self.slide_list.clear()
        for slide in project.slides:
            item = QListWidgetItem(f"第 {slide.index} 页 - {len(slide.boxes)} 个框")
            self.slide_list.addItem(item)
        if project.slides:
            self.slide_list.setCurrentRow(0)
        self.append_log("项目加载完成")

    def on_slide_changed(self, row: int):
        if not self.project or row < 0 or row >= len(self.project.slides):
            return
        self.current_slide = self.project.slides[row]
        self.render_current_slide()

    def render_current_slide(self):
        slide = self.current_slide
        if not slide:
            return
        self.scene.clear()
        self.current_items.clear()
        pixmap = QPixmap(str(slide.image_path))
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)
        self.scene.setSceneRect(pixmap.rect())
        for box in slide.boxes:
            item = EditableRectItem(box, self.on_item_changed)
            self.scene.addItem(item)
            self.current_items.append(item)
        self.watermark_cb.blockSignals(True)
        self.watermark_cb.setChecked(slide.remove_watermark)
        self.watermark_cb.blockSignals(False)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.on_scene_selection_changed()

    def on_item_changed(self, _item: EditableRectItem):
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
        self.current_slide.reset_boxes(self.pad_x.value(), self.pad_y.value())
        self.render_current_slide()

    def reset_selected_box(self):
        if not self.current_slide or not self.selected_item:
            return
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
        self.current_slide.boxes.remove(self.selected_item.box)
        self.render_current_slide()

    def add_box(self):
        if not self.current_slide:
            return
        width = min(280, self.current_slide.image_width - 40)
        height = 60
        left = max(20, (self.current_slide.image_width - width) // 2)
        top = max(20, (self.current_slide.image_height - height) // 2)
        box = OCRBox(
            text="新文本",
            score=1.0,
            bbox=(left, top, width, height),
            erase_rect=(left, top, left + width, top + height),
            enabled=True,
        )
        self.current_slide.boxes.append(box)
        self.render_current_slide()

    def on_text_edited(self):
        if not self.selected_item:
            return
        self.selected_item.box.text = self.text_edit.text().strip() or "新文本"
        self.on_scene_selection_changed()

    def on_selected_box_toggle(self, state: int):
        if not self.selected_item:
            return
        self.selected_item.box.enabled = state == Qt.CheckState.Checked.value
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
        self.append_log("开始导出可编辑 PPT")
        self.run_worker(export_editable_ppt, self.on_export_finished, self.project, Path(out_path))

    def on_export_finished(self, _result):
        self.append_log("导出完成")
        QMessageBox.information(self, "完成", "可编辑 PPT 已导出完成。")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
