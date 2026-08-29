import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core import OCRBox, PPTSlide, VisualAsset
from app.gui import EditableAssetItem, EditableRectItem, MainWindow
from PySide6.QtCore import QEventLoop, QPointF, QRectF, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea


class VisualAssetGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_box_edges_have_independent_resize_handles(self):
        box = OCRBox(text="", score=1.0, bbox=(10, 20, 100, 60), erase_rect=(10, 20, 110, 80))
        text_item = EditableRectItem(box, lambda *_args, **_kwargs: None)
        asset = VisualAsset(asset_id="device", bbox=(10, 20, 100, 60))
        asset_item = EditableAssetItem(asset, lambda *_args, **_kwargs: None)

        for item in (text_item, asset_item):
            rect = item.rect()
            self.assertEqual(item._handle_at(QPointF(rect.center().x(), rect.top())), "t")
            self.assertEqual(item._handle_at(QPointF(rect.right(), rect.center().y())), "r")
            self.assertEqual(item._handle_at(QPointF(rect.center().x(), rect.bottom())), "b")
            self.assertEqual(item._handle_at(QPointF(rect.left(), rect.center().y())), "l")

    def test_asset_resize_handle_hit_areas_extend_beyond_the_border(self):
        asset_item = EditableAssetItem(
            VisualAsset(asset_id="device", bbox=(10, 20, 100, 60)),
            lambda *_args, **_kwargs: None,
        )
        rect = asset_item.rect()
        points = (
            QPointF(rect.center().x(), rect.top() - 4),
            QPointF(rect.right() + 4, rect.center().y()),
            QPointF(rect.center().x(), rect.bottom() + 4),
            QPointF(rect.left() - 4, rect.center().y()),
        )
        for point in points:
            self.assertTrue(asset_item.shape().contains(point))

    def test_editing_candidate_bbox_invalidates_old_ai_mask(self):
        changes = []
        asset = VisualAsset(
            asset_id="device",
            bbox=(10, 20, 30, 40),
            confirmed=True,
            segmentation_mode="sam2",
            image_path=Path("device.png"),
            mask_path=Path("device.mask.png"),
        )
        item = EditableAssetItem(asset, lambda _item, before_change=False: changes.append(before_change))
        item.setPos(15, 25)
        item.setRect(QRectF(0, 0, 50, 60))
        item.sync_to_asset(record_undo=True)

        self.assertEqual(asset.bbox, (15, 25, 50, 60))
        self.assertFalse(asset.confirmed)
        self.assertEqual(asset.segmentation_mode, "opencv")
        self.assertIsNone(asset.mask_path)
        self.assertEqual(changes, [True, False])

    def test_restore_opencv_confirms_selected_candidate_for_export(self):
        asset = VisualAsset(
            asset_id="device",
            bbox=(0, 0, 20, 20),
            segmentation_mode="sam2",
            confirmed=True,
            confidence=0.9,
            image_path=Path("device.png"),
            mask_path=Path("device.mask.png"),
        )
        window = MainWindow.__new__(MainWindow)
        window.selected_asset_item = SimpleNamespace(asset=asset)
        window.push_undo_state = lambda: None
        window.render_current_slide = lambda: None

        window.restore_selected_asset_opencv()

        self.assertEqual(asset.segmentation_mode, "opencv")
        self.assertTrue(asset.confirmed)
        self.assertEqual(asset.status, "confirmed")
        self.assertIsNone(asset.mask_path)

    def test_add_visual_asset_creates_unconfirmed_manual_candidate(self):
        slide = PPTSlide(
            index=2,
            image_name="slide.png",
            image_path=Path("slide.png"),
            image_width=900,
            image_height=600,
        )
        window = MainWindow.__new__(MainWindow)
        window.current_slide = slide
        window.push_undo_state = lambda: None
        window.render_current_slide = lambda: None

        window.add_visual_asset()

        self.assertEqual(len(slide.visual_assets), 1)
        self.assertEqual(slide.visual_assets[0].source, "manual")
        self.assertFalse(slide.visual_assets[0].confirmed)

    def test_sidebar_scrolls_instead_of_squashing_ai_controls(self):
        window = MainWindow()
        window.resize(900, 520)
        window.show()
        self.app.processEvents()

        scroll_areas = window.findChildren(QScrollArea)
        self.assertEqual(len(scroll_areas), 1)
        sidebar = scroll_areas[0]
        self.assertTrue(sidebar.widgetResizable())
        self.assertEqual(
            sidebar.horizontalScrollBarPolicy(),
            sidebar.horizontalScrollBarPolicy().ScrollBarAlwaysOff,
        )
        self.assertGreater(sidebar.verticalScrollBar().maximum(), 0)
        self.assertGreaterEqual(
            window.asset_group.height(),
            window.asset_group.minimumSizeHint().height(),
        )
        self.assertEqual(window.redownload_model_btn.text(), "下载 / 重新下载模型")

        window.close()

    def test_model_download_uses_and_remembers_selected_directory(self):
        class FakeSettings:
            def __init__(self):
                self.values = {}

            def value(self, key, default=None):
                return self.values.get(key, default)

            def setValue(self, key, value):
                self.values[key] = value

        window = MainWindow()
        window.settings = FakeSettings()
        worker_calls = []
        window.run_worker = lambda *args, **kwargs: worker_calls.append((args, kwargs))

        with (
            patch("app.gui.QFileDialog.getExistingDirectory", return_value="D:/PPT-AI-Models") as choose_dir,
            patch("app.gui.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes),
        ):
            window.request_model_download(force=True)

        choose_dir.assert_called_once()
        self.assertEqual(window.settings.values["sam/model_directory"], "D:\\PPT-AI-Models")
        self.assertEqual(
            window.sam_download_destination,
            Path("D:/PPT-AI-Models/sam2.1_hiera_tiny.pt"),
        )
        self.assertEqual(len(worker_calls), 1)

        if window.sam_download_dialog:
            window.sam_download_dialog.close()
        window.close()

    def test_missing_sam_runtime_prompts_for_repair_and_restart(self):
        asset = VisualAsset(asset_id="device", bbox=(0, 0, 20, 20))
        window = MainWindow.__new__(MainWindow)
        window.append_log = lambda _message: None
        window.render_current_slide = lambda: None

        with patch("app.gui.QMessageBox.warning") as warning:
            window.on_asset_segmentation_failed(
                asset,
                "SAM runtime import failed",
                ModuleNotFoundError("No module named 'sam2'"),
            )

        message = warning.call_args.args[2]
        self.assertIn("重新安装", message)
        self.assertIn("重启", message)

    def test_ready_model_status_shows_cuda_device(self):
        window = MainWindow()
        with (
            patch("app.gui.verify_model", return_value=True),
            patch("app.gui.preferred_device", return_value="cuda", create=True),
        ):
            window.update_sam_status()

        self.assertIn("CUDA", window.sam_status.text())
        window.close()

    def test_worker_is_cleaned_up_before_failure_dialog_callback(self):
        window = MainWindow()
        loop = QEventLoop()
        worker_thread_seen_by_failure = []

        def fail(progress=None):
            raise RuntimeError("probe failure")

        def on_failed(_message, _exc):
            worker_thread_seen_by_failure.append(window.worker_thread)
            loop.quit()

        window.run_worker(fail, lambda _result: None, failed_cb=on_failed)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        self.app.processEvents()

        self.assertEqual(worker_thread_seen_by_failure, [None])
        window.close()


if __name__ == "__main__":
    unittest.main()
