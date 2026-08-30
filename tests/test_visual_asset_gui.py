import os
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from app.core import OCRBox, PPTSlide, VisualAsset
from app.gui import EditableAssetItem, EditableRectItem, MainWindow, SamMaskReviewDialog
from app.sam_segmentation import SegmentationResult
from PySide6.QtCore import QEventLoop, QPointF, QRectF, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QProgressBar, QScrollArea


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

    def test_sam_review_dialog_switches_candidates_and_shows_selected_score(self):
        source = np.full((80, 100, 3), 240, dtype=np.uint8)
        masks = np.zeros((3, 80, 100), dtype=np.uint8)
        masks[0, 12:35, 12:45] = 255
        masks[1, 11:38, 11:48] = 255
        masks[1, 20:30, 11:22] = 0
        masks[1, 20:30, 37:48] = 0
        masks[2, 15:30, 15:35] = 255
        result = SegmentationResult(masks, np.asarray([0.2, 0.9, 0.1]), 1, "cpu")
        asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))

        dialog = SamMaskReviewDialog(source, asset, result)
        self.assertIn("候选 2/3", dialog.candidate_label.text())
        self.assertIn("0.900", dialog.candidate_label.text())

        dialog.next_candidate()
        self.assertEqual(dialog.result.selected_index, 2)
        self.assertIn("候选 3/3", dialog.candidate_label.text())
        self.assertFalse(dialog.confirm_btn.isEnabled())
        dialog.previous_candidate()
        self.assertEqual(dialog.result.selected_index, 1)
        self.assertTrue(dialog.confirm_btn.isEnabled())
        dialog.close()

    def test_sam_review_dialog_tracks_positive_negative_points_and_can_clear_them(self):
        source = np.full((80, 100, 3), 240, dtype=np.uint8)
        mask = np.zeros((1, 80, 100), dtype=np.uint8)
        mask[0, 12:35, 12:45] = 255
        result = SegmentationResult(mask, np.asarray([0.8]), 0, "cpu")
        asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))
        changes = []
        dialog = SamMaskReviewDialog(source, asset, result, regenerate_cb=lambda points: changes.append(tuple(points)))

        dialog.add_prompt_point(20.0, 20.0, 1)
        dialog.add_prompt_point(8.0, 8.0, 0)
        self.assertEqual(dialog.points, [(20.0, 20.0, 1), (8.0, 8.0, 0)])
        dialog.undo_prompt_point()
        self.assertEqual(dialog.points, [(20.0, 20.0, 1)])
        dialog.clear_prompt_points()
        self.assertEqual(dialog.points, [])
        self.assertEqual(len(changes), 4)
        dialog.close()

    def test_sam_prompt_refresh_keeps_current_preview_zoom(self):
        source = np.full((80, 100, 3), 240, dtype=np.uint8)
        mask = np.zeros((1, 80, 100), dtype=np.uint8)
        mask[0, 12:35, 12:45] = 255
        result = SegmentationResult(mask, np.asarray([0.8]), 0, "cpu")
        asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))
        dialog = SamMaskReviewDialog(source, asset, result)
        dialog.preview_view.scale(1.4, 1.4)
        zoom_before = dialog.preview_view.transform().m11()

        dialog.add_prompt_point(20.0, 20.0, 1)

        self.assertAlmostEqual(dialog.preview_view.transform().m11(), zoom_before, places=5)
        self.assertEqual(dialog.points, [(20.0, 20.0, 1)])
        dialog.close()

    def test_sam_review_fits_preview_after_dialog_layout(self):
        source = np.full((800, 600, 3), 240, dtype=np.uint8)
        masks = np.zeros((1, 800, 600), dtype=np.uint8)
        masks[0, 100:700, 100:500] = 255
        result = SegmentationResult(masks, np.asarray([0.8]), 0, "cpu")
        asset = VisualAsset(asset_id="device", bbox=(100, 100, 400, 600))

        dialog = SamMaskReviewDialog(source, asset, result)
        dialog.resize(900, 700)
        dialog.show()
        self.app.processEvents()

        self.assertGreater(dialog.preview_view.transform().m11(), 0.5)
        dialog.close()

    def test_sam_review_busy_state_keeps_preview_position(self):
        source = np.full((80, 100, 3), 240, dtype=np.uint8)
        mask = np.zeros((1, 80, 100), dtype=np.uint8)
        mask[0, 12:35, 12:45] = 255
        result = SegmentationResult(mask, np.asarray([0.8]), 0, "cpu")
        asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))

        dialog = SamMaskReviewDialog(source, asset, result, regenerate_cb=lambda _points: None)
        dialog.show()
        self.app.processEvents()
        preview_top = dialog.preview_view.geometry().top()

        dialog.set_busy(True)
        self.app.processEvents()

        self.assertEqual(dialog.preview_view.geometry().top(), preview_top)
        dialog.close()

    def test_start_and_cancel_sam_review_leave_original_asset_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "slide.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            asset = VisualAsset(
                asset_id="device",
                bbox=(10, 10, 40, 30),
                source="manual",
                status="candidate",
                confirmed=False,
            )
            slide = PPTSlide(1, "slide.png", image_path, 100, 80, visual_assets=[asset])
            window = MainWindow.__new__(MainWindow)
            window.current_slide = slide
            window.project = SimpleNamespace(work_dir=root)
            window.pending_sam_asset = None
            window.pending_sam_points = []
            window.sam_review_dialog = None
            window.progress_label = QLabel("运行中...")
            window.progress_bar = QProgressBar()
            window.progress_bar.setRange(0, 0)
            window.run_worker = lambda *_args, **_kwargs: None
            before = asset.__dict__.copy()

            window._start_asset_segmentation(asset)
            window.on_sam_review_cancelled()

            self.assertEqual(asset.__dict__, before)
            self.assertEqual(window.progress_label.text(), "SAM 分割已取消")
            self.assertEqual((window.progress_bar.minimum(), window.progress_bar.maximum()), (0, 100))

    def test_sam_candidates_ready_replaces_running_status_with_confirmation_prompt(self):
        mask = np.zeros((1, 80, 100), dtype=np.uint8)
        mask[0, 12:35, 12:45] = 255
        result = SegmentationResult(mask, np.asarray([0.8]), 0, "cpu")
        asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))
        dialog = SimpleNamespace(points=[], set_result=lambda *_args: None)
        window = MainWindow.__new__(MainWindow)
        window.sam_review_dialog = dialog
        window.pending_sam_asset = asset
        window.pending_sam_slide = None
        window.pending_sam_points = []
        window.progress_label = QLabel("运行中...")
        window.progress_bar = QProgressBar()
        window.progress_bar.setRange(0, 0)
        window.append_log = lambda _message: None

        window.on_asset_segmented(
            {
                "asset": asset,
                "slide": SimpleNamespace(),
                "result": result,
                "points": [],
                "debug_dir": Path("debug"),
            }
        )

        self.assertEqual(window.progress_label.text(), "SAM 候选已生成，请检查并确认分割")
        self.assertEqual((window.progress_bar.minimum(), window.progress_bar.maximum()), (0, 100))
        self.assertEqual(window.progress_bar.value(), 100)

    def test_sam_worker_passes_writable_image_array_to_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "slide.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            slide = PPTSlide(1, "slide.png", image_path, 100, 80)
            asset = VisualAsset("device", (10, 10, 40, 30))
            captured = {}

            class CapturingEngine:
                def segment_with_box(self, image, box, **_kwargs):
                    captured["writeable"] = bool(image.flags.writeable)
                    masks = np.zeros((1, 80, 100), dtype=np.uint8)
                    masks[0, 12:35, 12:45] = 255
                    return SegmentationResult(masks, np.asarray([0.8]), 0, "cpu")

            window = MainWindow.__new__(MainWindow)
            window.sam_engine = CapturingEngine()
            project = SimpleNamespace(work_dir=root)
            with patch("app.gui.save_segmentation_debug", return_value=root / "debug"):
                window._run_asset_segmentation(project, slide, asset, [])

            self.assertTrue(captured["writeable"])

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
        asset = VisualAsset(asset_id="device", bbox=(0, 0, 20, 20), source="manual")
        before = asset.__dict__.copy()
        window = MainWindow.__new__(MainWindow)
        window.append_log = lambda _message: None
        window.render_current_slide = lambda: None
        window.progress_label = QLabel("运行中...")
        window.progress_bar = QProgressBar()
        window.progress_bar.setRange(0, 0)

        with patch("app.gui.QMessageBox.warning") as warning:
            window.on_asset_segmentation_failed(
                asset,
                "SAM runtime import failed",
                ModuleNotFoundError("No module named 'sam2'"),
            )

        message = warning.call_args.args[2]
        self.assertIn("重新安装", message)
        self.assertIn("重启", message)
        self.assertEqual(asset.__dict__, before)

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
