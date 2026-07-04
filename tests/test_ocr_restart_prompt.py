import unittest
from pathlib import Path
from types import SimpleNamespace

from app.core import OCRBox, PPTProject, PPTSlide
from app.gui import MainWindow


class FakeBar:
    def __init__(self):
        self.changed = False

    def setRange(self, *_args):
        self.changed = True

    def setValue(self, _value):
        self.changed = True


class OcrRestartPromptTest(unittest.TestCase):
    def _project(self):
        slide = PPTSlide(
            index=1,
            image_name="slide_01.png",
            image_path=Path("slide_01.png"),
            image_width=100,
            image_height=100,
            ocr_status="ok",
        )
        slide.boxes = [
            OCRBox(
                text="done",
                score=1.0,
                bbox=(1, 2, 3, 4),
                erase_rect=(1, 2, 4, 6),
            )
        ]
        return PPTProject(
            source_pptx=Path("source.pptx").resolve(),
            work_dir=Path("work"),
            images_dir=Path("images"),
            masks_dir=Path("masks"),
            cleaned_dir=Path("cleaned"),
            slides=[slide],
            slide_width=100,
            slide_height=100,
        )

    def _window(self):
        window = MainWindow.__new__(MainWindow)
        window.project = self._project()
        window.resolved_ocr_config = lambda: ("local", None, None)
        window.confirm_restart_ocr = lambda: False
        window.progress_bar = FakeBar()
        window.append_log = lambda _message: None
        window.refresh_slide_list = lambda: None
        window.slide_list = SimpleNamespace(setCurrentRow=lambda _row: None)
        window.run_next_ocr_page_called = False
        window.run_next_ocr_page = lambda: setattr(window, "run_next_ocr_page_called", True)
        return window

    def test_cancel_restart_keeps_existing_ocr_results(self):
        window = self._window()

        window.start_ocr()

        self.assertFalse(window.run_next_ocr_page_called)
        self.assertEqual(window.project.slides[0].ocr_status, "ok")
        self.assertEqual(window.project.slides[0].boxes[0].text, "done")
        self.assertFalse(window.progress_bar.changed)

    def test_confirm_restart_clears_existing_results_and_starts_ocr(self):
        window = self._window()
        window.confirm_restart_ocr = lambda: True

        window.start_ocr()

        self.assertTrue(window.run_next_ocr_page_called)
        self.assertEqual(window.project.slides[0].ocr_status, "pending")
        self.assertEqual(window.project.slides[0].boxes, [])

    def test_existing_ocr_result_detects_completed_status_or_boxes(self):
        window = self._window()

        self.assertTrue(window.has_existing_ocr_result())


if __name__ == "__main__":
    unittest.main()
