import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from app.core import PPTSlide, prepare_project


class PrepareProjectOcrFlowTest(unittest.TestCase):
    def _fake_extract(self):
        src = Mock()
        src.slide_width = 100
        src.slide_height = 100
        slides = [
            PPTSlide(
                index=1,
                image_name="slide_01.png",
                image_path=Path("slide_01.png"),
                image_width=100,
                image_height=100,
            )
        ]
        return src, slides

    def test_auto_ocr_false_loads_project_without_running_ocr(self):
        with TemporaryDirectory() as temp_dir:
            with (
                patch("app.core.extract_slide_images", return_value=self._fake_extract()),
                patch("app.core.load_project_cache", return_value=False),
                patch("app.core.run_ocr") as run_ocr,
            ):
                project = prepare_project(Path("source.pptx"), Path(temp_dir) / "work", auto_ocr=False)

        self.assertEqual(len(project.slides), 1)
        run_ocr.assert_not_called()

    def test_auto_ocr_true_still_runs_ocr_when_cache_is_missing(self):
        with TemporaryDirectory() as temp_dir:
            with (
                patch("app.core.extract_slide_images", return_value=self._fake_extract()),
                patch("app.core.load_project_cache", return_value=False),
                patch("app.core.run_ocr") as run_ocr,
            ):
                project = prepare_project(Path("source.pptx"), Path(temp_dir) / "work")

        run_ocr.assert_called_once_with(project.slides, None, ocr_backend="local", ocr_token=None)


if __name__ == "__main__":
    unittest.main()
