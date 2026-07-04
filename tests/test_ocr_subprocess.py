import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import OCRPageProcessError, PPTSlide, run_ocr_page_subprocess


class OcrSubprocessTest(unittest.TestCase):
    def _slide(self):
        return PPTSlide(
            index=3,
            image_name="slide_03.png",
            image_path=Path("slide_03.png"),
            image_width=200,
            image_height=100,
        )

    def test_nonzero_child_exit_reports_page_failure(self):
        def fake_run(cmd, **kwargs):
            output_path = Path(cmd[-1])
            output_path.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(
                cmd,
                3221225477,
                stdout="Windows fatal exception: access violation\n",
                stderr="",
            )

        with self.assertRaises(OCRPageProcessError) as raised:
            run_ocr_page_subprocess(self._slide(), runner=fake_run)

        self.assertEqual(raised.exception.slide_index, 3)
        self.assertEqual(raised.exception.returncode, 3221225477)
        self.assertIn("access violation", raised.exception.output)

    def test_successful_child_output_applies_boxes_to_slide(self):
        def fake_run(cmd, **kwargs):
            output_path = Path(cmd[-1])
            output_path.write_text(
                json.dumps(
                    {
                        "boxes": [
                            {
                                "text": "hello",
                                "score": 0.98,
                                "bbox": [10, 20, 30, 12],
                                "erase_rect": [8, 18, 42, 35],
                                "enabled": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with patch.object(tempfile, "TemporaryDirectory", wraps=tempfile.TemporaryDirectory):
            boxes = run_ocr_page_subprocess(self._slide(), runner=fake_run)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].text, "hello")
        self.assertEqual(boxes[0].bbox, (10, 20, 30, 12))


if __name__ == "__main__":
    unittest.main()
