import subprocess
import sys
import unittest


class LazyImportTests(unittest.TestCase):
    def test_run_gui_import_does_not_load_ocr_or_torch(self):
        script = (
            "import sys; "
            "import run_gui; "
            "loaded = [name for name in ('paddleocr', 'paddlex', 'torch') if name in sys.modules]; "
            "print(','.join(loaded)); "
            "raise SystemExit(1 if loaded else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gui_import_does_not_load_ocr_or_torch(self):
        script = (
            "import sys; "
            "from app.gui import main; "
            "loaded = [name for name in ('paddleocr', 'paddlex', 'torch') if name in sys.modules]; "
            "print(','.join(loaded)); "
            "raise SystemExit(1 if loaded else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
