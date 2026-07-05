import sys
import types
import unittest
import warnings
from io import StringIO

from app.export_worker import progress_printer, silence_dependency_info_logs


class ExportWorkerLoggingTest(unittest.TestCase):
    def test_silences_loguru_info_messages(self):
        calls = []
        logger = types.SimpleNamespace(
            remove=lambda: calls.append(("remove",)),
            add=lambda sink, level: calls.append(("add", sink, level)),
        )
        loguru = types.ModuleType("loguru")
        loguru.logger = logger

        original = sys.modules.get("loguru")
        sys.modules["loguru"] = loguru
        try:
            silence_dependency_info_logs()
        finally:
            if original is None:
                sys.modules.pop("loguru", None)
            else:
                sys.modules["loguru"] = original

        self.assertEqual(calls, [("remove",), ("add", sys.stderr, "WARNING")])

    def test_silences_future_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            silence_dependency_info_logs()
            warnings.warn("deprecated torch autocast", FutureWarning)

        self.assertEqual(caught, [])

    def test_progress_printer_writes_to_explicit_stream(self):
        visible = StringIO()
        hidden = StringIO()
        original_stdout = sys.stdout
        sys.stdout = hidden
        try:
            progress_printer(visible)("第1页已提升清晰度：slide_01.jpg")
            print("Tile 1/24")
        finally:
            sys.stdout = original_stdout

        self.assertEqual(visible.getvalue(), "第1页已提升清晰度：slide_01.jpg\n")
        self.assertEqual(hidden.getvalue(), "Tile 1/24\n")


if __name__ == "__main__":
    unittest.main()
