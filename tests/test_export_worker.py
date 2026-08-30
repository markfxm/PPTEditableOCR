import json
import sys
import tempfile
import types
import unittest
import warnings
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from app.export_worker import main, progress_printer, silence_dependency_info_logs
from app.quality_pipeline import QualityMode


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

    def test_worker_accepts_optional_quality_payload_without_persisted_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "project": {"ignored": True},
                        "output_pptx": str(Path(directory) / "output.pptx"),
                        "quality_mode": "local_fast",
                        "online_pages": [2],
                        "accepted_local_pages": [1],
                    }
                ),
                encoding="utf-8",
            )
            exported = []
            with (
                patch("app.export_worker.ppt_project_from_data", return_value=object()),
                patch("app.export_worker.export_editable_ppt", side_effect=lambda *args, **kwargs: exported.append((args, kwargs))),
            ):
                self.assertEqual(main([str(input_path)]), 0)

            _args, kwargs = exported[0]
            self.assertEqual(kwargs["quality_mode"], QualityMode.LOCAL_FAST)
            self.assertEqual(kwargs["online_pages"], {2})
            self.assertEqual(kwargs["accepted_local_pages"], {1})
            self.assertIsNone(kwargs["openai_api_key"])


if __name__ == "__main__":
    unittest.main()
