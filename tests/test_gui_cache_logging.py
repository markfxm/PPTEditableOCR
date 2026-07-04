import unittest
from pathlib import Path
from unittest.mock import patch

from app.gui import MainWindow


class GuiCacheLoggingTest(unittest.TestCase):
    def _window(self):
        window = MainWindow.__new__(MainWindow)
        window.project = object()
        window.logs = []
        window.append_log = window.logs.append
        return window

    def test_silent_cache_save_does_not_log_success(self):
        window = self._window()

        with patch("app.gui.save_project_cache", return_value=Path("cache.json")):
            window.save_current_cache(show_message=False, log_success=False)

        self.assertEqual(window.logs, [])

    def test_final_cache_save_logs_recognition_result_saved(self):
        window = self._window()

        with patch("app.gui.save_project_cache", return_value=Path("cache.json")):
            window.save_current_cache(show_message=False, success_prefix="识别结果已保存")

        self.assertEqual(window.logs, ["识别结果已保存：cache.json"])


if __name__ == "__main__":
    unittest.main()
