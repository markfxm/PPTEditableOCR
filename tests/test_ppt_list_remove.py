import unittest
from pathlib import Path
from types import SimpleNamespace

from app.gui import MainWindow
from app.gui import prepare_pdf_project


class FakeList:
    def __init__(self):
        self.items = []
        self.current_row = -1

    def takeItem(self, row):
        return self.items.pop(row)

    def count(self):
        return len(self.items)

    def setCurrentRow(self, row):
        self.current_row = row


class FakeClearable:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class PptListRemoveTest(unittest.TestCase):
    def _window(self):
        window = MainWindow.__new__(MainWindow)
        window.ppt_paths = [Path("a.pptx").resolve(), Path("b.pptx").resolve()]
        window.ppt_file_list = FakeList()
        window.ppt_file_list.items = [object(), object()]
        window.selecting_ppt_list_item = False
        window.worker_thread = None
        window.project = SimpleNamespace(source_pptx=window.ppt_paths[0])
        window.current_slide = object()
        window.selected_item = object()
        window.current_items = [object()]
        window.undo_stacks = {1: [object()]}
        window.pending_select_index = 0
        window.slide_list = FakeClearable()
        window.scene = FakeClearable()
        window.save_cache_action = SimpleNamespace(setEnabled=lambda _enabled: None)
        window.continue_export_btn = SimpleNamespace(setEnabled=lambda _enabled: None)
        window.start_ocr_btn = SimpleNamespace(setEnabled=lambda _enabled: None)
        window.progress_label = SimpleNamespace(setText=lambda _text: None)
        window.progress_bar = SimpleNamespace(setRange=lambda *_args: None, setValue=lambda _value: None)
        window.update_undo_action = lambda: None
        return window

    def test_remove_current_ppt_clears_project_state(self):
        window = self._window()

        window.remove_ppt_from_recent_list(0)

        self.assertEqual(window.ppt_paths, [Path("b.pptx").resolve()])
        self.assertIsNone(window.project)
        self.assertIsNone(window.current_slide)
        self.assertEqual(window.current_items, [])
        self.assertEqual(window.undo_stacks, {})
        self.assertTrue(window.slide_list.cleared)
        self.assertTrue(window.scene.cleared)

    def test_remove_other_ppt_keeps_current_project(self):
        window = self._window()

        window.remove_ppt_from_recent_list(1)

        self.assertEqual(window.ppt_paths, [Path("a.pptx").resolve()])
        self.assertIsNotNone(window.project)

    def test_ppt_list_item_uses_only_custom_widget_text_with_remove_button_first(self):
        source = Path("app/gui.py").read_text(encoding="utf-8")

        self.assertIn("item = QListWidgetItem()", source)
        self.assertNotIn("item = QListWidgetItem(source.name)", source)
        self.assertLess(source.index('layout.addWidget(remove_btn)'), source.index('layout.addWidget(label, 1)'))

    def test_load_pdf_source_uses_direct_pdf_project_loader(self):
        window = MainWindow.__new__(MainWindow)
        window.worker_thread = None
        window.add_ppt_to_recent_list = lambda *_args, **_kwargs: None
        window.clear_current_project = lambda: None
        window.append_log = lambda _message: None
        calls = []

        def fake_run_worker(fn, finished_cb, *args, **kwargs):
            calls.append((fn, finished_cb, args, kwargs))

        window.run_worker = fake_run_worker
        window.on_project_loaded = object()

        window.load_source_path(Path("source.pdf"), add_to_list=True, select_in_list=True)

        self.assertEqual(calls[0][0], prepare_pdf_project)
        self.assertEqual(calls[0][2][0], Path("source.pdf").resolve())
        self.assertFalse(calls[0][3]["auto_ocr"])


if __name__ == "__main__":
    unittest.main()
