import unittest

from app.gui import FLOW_TEXT


class GuiFlowTextTest(unittest.TestCase):
    def test_flow_text_lists_iopaint_after_reviewing_ocr_boxes(self):
        self.assertIn("3. 检查识别框", FLOW_TEXT)
        self.assertIn("4. IOPaint 擦除", FLOW_TEXT)
        self.assertIn("5. 选择是否清晰化", FLOW_TEXT)

    def test_flow_text_lists_quality_review_before_final_export(self):
        self.assertIn("质量检查", FLOW_TEXT)
        self.assertIn("可选在线修复", FLOW_TEXT)


if __name__ == "__main__":
    unittest.main()
