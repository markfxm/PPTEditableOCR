import unittest
from pathlib import Path


class GuiLayoutTextTest(unittest.TestCase):
    def test_ocr_token_status_has_no_left_label(self):
        source = Path("app/gui.py").read_text(encoding="utf-8")

        self.assertNotIn('addRow("远端令牌", self.ocr_token_status)', source)

    def test_enhance_image_option_is_optional_and_off_by_default(self):
        source = Path("app/gui.py").read_text(encoding="utf-8")

        self.assertIn('QCheckBox("导出时清晰化底图（RealESRGAN）（可选）")', source)
        self.assertIn("self.enhance_images_cb.setChecked(False)", source)
        self.assertNotIn("self.enhance_images_cb.setChecked(True)", source)


    def test_pdf_gui_import_uses_direct_project_loading(self):
        source = Path("app/gui.py").read_text(encoding="utf-8")

        self.assertIn("prepare_pdf_project", source)
        self.assertNotIn("convert_pdf_to_pptx", source)
        self.assertNotIn("PDF 载入完成，未生成中间 PPT。请选择 OCR 方式后点击", source)


if __name__ == "__main__":
    unittest.main()
