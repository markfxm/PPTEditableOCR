import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as _Image
from app import core as _desktop_core  # Preload bundled binary dependencies for the mirrored web module.


def load_web_core():
    paddleocr = types.ModuleType("paddleocr")
    paddleocr.PaddleOCR = object
    sys.modules["paddleocr"] = paddleocr
    path = Path(__file__).resolve().parents[1] / "web_deploy" / "backend" / "ppttoedit_core" / "app" / "core.py"
    spec = importlib.util.spec_from_file_location("ppttoedit_web_core_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WebCoreAssetRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_web_core()

    def test_overlapping_text_does_not_punch_web_asset_alpha(self):
        image = np.full((60, 80, 3), (0, 160, 220), dtype=np.uint8)
        asset = self.core.VisualAsset("asset", (10, 10, 50, 40))
        box = self.core.OCRBox("A", 1.0, (20, 20, 10, 10), (20, 20, 30, 30))

        alpha = self.core.visual_asset_alpha_mask(image, asset, [box])

        self.assertEqual(alpha[25, 25], 255)

    def test_web_page_finish_uses_flat_surroundings_instead_of_generated_artifact(self):
        original = np.full((60, 80, 3), (30, 50, 70), dtype=np.uint8)
        generated = np.full_like(original, (220, 20, 20))
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[10:50, 15:65] = 255

        finished = self.core.finish_page_inpaint(original, generated, mask)

        self.assertLess(int(finished[30, 40, 0]), 100)
        self.assertEqual(tuple(finished[0, 0]), tuple(original[0, 0]))


if __name__ == "__main__":
    unittest.main()
