import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from app.core import OCRBox, PPTProject, PPTSlide, rebuild_ppt, upscale_cleaned_images


class UpscaleCleanedImagesTest(unittest.TestCase):
    def test_upscales_cleaned_images_with_realesrgan_and_preserves_rgb_colors(self):
        with TemporaryDirectory() as temp_dir:
            cleaned_dir = Path(temp_dir)
            image_path = cleaned_dir / "slide_01.png"
            Image.new("RGB", (2, 1), (10, 20, 30)).save(image_path)

            calls = []

            class FakeUpscaler:
                name = "RealESRGAN"

                def __init__(self, model_name, device, no_half=False):
                    calls.append(("init", model_name, device, no_half))

                def gen_image(self, rgb_np_img, request):
                    calls.append(("gen_image", tuple(rgb_np_img[0, 0]), request.scale))
                    return np.array([[[90, 80, 70], [60, 50, 40]]], dtype=np.uint8)

            class FakeRequest:
                def __init__(self, name, image, scale):
                    self.name = name
                    self.image = image
                    self.scale = scale

            plugins_module = types.ModuleType("iopaint.plugins")
            plugins_module.RealESRGANUpscaler = FakeUpscaler
            schema_module = types.ModuleType("iopaint.schema")
            schema_module.Device = types.SimpleNamespace(cpu="cpu")
            schema_module.RunPluginRequest = FakeRequest
            model_utils_module = types.ModuleType("iopaint.model.utils")
            model_utils_module.torch_gc = lambda: calls.append(("torch_gc",))

            with unittest.mock.patch.dict(
                sys.modules,
                {
                    "iopaint.plugins": plugins_module,
                    "iopaint.schema": schema_module,
                    "iopaint.model.utils": model_utils_module,
                },
            ):
                processed = upscale_cleaned_images(cleaned_dir, scale=2, progress=None)

            self.assertEqual(processed, 1)
            self.assertIn(("init", "realesr-general-x4v3", "cpu", True), calls)
            self.assertIn(("gen_image", (10, 20, 30), 2), calls)
            self.assertIn(("torch_gc",), calls)
            with Image.open(image_path) as output:
                self.assertEqual(output.size, (2, 1))
                self.assertEqual(output.getpixel((0, 0)), (70, 80, 90))

    def test_rebuild_keeps_text_coordinates_based_on_original_slide_image_size(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            cleaned_dir = root / "cleaned"
            masks_dir = root / "masks"
            images_dir.mkdir()
            cleaned_dir.mkdir()
            masks_dir.mkdir()

            original_path = images_dir / "slide_01.png"
            cleaned_path = cleaned_dir / "slide_01.png"
            Image.new("RGB", (100, 100), (255, 255, 255)).save(original_path)
            Image.new("RGB", (200, 200), (255, 255, 255)).save(cleaned_path)

            project = PPTProject(
                source_pptx=root / "source.pptx",
                work_dir=root,
                images_dir=images_dir,
                masks_dir=masks_dir,
                cleaned_dir=cleaned_dir,
                slides=[
                    PPTSlide(
                        index=1,
                        image_name="slide_01.png",
                        image_path=original_path,
                        image_width=100,
                        image_height=100,
                        boxes=[
                            OCRBox(
                                text="Hello",
                                score=1.0,
                                bbox=(50, 20, 10, 10),
                                erase_rect=(50, 20, 60, 30),
                            )
                        ],
                    )
                ],
                slide_width=914400,
                slide_height=914400,
            )

            output_path = root / "out.pptx"
            rebuild_ppt(project, output_path)

            from pptx import Presentation

            deck = Presentation(str(output_path))
            text_shape = next(shape for shape in deck.slides[0].shapes if getattr(shape, "text", "") == "Hello")
            self.assertEqual(text_shape.left, 457200)


if __name__ == "__main__":
    unittest.main()
