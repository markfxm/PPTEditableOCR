import sys
import types
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from app.core import (
    OCRBox,
    PPTProject,
    PPTSlide,
    cleaned_image_path,
    export_editable_ppt,
    prefer_system_cuda_torch,
    preferred_iopaint_device,
    preferred_paddleocr_device,
    rebuild_ppt,
    run_iopaint,
    upscale_cleaned_images,
)


class UpscaleCleanedImagesTest(unittest.TestCase):
    def test_prefer_system_cuda_torch_keeps_cuda_module_loaded(self):
        loaded_modules = {}
        cuda_torch = types.SimpleNamespace(
            __name__="torch",
            cuda=types.SimpleNamespace(is_available=lambda: True),
        )

        def fake_import(name):
            loaded_modules[name] = cuda_torch
            sys.modules[name] = cuda_torch
            return cuda_torch

        with unittest.mock.patch.dict(sys.modules, {}, clear=True):
            selected = prefer_system_cuda_torch(
                original_sys_path=["system-site-packages"],
                bundled_paths={Path("bundled")},
                import_module=fake_import,
            )

            self.assertTrue(selected)
            self.assertIs(sys.modules["torch"], cuda_torch)

    def test_prefer_system_cuda_torch_removes_cpu_only_module(self):
        cpu_torch = types.SimpleNamespace(
            __name__="torch",
            cuda=types.SimpleNamespace(is_available=lambda: False),
        )

        def fake_import(name):
            sys.modules[name] = cpu_torch
            sys.modules["torch.cuda"] = types.SimpleNamespace()
            return cpu_torch

        with unittest.mock.patch.dict(sys.modules, {}, clear=True):
            selected = prefer_system_cuda_torch(
                original_sys_path=["system-site-packages"],
                bundled_paths={Path("bundled")},
                import_module=fake_import,
            )

            self.assertFalse(selected)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("torch.cuda", sys.modules)

    def test_prefers_cuda_for_iopaint_when_torch_cuda_is_available(self):
        device = types.SimpleNamespace(cpu="cpu", cuda="cuda")
        torch_module = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))

        self.assertEqual(preferred_iopaint_device(device, torch_module=torch_module), "cuda")

    def test_uses_cpu_for_iopaint_when_torch_cuda_is_unavailable(self):
        device = types.SimpleNamespace(cpu="cpu", cuda="cuda")
        torch_module = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        self.assertEqual(preferred_iopaint_device(device, torch_module=torch_module), "cpu")

    def test_prefers_gpu_for_paddleocr_when_paddle_is_cuda_build(self):
        paddle_module = types.SimpleNamespace(
            device=types.SimpleNamespace(is_compiled_with_cuda=lambda: True)
        )

        self.assertEqual(preferred_paddleocr_device(paddle_module=paddle_module), "gpu")

    def test_uses_cpu_for_paddleocr_when_paddle_is_cpu_build(self):
        paddle_module = types.SimpleNamespace(
            device=types.SimpleNamespace(is_compiled_with_cuda=lambda: False)
        )

        self.assertEqual(preferred_paddleocr_device(paddle_module=paddle_module), "cpu")

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
            schema_module.Device = types.SimpleNamespace(cpu="cpu", cuda="cuda")
            schema_module.RunPluginRequest = FakeRequest
            model_utils_module = types.ModuleType("iopaint.model.utils")
            model_utils_module.torch_gc = lambda: calls.append(("torch_gc",))
            torch_module = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            messages = []

            with unittest.mock.patch.dict(
                sys.modules,
                {
                    "iopaint.plugins": plugins_module,
                    "iopaint.schema": schema_module,
                    "iopaint.model.utils": model_utils_module,
                    "torch": torch_module,
                },
            ):
                processed = upscale_cleaned_images(cleaned_dir, scale=2, progress=messages.append)

            self.assertEqual(processed, 1)
            self.assertIn(("init", "realesr-general-x4v3", "cuda", False), calls)
            self.assertIn(("gen_image", (10, 20, 30), 2), calls)
            self.assertIn(("torch_gc",), calls)
            self.assertIn("第 1 页已提升清晰度：slide_01.jpg", messages)
            with Image.open(image_path) as output:
                self.assertEqual(output.size, (2, 1))
                self.assertEqual(output.getpixel((0, 0)), (70, 80, 90))
            jpeg_path = cleaned_dir / "slide_01.jpg"
            self.assertTrue(jpeg_path.exists())
            with Image.open(jpeg_path) as output:
                self.assertEqual(output.format, "JPEG")

    def test_cleaned_image_path_prefers_compressed_jpeg_for_ppt_embedding(self):
        with TemporaryDirectory() as temp_dir:
            cleaned_dir = Path(temp_dir)
            png_path = cleaned_dir / "slide_01.png"
            jpeg_path = cleaned_dir / "slide_01.jpg"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(png_path)
            Image.new("RGB", (10, 10), (255, 255, 255)).save(jpeg_path, format="JPEG")

            self.assertEqual(cleaned_image_path(cleaned_dir, "slide_01.png"), jpeg_path)

    def test_iopaint_logs_each_generated_cleaned_page(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            masks_dir = root / "masks"
            cleaned_dir = root / "cleaned"
            images_dir.mkdir()
            masks_dir.mkdir()
            Image.new("RGB", (2, 1), (10, 20, 30)).save(images_dir / "slide_01.png")
            Image.new("L", (2, 1), 255).save(masks_dir / "slide_01.png")

            calls = []

            class FakeModelManager:
                def __init__(self, name, device):
                    calls.append(("init", name, device))

                def __call__(self, image, mask, request):
                    calls.append(("call", tuple(image[0, 0]), int(mask[0, 0])))
                    return np.array([[[90, 80, 70], [60, 50, 40]]], dtype=np.uint8)

            class FakeRequest:
                pass

            download_module = types.ModuleType("iopaint.download")
            download_module.scan_models = lambda: [types.SimpleNamespace(name="lama")]
            download_module.cli_download_model = lambda name: calls.append(("download", name))
            helper_module = types.ModuleType("iopaint.helper")

            def fake_pil_to_bytes(image, output_format, quality, infos):
                import io

                buffer = io.BytesIO()
                image.save(buffer, format=output_format.upper())
                return buffer.getvalue()

            helper_module.pil_to_bytes = fake_pil_to_bytes
            model_utils_module = types.ModuleType("iopaint.model.utils")
            model_utils_module.torch_gc = lambda: calls.append(("torch_gc",))
            model_manager_module = types.ModuleType("iopaint.model_manager")
            model_manager_module.ModelManager = FakeModelManager
            schema_module = types.ModuleType("iopaint.schema")
            schema_module.Device = types.SimpleNamespace(cpu="cpu", cuda="cuda")
            schema_module.InpaintRequest = FakeRequest
            torch_module = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            messages = []

            with unittest.mock.patch.dict(
                sys.modules,
                {
                    "iopaint.download": download_module,
                    "iopaint.helper": helper_module,
                    "iopaint.model.utils": model_utils_module,
                    "iopaint.model_manager": model_manager_module,
                    "iopaint.schema": schema_module,
                    "torch": torch_module,
                },
            ):
                run_iopaint(images_dir, masks_dir, cleaned_dir, progress=messages.append)

            self.assertIn(("init", "lama", "cuda"), calls)
            self.assertTrue((cleaned_dir / "slide_01.png").exists())
            self.assertTrue((cleaned_dir / "slide_01.jpg").exists())
            self.assertIn("第 1 页 IOPaint 擦除后已生成：slide_01.jpg", messages)

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

    def test_rebuild_embeds_compressed_jpeg_when_available(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            cleaned_dir = root / "cleaned"
            masks_dir = root / "masks"
            images_dir.mkdir()
            cleaned_dir.mkdir()
            masks_dir.mkdir()

            original_path = images_dir / "slide_01.png"
            cleaned_png_path = cleaned_dir / "slide_01.png"
            cleaned_jpeg_path = cleaned_dir / "slide_01.jpg"
            Image.new("RGB", (100, 100), (255, 255, 255)).save(original_path)
            Image.new("RGB", (100, 100), (255, 255, 255)).save(cleaned_png_path)
            Image.new("RGB", (100, 100), (255, 255, 255)).save(cleaned_jpeg_path, format="JPEG")

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
                        boxes=[],
                    )
                ],
                slide_width=914400,
                slide_height=914400,
            )

            output_path = root / "out.pptx"
            rebuild_ppt(project, output_path)

            with zipfile.ZipFile(output_path) as deck:
                media_names = [name for name in deck.namelist() if name.startswith("ppt/media/")]
            self.assertTrue(any(name.lower().endswith((".jpg", ".jpeg")) for name in media_names))

    def test_export_can_skip_realesrgan_enhancement(self):
        project = PPTProject(
            source_pptx=Path("source.pptx"),
            work_dir=Path("work"),
            images_dir=Path("images"),
            masks_dir=Path("masks"),
            cleaned_dir=Path("cleaned"),
            slides=[],
            slide_width=914400,
            slide_height=914400,
        )
        messages = []
        with (
            unittest.mock.patch("app.core.build_masks"),
            unittest.mock.patch("app.core.run_iopaint"),
            unittest.mock.patch("app.core.upscale_cleaned_images") as upscale,
            unittest.mock.patch("app.core.rebuild_ppt"),
        ):
            export_editable_ppt(project, Path("out.pptx"), progress=messages.append, enhance_images=False)

        upscale.assert_not_called()
        self.assertIn("已跳过 RealESRGAN 底图清晰化", messages)


if __name__ == "__main__":
    unittest.main()
