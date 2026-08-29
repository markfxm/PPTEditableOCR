import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.sam_segmentation import (
    MODEL_SHA256,
    SamSegmentationEngine,
    _create_predictor,
    download_model,
    model_path,
    verify_model,
)


class FakePredictor:
    def __init__(self, device="cpu", fail_predict=False):
        self.device = device
        self.fail_predict = fail_predict
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1
        self.image_shape = image.shape[:2]

    def predict(self, box, multimask_output=True):
        if self.fail_predict:
            raise RuntimeError("CUDA out of memory")
        height, width = self.image_shape
        masks = np.zeros((3, height, width), dtype=bool)
        masks[0, 2:5, 3:7] = True
        masks[1, 1:6, 2:8] = True
        masks[2, :, :] = True
        return masks, np.asarray([0.2, 0.9, 0.1]), np.zeros((3, 256, 256), dtype=np.float32)


class SamModelManagementTests(unittest.TestCase):
    def test_setup_installs_matching_cuda_torch_when_nvidia_is_available(self):
        script = (Path(__file__).resolve().parent.parent / "setup_dev.ps1").read_text(encoding="utf-8")
        self.assertIn("Get-Command nvidia-smi", script)
        self.assertIn("https://download.pytorch.org/whl/cu130", script)
        self.assertIn("torch==2.11.0+cu130", script)
        self.assertIn("torchvision==0.26.0+cu130", script)

    def test_model_path_uses_local_appdata(self):
        path = model_path(local_appdata=Path("C:/LocalData"))
        self.assertEqual(
            path,
            Path("C:/LocalData/PPTEditableOCR/models/sam2/sam2.1_hiera_tiny.pt"),
        )

    def test_verify_model_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.pt"
            path.write_bytes(b"wrong")
            self.assertFalse(verify_model(path, expected_sha256=MODEL_SHA256))

    def test_download_verifies_before_replacing_destination(self):
        payload = b"verified model"
        expected = hashlib.sha256(payload).hexdigest()

        def downloader(_url, target, _progress, _cancelled):
            target.write_bytes(payload)

        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.pt"
            result = download_model(
                destination=destination,
                expected_sha256=expected,
                downloader=downloader,
            )
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".pt.download").exists())

    def test_download_removes_partial_file_when_hash_is_wrong(self):
        def downloader(_url, target, _progress, _cancelled):
            target.write_bytes(b"corrupt")

        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.pt"
            with self.assertRaisesRegex(RuntimeError, "校验失败"):
                download_model(
                    destination=destination,
                    expected_sha256="0" * 64,
                    downloader=downloader,
                )
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".pt.download").exists())

    def test_download_keeps_existing_partial_for_resumable_downloader(self):
        payload = b"complete model"
        expected = hashlib.sha256(payload).hexdigest()

        def downloader(_url, target, _progress, _cancelled):
            self.assertEqual(target.read_bytes(), b"complete ")
            with target.open("ab") as stream:
                stream.write(b"model")

        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.pt"
            destination.with_suffix(".pt.download").write_bytes(b"complete ")
            result = download_model(
                destination=destination,
                expected_sha256=expected,
                downloader=downloader,
            )
            self.assertEqual(result.read_bytes(), payload)


class SamSegmentationEngineTests(unittest.TestCase):
    def test_create_predictor_supports_read_only_device_property(self):
        class FakeModel:
            def eval(self):
                self.eval_called = True

        class ReadOnlyDevicePredictor:
            def __init__(self, model):
                self.model = model

            @property
            def device(self):
                return "cpu"

        model = FakeModel()
        torch_module = types.ModuleType("torch")
        thread_limits = []
        torch_module.get_num_threads = lambda: 16
        torch_module.set_num_threads = thread_limits.append
        build_module = types.ModuleType("sam2.build_sam")
        build_module.build_sam2 = lambda *_args, **_kwargs: model
        predictor_module = types.ModuleType("sam2.sam2_image_predictor")
        predictor_module.SAM2ImagePredictor = ReadOnlyDevicePredictor

        with patch.dict(
            sys.modules,
            {
                "torch": torch_module,
                "sam2.build_sam": build_module,
                "sam2.sam2_image_predictor": predictor_module,
            },
        ):
            try:
                predictor = _create_predictor(Path("model.pt"), "cpu")
            except AttributeError as exc:
                self.fail(f"predictor device property must not be assigned: {exc}")

        self.assertIs(predictor.model, model)
        self.assertTrue(model.eval_called)
        self.assertEqual(thread_limits, [4])

    def test_reuses_page_embedding_and_selects_highest_quality_mask(self):
        predictor = FakePredictor()
        engine = SamSegmentationEngine(
            Path("model.pt"),
            device="cpu",
            predictor_factory=lambda _path, _device: predictor,
        )
        image = np.zeros((10, 12, 3), dtype=np.uint8)

        first = engine.segment_with_box(image, (2, 1, 8, 6), image_key="slide-1")
        second = engine.segment_with_box(image, (3, 2, 9, 7), image_key="slide-1")

        self.assertEqual(predictor.set_image_calls, 1)
        self.assertEqual(first.confidence, 0.9)
        self.assertEqual(first.device, "cpu")
        self.assertEqual(first.mask.shape, (10, 12))
        self.assertEqual(int(np.count_nonzero(first.mask)), 30)
        self.assertEqual(second.model_id, "sam2.1_hiera_tiny")

    def test_cuda_failure_retries_with_cpu_predictor(self):
        created = []

        def factory(_path, device):
            created.append(device)
            return FakePredictor(device=device, fail_predict=device == "cuda")

        engine = SamSegmentationEngine(
            Path("model.pt"),
            device="cuda",
            predictor_factory=factory,
        )
        result = engine.segment_with_box(
            np.zeros((10, 12, 3), dtype=np.uint8),
            (2, 1, 8, 6),
            image_key="slide-1",
        )

        self.assertEqual(created, ["cuda", "cpu"])
        self.assertEqual(result.device, "cpu")


if __name__ == "__main__":
    unittest.main()
