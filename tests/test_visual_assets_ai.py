import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core import (
    OCRBox,
    PPTProject,
    PPTSlide,
    VisualAsset,
    build_masks,
    ppt_project_from_data,
    ppt_project_to_data,
    repair_visual_assets,
    store_visual_asset_mask,
    visual_asset_alpha_mask,
)
from app.sam_segmentation import SegmentationResult


class VisualAssetAITests(unittest.TestCase):
    def make_slide(self, root: Path) -> PPTSlide:
        image_path = root / "slide.png"
        Image.new("RGB", (100, 80), "white").save(image_path)
        return PPTSlide(
            index=1,
            image_name="slide.png",
            image_path=image_path,
            image_width=100,
            image_height=80,
        )

    def make_project(self, root: Path, slide: PPTSlide) -> PPTProject:
        return PPTProject(
            source_pptx=root / "source.pptx",
            work_dir=root,
            images_dir=root,
            masks_dir=root / "masks",
            cleaned_dir=root / "cleaned",
            assets_dir=root / "assets",
            slides=[slide],
            slide_width=1000,
            slide_height=800,
        )

    def test_ai_asset_fields_round_trip_and_legacy_defaults_remain_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            slide.visual_assets = [
                VisualAsset(
                    asset_id="device",
                    bbox=(10, 10, 40, 30),
                    segmentation_mode="sam2",
                    confidence=0.91,
                    confirmed=True,
                    model_id="sam2.1_hiera_tiny",
                    mask_version=1,
                )
            ]
            restored = ppt_project_from_data(ppt_project_to_data(self.make_project(root, slide)))
            asset = restored.slides[0].visual_assets[0]
            self.assertEqual(asset.segmentation_mode, "sam2")
            self.assertEqual(asset.confidence, 0.91)
            self.assertTrue(asset.confirmed)
            self.assertEqual(asset.model_id, "sam2.1_hiera_tiny")
            self.assertEqual(asset.mask_version, 1)

            legacy = ppt_project_to_data(self.make_project(root, slide))
            legacy_asset = legacy["slides"][0]["visual_assets"][0]
            for key in ("segmentation_mode", "confidence", "confirmed", "model_id", "mask_version"):
                legacy_asset.pop(key)
            old = ppt_project_from_data(legacy).slides[0].visual_assets[0]
            self.assertEqual(old.segmentation_mode, "opencv")
            self.assertFalse(old.confirmed)

    def test_stored_ai_mask_is_limited_cleaned_and_keeps_ocr_area_for_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            slide.boxes = [
                OCRBox(
                    text="label",
                    score=1.0,
                    bbox=(20, 20, 10, 5),
                    erase_rect=(20, 20, 30, 25),
                    text_regions=(((20, 20), (30, 20), (30, 25), (20, 25)),),
                )
            ]
            asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))
            slide.visual_assets = [asset]
            raw = np.zeros((80, 100), dtype=np.uint8)
            raw[5:60, 5:80] = 255
            raw[15, 15] = 255
            result = SegmentationResult(raw, 0.88, "cpu")

            store_visual_asset_mask(self.make_project(root, slide), slide, asset, result)
            alpha = visual_asset_alpha_mask(
                np.asarray(Image.open(slide.image_path).convert("RGB")),
                asset,
                slide.boxes,
            )

            self.assertEqual(asset.segmentation_mode, "sam2")
            self.assertEqual(asset.status, "confirmed")
            self.assertTrue(asset.confirmed)
            self.assertGreater(alpha[22, 25], 0)
            self.assertEqual(alpha[5, 5], 0)
            self.assertGreater(np.count_nonzero(alpha[10:40, 10:50]), 0)
            self.assertTrue(asset.mask_path.is_file())
            stored = np.asarray(Image.open(asset.mask_path).convert("L"))
            self.assertGreater(stored[10, 10], 0)

    def test_unconfirmed_candidate_does_not_modify_export_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            image = np.asarray(Image.open(slide.image_path).convert("RGB")).copy()
            image[10:40, 10:50] = (255, 0, 0)
            Image.fromarray(image).save(slide.image_path)
            slide.visual_assets = [VisualAsset(asset_id="candidate", bbox=(10, 10, 40, 30))]
            project = self.make_project(root, slide)

            build_masks(project)

            exported_mask = np.asarray(Image.open(project.masks_dir / slide.image_name).convert("L"))
            self.assertEqual(int(np.count_nonzero(exported_mask)), 0)

    def test_repair_visual_asset_replaces_overlapping_text_without_changing_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            image = np.full((80, 100, 3), (20, 80, 120), dtype=np.uint8)
            image[25:35, 25:35] = (240, 240, 240)
            Image.fromarray(image).save(slide.image_path)
            slide.boxes = [OCRBox("A", 1.0, (25, 25, 10, 10), (24, 24, 36, 36), line_height=10)]
            asset = VisualAsset("device", (10, 10, 40, 40), confirmed=True)
            slide.visual_assets = [asset]
            project = self.make_project(root, slide)
            build_masks(project)
            alpha_before = np.asarray(Image.open(asset.image_path).convert("RGBA"))[:, :, 3].copy()

            class FakeModel:
                def __call__(self, crop, mask, request):
                    result = crop[:, :, ::-1].copy()
                    result[mask > 0] = (30, 20, 10)
                    return result

            repair_visual_assets(project, FakeModel(), object())

            repaired = np.asarray(Image.open(asset.image_path).convert("RGBA"))
            self.assertTrue(np.array_equal(repaired[:, :, 3], alpha_before))
            self.assertLess(int(repaired[20, 20, 0]), 240)
            self.assertEqual(asset.mask_version, 2)

    def test_repair_visual_asset_masks_neighboring_asset_from_model_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            image = np.full((80, 100, 3), (20, 80, 120), dtype=np.uint8)
            image[10:50, 45:65] = (255, 0, 0)
            Image.fromarray(image).save(slide.image_path)
            slide.boxes = [OCRBox("A", 1.0, (25, 25, 10, 10), (24, 24, 36, 36), line_height=10)]
            current = VisualAsset("current", (10, 10, 30, 40), confirmed=True)
            neighbor = VisualAsset("neighbor", (45, 10, 20, 40), confirmed=True)
            slide.visual_assets = [current, neighbor]
            project = self.make_project(root, slide)
            build_masks(project)
            captured = {}

            class CapturingModel:
                def __call__(self, crop, mask, request):
                    captured["crop"] = crop.copy()
                    captured["mask"] = mask.copy()
                    return crop[:, :, ::-1]

            repair_visual_assets(project, CapturingModel(), object())

            self.assertFalse(np.any(np.all(captured["crop"] == (255, 0, 0), axis=2)))
            self.assertGreater(int(np.count_nonzero(captured["mask"])), 0)

    def test_repair_visual_asset_failure_keeps_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            slide.boxes = [OCRBox("A", 1.0, (25, 25, 10, 10), (24, 24, 36, 36), line_height=10)]
            asset = VisualAsset("device", (10, 10, 40, 40), confirmed=True)
            slide.visual_assets = [asset]
            project = self.make_project(root, slide)
            build_masks(project)
            original_bytes = asset.image_path.read_bytes()
            messages = []

            class FailingModel:
                def __call__(self, crop, mask, request):
                    raise RuntimeError("repair failed")

            repair_visual_assets(project, FailingModel(), object(), progress=messages.append)

            self.assertEqual(asset.image_path.read_bytes(), original_bytes)
            self.assertEqual(asset.status, "repair_warning")
            self.assertTrue(any("保留原图" in message for message in messages))

    def test_neighbor_context_mask_does_not_erase_overlap_inside_current_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            image = np.full((80, 100, 3), (20, 80, 120), dtype=np.uint8)
            image[20:40, 35:45] = (10, 200, 30)
            Image.fromarray(image).save(slide.image_path)
            slide.boxes = [OCRBox("A", 1.0, (20, 20, 8, 8), (19, 19, 29, 29), line_height=8)]
            current = VisualAsset("current", (10, 10, 40, 40), confirmed=True)
            neighbor = VisualAsset("neighbor", (35, 10, 25, 40), confirmed=True)
            slide.visual_assets = [current, neighbor]
            project = self.make_project(root, slide)
            build_masks(project)

            class IdentityModel:
                def __call__(self, crop, mask, request):
                    return crop[:, :, ::-1]

            repair_visual_assets(project, IdentityModel(), object())

            repaired = np.asarray(Image.open(current.image_path).convert("RGB"))
            self.assertEqual(tuple(repaired[15, 28]), (10, 200, 30))


if __name__ == "__main__":
    unittest.main()
