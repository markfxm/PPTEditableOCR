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
    evaluate_segmentation_mask,
    ppt_project_from_data,
    ppt_project_to_data,
    repair_visual_assets,
    save_segmentation_debug,
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
                    sam_selected_index=2,
                    sam_points=((12.0, 14.0, 1), (8.0, 9.0, 0)),
                )
            ]
            restored = ppt_project_from_data(ppt_project_to_data(self.make_project(root, slide)))
            asset = restored.slides[0].visual_assets[0]
            self.assertEqual(asset.segmentation_mode, "sam2")
            self.assertEqual(asset.confidence, 0.91)
            self.assertTrue(asset.confirmed)
            self.assertEqual(asset.model_id, "sam2.1_hiera_tiny")
            self.assertEqual(asset.mask_version, 1)
            self.assertEqual(asset.sam_selected_index, 2)
            self.assertEqual(asset.sam_points, ((12.0, 14.0, 1), (8.0, 9.0, 0)))

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
            raw[12:38, 12:48] = 255
            raw[18:32, 20:40] = 0
            result = SegmentationResult(
                np.stack([raw]),
                np.asarray([0.88]),
                0,
                "cpu",
            )

            store_visual_asset_mask(
                self.make_project(root, slide),
                slide,
                asset,
                result,
                points=[(22.0, 20.0, 1)],
            )
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
            self.assertEqual(asset.sam_selected_index, 0)
            self.assertEqual(asset.sam_points, ((22.0, 20.0, 1),))
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

    def test_segmentation_warning_asset_does_not_modify_export_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            mask_path = root / "warning.mask.png"
            Image.new("L", (40, 30), 255).save(mask_path)
            slide.visual_assets = [
                VisualAsset(
                    asset_id="warning",
                    bbox=(10, 10, 40, 30),
                    segmentation_mode="sam2",
                    confirmed=True,
                    status="segmentation_warning",
                    segmentation_warning="疑似选中背景矩形",
                    mask_path=mask_path,
                )
            ]
            project = self.make_project(root, slide)

            build_masks(project)

            exported_mask = np.asarray(Image.open(project.masks_dir / slide.image_name).convert("L"))
            self.assertEqual(int(np.count_nonzero(exported_mask)), 0)

    def test_missing_sam_mask_does_not_fall_back_to_foreground_detection(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        image[10:40, 10:50] = (255, 0, 0)
        asset = VisualAsset(
            asset_id="missing",
            bbox=(10, 10, 40, 30),
            segmentation_mode="sam2",
            confirmed=True,
        )

        alpha = visual_asset_alpha_mask(image, asset, [])

        self.assertEqual(int(np.count_nonzero(alpha)), 0)

    def test_segmentation_validation_flags_partial_and_background_rectangle_masks(self):
        asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))
        partial = np.zeros((80, 100), dtype=np.uint8)
        partial[10:14, 10:50] = 255
        rectangle = np.zeros_like(partial)
        rectangle[10:40, 10:50] = 255

        partial_metrics, partial_warning = evaluate_segmentation_mask(partial, asset)
        rectangle_metrics, rectangle_warning = evaluate_segmentation_mask(rectangle, asset)

        self.assertLess(partial_metrics["area_ratio"], 0.2)
        self.assertIn("局部", partial_warning)
        self.assertGreater(rectangle_metrics["fill_ratio"], 0.9)
        self.assertIn("矩形", rectangle_warning)

    def test_segmentation_validation_flags_solid_rectangle_even_with_padding_in_user_box(self):
        asset = VisualAsset(asset_id="hourglass", bbox=(10, 10, 40, 60))
        padded_rectangle = np.zeros((100, 80), dtype=np.uint8)
        padded_rectangle[18:62, 15:45] = 255

        metrics, warning = evaluate_segmentation_mask(padded_rectangle, asset)

        self.assertGreater(metrics["fill_ratio"], 0.92)
        self.assertGreater(metrics["area_ratio"], 0.40)
        self.assertIn("矩形", warning)

    def test_debug_output_keeps_all_candidates_scores_and_selected_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = self.make_slide(root)
            asset = VisualAsset(asset_id="device", bbox=(10, 10, 40, 30))
            masks = np.zeros((3, 80, 100), dtype=np.uint8)
            masks[0, 12:35, 12:45] = 255
            masks[1, 10:40, 10:50] = 255
            masks[2, 20:30, 20:35] = 255
            result = SegmentationResult(masks, np.asarray([0.2, 0.9, 0.1]), 1, "cpu")

            debug_dir = save_segmentation_debug(
                self.make_project(root, slide),
                slide,
                asset,
                result,
                points=[(20.0, 20.0, 1), (8.0, 8.0, 0)],
                run_id="test-run",
            )

            self.assertTrue((debug_dir / "box.json").is_file())
            self.assertTrue((debug_dir / "scores.json").is_file())
            self.assertTrue((debug_dir / "candidate_1_raw.png").is_file())
            self.assertTrue((debug_dir / "candidate_2_raw.png").is_file())
            self.assertTrue((debug_dir / "candidate_3_raw.png").is_file())
            self.assertTrue((debug_dir / "selected_cleaned.png").is_file())

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
            self.assertEqual(tuple(repaired[20, 20, :3]), (10, 20, 30))
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
