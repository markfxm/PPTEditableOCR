from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.quality_pipeline import (
    PageQualityResult,
    OpenAIImageRepairBackend,
    QualityMode,
    QualityPipeline,
    QualityStatus,
    build_background_erase_mask,
    classify_background,
    decontaminate_asset_rgba,
    expand_text_erase_mask,
    refine_asset_alpha,
)
from app.core import PPTProject, PPTSlide, run_quality_iopaint


class QualityPipelineTest(unittest.TestCase):
    def write_image(self, root: Path, name: str, array: np.ndarray) -> Path:
        path = root / name
        Image.fromarray(array.astype(np.uint8), "RGB").save(path)
        return path

    def test_manifest_reuses_validated_page_when_source_and_settings_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_image(root, "slide.png", np.full((40, 60, 3), 240, np.uint8))
            cleaned = self.write_image(root, "cleaned.png", np.full((40, 60, 3), 240, np.uint8))
            pipeline = QualityPipeline(root / "quality")

            first = pipeline.begin_page(1, source, {"mask_version": 3, "backend": "local"})
            self.assertFalse(first.reused)
            pipeline.complete_page(
                first,
                PageQualityResult(
                    page_index=1,
                    status=QualityStatus.VALIDATED,
                    mode=QualityMode.LOCAL_FAST,
                    issues=(),
                    score=1.0,
                    source_path=source,
                    cleaned_path=cleaned,
                ),
            )

            second = pipeline.begin_page(1, source, {"mask_version": 3, "backend": "local"})

            self.assertTrue(second.reused)
            self.assertEqual(second.result.status, QualityStatus.VALIDATED)
            self.assertEqual(second.result.cleaned_path, cleaned)

    def test_manifest_invalidates_page_when_settings_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_image(root, "slide.png", np.full((40, 60, 3), 240, np.uint8))
            pipeline = QualityPipeline(root / "quality")

            first = pipeline.begin_page(1, source, {"mask_version": 3})
            pipeline.complete_page(
                first,
                PageQualityResult(
                    page_index=1,
                    status=QualityStatus.ACCEPTED_LOCAL,
                    mode=QualityMode.LOCAL_REVIEWED,
                    issues=(),
                    score=1.0,
                    source_path=source,
                    cleaned_path=source,
                ),
            )

            changed = pipeline.begin_page(1, source, {"mask_version": 4})

            self.assertFalse(changed.reused)
            self.assertEqual(changed.result.status, QualityStatus.PENDING)

    def test_accept_local_marks_reviewed_page_reusable_without_changing_its_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_image(root, "slide.png", np.full((40, 60, 3), 240, np.uint8))
            cleaned = self.write_image(root, "cleaned.png", np.full((40, 60, 3), 240, np.uint8))
            pipeline = QualityPipeline(root / "quality")
            session = pipeline.begin_page(1, source, {"mask_version": 3})
            pipeline.complete_page(
                session,
                PageQualityResult(1, QualityStatus.REVIEW_REQUIRED, QualityMode.LOCAL_FAST, ("blurred_repair",), 0.5, source, cleaned),
            )

            accepted = pipeline.accept_local(1)
            reused = pipeline.begin_page(1, source, {"mask_version": 3})

            self.assertEqual(accepted.status, QualityStatus.ACCEPTED_LOCAL)
            self.assertEqual(accepted.mode, QualityMode.LOCAL_REVIEWED)
            self.assertTrue(reused.reused)
            self.assertEqual(reused.result.cleaned_path, cleaned)

    def test_classify_background_distinguishes_flat_regular_and_complex(self):
        flat = np.full((80, 120, 3), (70, 120, 180), dtype=np.uint8)
        grid = np.full((80, 120, 3), 245, dtype=np.uint8)
        grid[:, ::12] = 170
        grid[::10, :] = 170
        rng = np.random.default_rng(8)
        complex_image = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)

        self.assertEqual(classify_background(flat), "flat_or_gradient")
        self.assertEqual(classify_background(grid), "regular_texture")
        self.assertEqual(classify_background(complex_image), "complex")

    def test_quality_evaluation_requests_review_when_masked_region_is_visibly_blurry(self):
        source = np.full((80, 120, 3), 230, dtype=np.uint8)
        for x in range(0, 120, 12):
            cv2.line(source, (x, 0), (x, 79), (30, 30, 30), 2)
        repaired = cv2.GaussianBlur(source, (21, 21), 0)
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[:, 42:78] = 255
        pipeline = QualityPipeline(Path(tempfile.gettempdir()) / "ppttoedit-quality-test")

        result = pipeline.evaluate_local_quality(1, source, repaired, mask)

        self.assertEqual(result.status, QualityStatus.REVIEW_REQUIRED)
        self.assertIn("blurred_repair", result.issues)

    def test_background_erase_mask_expands_beyond_fine_asset_alpha(self):
        alpha = np.zeros((60, 80), dtype=np.uint8)
        alpha[20:40, 30:50] = 255

        erase = build_background_erase_mask(alpha, expansion_px=5)

        self.assertEqual(int(erase[30, 40]), 255)
        self.assertEqual(int(erase[18, 40]), 255)
        self.assertEqual(int(alpha[18, 40]), 0)

    def test_decontamination_removes_background_colour_from_soft_asset_edge(self):
        rgba = np.zeros((3, 3, 4), dtype=np.uint8)
        rgba[:, :, :3] = (20, 80, 220)
        rgba[:, :, 3] = 255
        rgba[1, 0, :3] = (170, 180, 235)
        rgba[1, 0, 3] = 80

        cleaned = decontaminate_asset_rgba(rgba, background_rgb=(250, 250, 250))

        self.assertGreater(int(cleaned[1, 0, 2]), int(cleaned[1, 0, 0]))
        self.assertLess(int(cleaned[1, 0, 0]), int(rgba[1, 0, 0]))
        self.assertEqual(int(cleaned[1, 0, 3]), 80)

    def test_alpha_refinement_preserves_shape_and_keeps_known_foreground_opaque_without_matting_runtime(self):
        image = np.full((20, 30, 3), 240, dtype=np.uint8)
        alpha = np.zeros((20, 30), dtype=np.uint8)
        alpha[5:15, 10:20] = 255

        refined = refine_asset_alpha(image, alpha)

        self.assertEqual(refined.shape, alpha.shape)
        self.assertEqual(refined.dtype, np.uint8)
        self.assertGreater(int(refined[10, 15]), 240)
        self.assertEqual(int(refined[0, 0]), 0)

    def test_text_erase_mask_covers_antialias_and_shadow_beyond_detected_strokes(self):
        strokes = np.zeros((50, 90), dtype=np.uint8)
        strokes[20:30, 35:55] = 255

        erase = expand_text_erase_mask(strokes, line_height=20)

        self.assertEqual(int(erase[25, 45]), 255)
        self.assertEqual(int(erase[16, 45]), 255)

    def test_openai_backend_sends_only_supplied_page_and_mask_and_writes_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_image(root, "page.png", np.full((20, 30, 3), 220, np.uint8))
            mask = root / "mask.png"
            Image.new("L", (30, 20), 255).save(mask)
            output = root / "edited.png"
            calls = []

            class FakeImages:
                def edit(self, **kwargs):
                    calls.append(kwargs)
                    return type("Response", (), {"data": [type("Image", (), {"b64_json": base64.b64encode(b"png-data").decode("ascii")})()]})()

            backend = OpenAIImageRepairBackend(api_key="not-written-anywhere", client=type("Client", (), {"images": FakeImages()})())
            result = backend.repair_background(source, mask, output, "remove the selected text")

            self.assertEqual(result.output_path, output)
            self.assertEqual(output.read_bytes(), b"png-data")
            self.assertEqual(calls[0]["model"], "gpt-image-2")
            self.assertEqual(calls[0]["prompt"], "remove the selected text")
            self.assertNotIn("not-written-anywhere", output.read_text(errors="ignore"))

    def test_second_quality_export_reuses_validated_page_without_invoking_iopaint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            masks = root / "masks"
            cleaned = root / "cleaned"
            images.mkdir()
            masks.mkdir()
            source = self.write_image(images, "slide.png", np.full((40, 60, 3), 220, np.uint8))
            mask = np.zeros((40, 60), dtype=np.uint8)
            mask[12:24, 20:40] = 255
            Image.fromarray(mask).save(masks / "slide.png")
            project = PPTProject(
                source_pptx=root / "source.pptx",
                work_dir=root,
                images_dir=images,
                masks_dir=masks,
                cleaned_dir=cleaned,
                assets_dir=root / "assets",
                slides=[PPTSlide(1, "slide.png", source, 60, 40)],
                slide_width=600,
                slide_height=400,
            )
            calls = []

            def fake_iopaint(_images, _masks, output_dir, *_args, **_kwargs):
                calls.append(True)
                output_dir.mkdir(parents=True, exist_ok=True)
                Image.open(source).save(output_dir / "slide.png")

            with patch("app.core.run_iopaint", side_effect=fake_iopaint):
                first = run_quality_iopaint(project)
                second = run_quality_iopaint(project)

            self.assertEqual(len(calls), 1)
            self.assertEqual(first[0].status, QualityStatus.VALIDATED)
            self.assertTrue(second[0].score >= 0.99)


if __name__ == "__main__":
    unittest.main()
