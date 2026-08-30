from __future__ import annotations

import hashlib
import json
import base64
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


QUALITY_MANIFEST_VERSION = 1


class QualityMode(str, Enum):
    LOCAL_FAST = "local_fast"
    LOCAL_REVIEWED = "local_reviewed"
    ONLINE_REPAIR = "online_repair"


class QualityStatus(str, Enum):
    PENDING = "pending"
    LOCAL_PROCESSED = "local_processed"
    REVIEW_REQUIRED = "review_required"
    ONLINE_APPROVED = "online_approved"
    ACCEPTED_LOCAL = "accepted_local"
    VALIDATED = "validated"
    FAILED = "failed"


@dataclass(frozen=True)
class PageQualityResult:
    page_index: int
    status: QualityStatus
    mode: QualityMode
    issues: tuple[str, ...]
    score: float
    source_path: Path | None
    cleaned_path: Path | None
    background_kind: str = "complex"

    def to_data(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "status": self.status.value,
            "mode": self.mode.value,
            "issues": list(self.issues),
            "score": self.score,
            "source_path": str(self.source_path) if self.source_path else None,
            "cleaned_path": str(self.cleaned_path) if self.cleaned_path else None,
            "background_kind": self.background_kind,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "PageQualityResult":
        return cls(
            page_index=int(data["page_index"]),
            status=QualityStatus(str(data["status"])),
            mode=QualityMode(str(data["mode"])),
            issues=tuple(str(issue) for issue in data.get("issues", [])),
            score=float(data.get("score", 0.0)),
            source_path=Path(data["source_path"]) if data.get("source_path") else None,
            cleaned_path=Path(data["cleaned_path"]) if data.get("cleaned_path") else None,
            background_kind=str(data.get("background_kind") or "complex"),
        )


@dataclass(frozen=True)
class PageQualitySession:
    page_index: int
    source_hash: str
    settings_hash: str
    result: PageQualityResult
    reused: bool


@dataclass(frozen=True)
class RepairResult:
    output_path: Path
    model: str


class ImageRepairBackend(Protocol):
    def repair_background(self, source_path: Path, mask_path: Path, output_path: Path, prompt: str) -> RepairResult:
        ...


class OpenAIImageRepairBackend:
    """Minimal, explicit OpenAI image-edit client used only after page-level approval."""

    def __init__(self, api_key: str | None = None, *, client=None, model: str = "gpt-image-2"):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client
        self.model = model

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("未设置 OPENAI_API_KEY，无法执行在线高质量修复。")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 OpenAI Python 依赖，无法执行在线高质量修复。") from exc
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def repair_background(self, source_path: Path, mask_path: Path, output_path: Path, prompt: str) -> RepairResult:
        source_path = Path(source_path)
        mask_path = Path(mask_path)
        output_path = Path(output_path)
        if not source_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError("在线修复需要存在的页面图像和透明蒙版。")
        with source_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
            response = self._get_client().images.edit(
                model=self.model,
                image=image_file,
                mask=mask_file,
                prompt=prompt,
            )
        data = getattr(response, "data", None) or []
        encoded = getattr(data[0], "b64_json", None) if data else None
        if not encoded:
            raise RuntimeError("在线修复未返回图像数据。")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(encoded))
        return RepairResult(output_path=output_path, model=self.model)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings_hash(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_repeated_edge_peaks(profile: np.ndarray) -> bool:
    if profile.size < 8:
        return False
    baseline = float(np.median(profile))
    peak = float(np.max(profile))
    if peak < max(0.12, baseline * 2.0):
        return False
    threshold = baseline + max(0.08, (peak - baseline) * 0.6)
    active = profile >= threshold
    starts = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
    return len(starts) >= 2


def classify_background(image: np.ndarray) -> str:
    """Classify a page base into deterministic, regular, or generative-repair candidates."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("背景分类需要 RGB 图像。")
    gray = cv2.cvtColor(np.asarray(image[:, :, :3]), cv2.COLOR_RGB2GRAY)
    if float(np.std(gray.astype(np.float32))) < 12.0:
        return "flat_or_gradient"

    edges = cv2.Canny(gray, 60, 140)
    horizontal_profile = np.mean(edges > 0, axis=1)
    vertical_profile = np.mean(edges > 0, axis=0)
    if _has_repeated_edge_peaks(horizontal_profile) or _has_repeated_edge_peaks(vertical_profile):
        return "regular_texture"
    return "complex"


def build_background_erase_mask(alpha: np.ndarray, expansion_px: int) -> np.ndarray:
    """Expand a fine foreground alpha only for erasing its original page pixels."""
    binary = np.zeros(np.asarray(alpha).shape[:2], dtype=np.uint8)
    binary[np.asarray(alpha) > 8] = 255
    if not np.any(binary):
        return binary
    radius = max(1, int(expansion_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(binary, kernel, iterations=1)


def expand_text_erase_mask(strokes: np.ndarray, line_height: int) -> np.ndarray:
    """Include antialiasing, outline and modest shadow pixels around detected text ink."""
    binary = np.zeros(np.asarray(strokes).shape[:2], dtype=np.uint8)
    binary[np.asarray(strokes) > 0] = 255
    if not np.any(binary):
        return binary
    radius = min(14, max(2, int(round(max(1, line_height) * 0.18))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(binary, kernel, iterations=1)


def decontaminate_asset_rgba(rgba: np.ndarray, background_rgb: tuple[int, int, int]) -> np.ndarray:
    """Remove the known page background colour from partially transparent asset edge pixels."""
    image = np.asarray(rgba)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("边缘去污染需要 RGBA 图片。")
    result = image.copy()
    alpha = result[:, :, 3].astype(np.float32) / 255.0
    soft = (alpha > 0.02) & (alpha < 0.995)
    if not np.any(soft):
        return result
    foreground = result[:, :, :3].astype(np.float32)
    background = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    alpha_3 = np.maximum(alpha[:, :, None], 0.02)
    recovered = (foreground - (1.0 - alpha_3) * background) / alpha_3
    foreground[soft] = np.clip(recovered[soft], 0, 255)
    result[:, :, :3] = np.rint(foreground).astype(np.uint8)
    return result


def refine_asset_alpha(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Use local alpha matting when available; keep a conservative OpenCV fallback offline."""
    source_alpha = np.asarray(alpha, dtype=np.uint8)
    if image.shape[:2] != source_alpha.shape:
        raise ValueError("图片与 alpha 蒙版尺寸必须一致。")
    binary = np.zeros_like(source_alpha)
    binary[source_alpha > 8] = 255
    if not np.any(binary):
        return binary
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
    expanded = cv2.dilate(binary, np.ones((5, 5), dtype=np.uint8), iterations=1)
    trimap = np.zeros(source_alpha.shape, dtype=np.float64)
    trimap[expanded > 0] = 0.5
    trimap[eroded > 0] = 1.0
    try:
        from pymatting import estimate_alpha_cf

        normalized = np.asarray(image[:, :, :3], dtype=np.float64) / 255.0
        refined = estimate_alpha_cf(normalized, trimap)
        refined = np.clip(np.rint(refined * 255.0), 0, 255).astype(np.uint8)
        refined[eroded > 0] = 255
        return refined
    except (ImportError, ValueError, RuntimeError):
        softened = cv2.GaussianBlur(binary, (3, 3), sigmaX=0.6)
        softened[eroded > 0] = 255
        return softened


def _masked_laplacian_variance(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(np.asarray(image[:, :, :3]), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    values = laplacian[np.asarray(mask) > 0]
    return float(np.var(values)) if values.size else 0.0


class QualityPipeline:
    """Persist page-level quality decisions and identify local repair failures."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _manifest_path(self, page_index: int) -> Path:
        return self.page_dir(page_index) / "quality_manifest.json"

    def page_dir(self, page_index: int) -> Path:
        return self.root / f"page_{page_index:03d}"

    def load_results(self) -> list[PageQualityResult]:
        if not self.root.is_dir():
            return []
        results = []
        for manifest_path in sorted(self.root.glob("page_*/quality_manifest.json")):
            payload = self._load_manifest(manifest_path)
            if payload:
                results.append(PageQualityResult.from_data(payload["result"]))
        return sorted(results, key=lambda result: result.page_index)

    def accept_local(self, page_index: int) -> PageQualityResult:
        manifest_path = self._manifest_path(page_index)
        payload = self._load_manifest(manifest_path)
        if not payload:
            raise FileNotFoundError(f"未找到第 {page_index} 页质量结果。")
        current = PageQualityResult.from_data(payload["result"])
        if not current.cleaned_path or not current.cleaned_path.is_file():
            raise FileNotFoundError(f"第 {page_index} 页缺少可接受的本地修复结果。")
        accepted = PageQualityResult(
            page_index=current.page_index,
            status=QualityStatus.ACCEPTED_LOCAL,
            mode=QualityMode.LOCAL_REVIEWED,
            issues=current.issues,
            score=current.score,
            source_path=current.source_path,
            cleaned_path=current.cleaned_path,
            background_kind=current.background_kind,
        )
        payload["result"] = accepted.to_data()
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return accepted

    def begin_page(
        self,
        page_index: int,
        source_path: Path,
        settings: dict[str, Any],
    ) -> PageQualitySession:
        source_path = Path(source_path)
        source_hash = _sha256_file(source_path)
        settings_hash = _settings_hash(settings)
        manifest_path = self._manifest_path(page_index)
        existing = self._load_manifest(manifest_path)
        if existing and existing.get("source_hash") == source_hash and existing.get("settings_hash") == settings_hash:
            result = PageQualityResult.from_data(existing["result"])
            if (
                result.status in {QualityStatus.ACCEPTED_LOCAL, QualityStatus.VALIDATED}
                and result.cleaned_path
                and result.cleaned_path.is_file()
            ):
                return PageQualitySession(page_index, source_hash, settings_hash, result, reused=True)

        return PageQualitySession(
            page_index=page_index,
            source_hash=source_hash,
            settings_hash=settings_hash,
            result=PageQualityResult(
                page_index=page_index,
                status=QualityStatus.PENDING,
                mode=QualityMode.LOCAL_FAST,
                issues=(),
                score=0.0,
                source_path=source_path,
                cleaned_path=None,
            ),
            reused=False,
        )

    def complete_page(self, session: PageQualitySession, result: PageQualityResult) -> Path:
        if result.page_index != session.page_index:
            raise ValueError("质量结果的页码与会话不一致。")
        manifest_path = self._manifest_path(session.page_index)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": QUALITY_MANIFEST_VERSION,
            "source_hash": session.source_hash,
            "settings_hash": session.settings_hash,
            "result": result.to_data(),
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def evaluate_local_quality(
        self,
        page_index: int,
        source: np.ndarray,
        repaired: np.ndarray,
        erase_mask: np.ndarray,
        *,
        source_path: Path | None = None,
        cleaned_path: Path | None = None,
    ) -> PageQualityResult:
        if source.shape != repaired.shape or source.shape[:2] != erase_mask.shape[:2]:
            raise ValueError("原图、修复图和擦除蒙版尺寸必须一致。")
        mask = np.asarray(erase_mask) > 0
        background_kind = classify_background(source)
        issues: list[str] = []
        if not np.any(mask):
            issues.append("empty_erase_mask")
        else:
            source_detail = _masked_laplacian_variance(source, mask)
            repaired_detail = _masked_laplacian_variance(repaired, mask)
            if source_detail >= 80.0 and repaired_detail < source_detail * 0.35:
                issues.append("blurred_repair")
            source_mean = np.mean(source[mask].astype(np.float32), axis=0)
            repaired_mean = np.mean(repaired[mask].astype(np.float32), axis=0)
            if float(np.linalg.norm(source_mean - repaired_mean)) > 90.0:
                issues.append("color_drift")
        if background_kind == "complex" and issues:
            issues.append("complex_background")

        score = max(0.0, 1.0 - 0.25 * len(issues))
        status = QualityStatus.REVIEW_REQUIRED if issues else QualityStatus.LOCAL_PROCESSED
        return PageQualityResult(
            page_index=page_index,
            status=status,
            mode=QualityMode.LOCAL_FAST,
            issues=tuple(issues),
            score=score,
            source_path=source_path,
            cleaned_path=cleaned_path,
            background_kind=background_kind,
        )

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != QUALITY_MANIFEST_VERSION or not isinstance(payload.get("result"), dict):
            return None
        return payload
