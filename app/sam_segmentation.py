from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


MODEL_ID = "sam2.1_hiera_tiny"
MODEL_FILENAME = f"{MODEL_ID}.pt"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
MODEL_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

ProgressCB = Callable[[int, int], None] | None
CancelCB = Callable[[], bool] | None
Downloader = Callable[[str, Path, ProgressCB, CancelCB], None]


@dataclass(frozen=True)
class SegmentationResult:
    masks: np.ndarray
    scores: np.ndarray
    selected_index: int
    device: str
    model_id: str = MODEL_ID

    @property
    def mask(self) -> np.ndarray:
        return self.masks[self.selected_index]

    @property
    def confidence(self) -> float:
        return float(self.scores[self.selected_index])

    def select(self, index: int) -> "SegmentationResult":
        if index < 0 or index >= len(self.masks):
            raise IndexError("SAM 候选蒙版索引超出范围。")
        return SegmentationResult(self.masks, self.scores, int(index), self.device, self.model_id)


def model_path(local_appdata: Path | None = None) -> Path:
    root = local_appdata or Path(os.environ.get("LOCALAPPDATA") or Path.home())
    return root / "PPTEditableOCR" / "models" / "sam2" / MODEL_FILENAME


def verify_model(path: Path, expected_sha256: str = MODEL_SHA256) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.lower()


def _download_url(url: str, target: Path, progress: ProgressCB, cancelled: CancelCB):
    existing = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = existing > 0 and getattr(response, "status", None) == 206
        done = existing if resumed else 0
        total = done + int(response.headers.get("Content-Length") or 0)
        mode = "ab" if resumed else "wb"
        stream = target.open(mode)
        try:
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("模型下载已取消")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        finally:
            stream.close()


def download_model(
    destination: Path | None = None,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    progress: ProgressCB = None,
    cancelled: CancelCB = None,
    downloader: Downloader = _download_url,
) -> Path:
    destination = destination or model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".download")
    try:
        downloader(url, partial, progress, cancelled)
        if not verify_model(partial, expected_sha256):
            raise RuntimeError("SAM 2.1 模型校验失败，请重新下载。")
        os.replace(partial, destination)
        return destination
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def preferred_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _create_predictor(checkpoint: Path, device: str):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_sam2(MODEL_CONFIG, str(checkpoint), device=device)
    model.eval()
    predictor = SAM2ImagePredictor(model)
    return predictor


class SamSegmentationEngine:
    def __init__(
        self,
        checkpoint: Path,
        device: str = "auto",
        predictor_factory: Callable[[Path, str], object] = _create_predictor,
    ):
        self.checkpoint = Path(checkpoint)
        self.requested_device = preferred_device() if device == "auto" else device
        self.predictor_factory = predictor_factory
        self.predictor = None
        self.device = self.requested_device
        self.image_key: object | None = None
        self.image: np.ndarray | None = None

    def _load(self, device: str):
        self.predictor = self.predictor_factory(self.checkpoint, device)
        self.device = device
        self.image_key = None

    def _set_image(self, image: np.ndarray, image_key: object):
        if self.predictor is None:
            self._load(self.device)
        if self.image_key != image_key:
            self.predictor.set_image(image)
            self.image_key = image_key
            self.image = image

    def segment_with_box(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
        point_coords: list[tuple[float, float]] | np.ndarray | None = None,
        point_labels: list[int] | np.ndarray | None = None,
        image_key: object | None = None,
    ) -> SegmentationResult:
        coords = None if point_coords is None else np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        labels = None if point_labels is None else np.asarray(point_labels, dtype=np.int32).reshape(-1)
        if (coords is None) != (labels is None) or (coords is not None and len(coords) != len(labels)):
            raise ValueError("SAM 提示点坐标和标签必须一一对应。")
        key = image_key if image_key is not None else id(image)
        try:
            return self._segment(image, box, coords, labels, key)
        except Exception:
            if self.device != "cuda":
                raise
            self._load("cpu")
            return self._segment(image, box, coords, labels, key)

    def _segment(self, image, box, point_coords, point_labels, image_key):
        self._set_image(image, image_key)
        masks, scores, _logits = self.predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        scores = np.asarray(scores).reshape(-1)
        masks = np.asarray(masks)
        if masks.ndim != 3 or not len(scores):
            raise RuntimeError("SAM 2.1 未返回有效蒙版。")
        if len(masks) != len(scores):
            raise RuntimeError("SAM 2.1 候选蒙版与评分数量不一致。")
        binary_masks = (masks > 0).astype(np.uint8) * 255
        return SegmentationResult(
            masks=binary_masks,
            scores=scores.astype(np.float64),
            selected_index=int(np.argmax(scores)),
            device=self.device,
        )


def segment_with_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    checkpoint: Path,
    device: str = "auto",
    point_coords: list[tuple[float, float]] | np.ndarray | None = None,
    point_labels: list[int] | np.ndarray | None = None,
) -> SegmentationResult:
    return SamSegmentationEngine(checkpoint, device=device).segment_with_box(
        image,
        box,
        point_coords=point_coords,
        point_labels=point_labels,
    )
