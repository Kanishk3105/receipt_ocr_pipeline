from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import easyocr
import cv2
import numpy as np
import torch


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: list[list[float]]


@dataclass
class OCRRunMeta:
    retry_count: int
    original_size: tuple[int, int]
    attempted_sizes: list[tuple[int, int]]
    capped: bool


class OCRFailure(RuntimeError):
    """Raised when OCR fails all fallback attempts for a single image."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[str],
        attempted_sizes: list[tuple[int, int]],
        original_size: tuple[int, int],
        retry_count: int,
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.attempted_sizes = attempted_sizes
        self.original_size = original_size
        self.retry_count = retry_count


class OCREngine:
    def __init__(self, cfg: dict[str, Any]):
        requested_gpu = bool(cfg.get("gpu", False))
        # Hard requirement: disable GPU completely when CUDA isn't available.
        cuda_available = bool(torch.cuda.is_available())
        use_gpu = requested_gpu and cuda_available
        self.reader = easyocr.Reader(
            cfg.get("languages", ["en"]),
            gpu=use_gpu,
            verbose=False,
        )
        self.using_gpu = use_gpu
        self.paragraph = bool(cfg.get("paragraph", False))
        self.min_conf = float(cfg.get("min_ocr_confidence", 0.05))
        self.max_side = int(cfg.get("max_side", 2200))
        self.retry_max_side = int(cfg.get("retry_max_side", 1600))
        self.last_retry_max_side = int(cfg.get("last_retry_max_side", 1024))

    @staticmethod
    def _resize_keep_aspect(image: np.ndarray, max_side: int) -> np.ndarray:
        h, w = image.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return image
        scale = max_side / float(side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def detect_orientation(self, image: np.ndarray) -> int:
        """Probe OCR confidence at all 4 rotations on a downscaled copy.

        Returns one of {0, 90, 180, 270} — the rotation (clockwise) that
        produced the strongest OCR signal. Using OCR-derived confidence
        catches upside-down receipts that gradient-based heuristics miss.
        """
        small = self._resize_keep_aspect(image, 800)
        candidates = {
            0: small,
            90: cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(small, cv2.ROTATE_180),
            270: cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        best_angle = 0
        best_score = -1.0
        for angle, img in candidates.items():
            try:
                results = self.reader.readtext(img, paragraph=False, detail=1)
            except Exception:  # noqa: BLE001
                continue
            score = sum(float(c) for _, _, c in results) if results else 0.0
            if score > best_score:
                best_score = score
                best_angle = angle
        return best_angle

    @staticmethod
    def apply_rotation(image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def run(self, image: np.ndarray) -> tuple[list[OCRLine], OCRRunMeta]:
        h, w = image.shape[:2]
        original_size = (int(w), int(h))
        # Stability-focused fallback sequence (3 attempts):
        # 1) original image
        # 2) downscaled image
        # 3) retry again with smaller size
        caps = [None, self.retry_max_side, self.last_retry_max_side]

        errors: list[str] = []
        attempts: list[tuple[int, int]] = []
        raw: list[Any] | None = None
        retry_count = 0
        for idx, cap in enumerate(caps):
            candidate = image if cap is None else self._resize_keep_aspect(image, cap)
            ch, cw = candidate.shape[:2]
            attempts.append((int(cw), int(ch)))
            try:
                raw = self.reader.readtext(candidate, paragraph=self.paragraph, detail=1)
                retry_count = idx
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"attempt={idx} size={cw}x{ch} err={exc}")
                raw = None

        if raw is None:
            raise OCRFailure(
                "OCR failed after retries: " + " | ".join(errors) if errors else "OCR failed after retries",
                errors=errors,
                attempted_sizes=attempts,
                original_size=original_size,
                retry_count=retry_count,
            )

        lines: list[OCRLine] = []
        for box, text, conf in raw:
            if not text or float(conf) < self.min_conf:
                continue
            lines.append(
                OCRLine(
                    text=text.strip(),
                    confidence=float(conf),
                    bbox=[[float(x), float(y)] for x, y in box],
                )
            )
        meta = OCRRunMeta(
            retry_count=retry_count,
            original_size=original_size,
            attempted_sizes=attempts,
            capped=any(s != original_size for s in attempts),
        )
        return lines, meta

