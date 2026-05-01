from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _estimate_skew(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def preprocess_image(image_path: Path, cfg: dict[str, Any]) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Unreadable image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cfg.get("enable_denoise", True):
        h = int(cfg.get("denoise_h", 12))
        gray = cv2.fastNlMeansDenoising(gray, h=h)

    # Orthogonal (90/180/270) rotation is decided by OCREngine.detect_orientation
    # in pipeline.py — confidence-probe beats gradient heuristics on rotated receipts.

    angle = _estimate_skew(gray)
    if math.fabs(angle) >= float(cfg.get("deskew_min_angle_abs", 0.4)):
        gray = _rotate(gray, angle)

    if cfg.get("enable_clahe", True):
        clip_limit = float(cfg.get("clahe_clip_limit", 2.0))
        tile = int(cfg.get("clahe_tile_grid", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        gray = clahe.apply(gray)

    if cfg.get("enable_adaptive_threshold", False):
        block_size = int(cfg.get("adaptive_block_size", 31))
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        c_val = int(cfg.get("adaptive_c", 10))
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )
    elif cfg.get("enable_global_threshold_fallback", True):
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray

