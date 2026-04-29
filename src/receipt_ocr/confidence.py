from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ocr_engine import OCRLine


@dataclass
class FieldConfidence:
    value: Any
    confidence: float
    low_confidence: bool


def _mean_ocr(lines: list[OCRLine]) -> float:
    if not lines:
        return 0.0
    return sum(line.confidence for line in lines) / len(lines)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(pattern: float, keyword: float, consistency: float, ocr: float, w: dict[str, float]) -> float:
    total = (
        w.get("ocr", 0.45) * ocr
        + w.get("pattern", 0.25) * pattern
        + w.get("keyword", 0.15) * keyword
        + w.get("consistency", 0.15) * consistency
    )
    return _clip(total)


def score_receipt_fields(extracted: dict[str, Any], lines: list[OCRLine], cfg: dict[str, Any]) -> dict[str, FieldConfidence]:
    weights = cfg.get("weights", {})
    low_thr = float(cfg.get("low_confidence_threshold", 0.70))
    mean_ocr = _mean_ocr(lines)
    joined_low = "\n".join([line.text.lower() for line in lines])

    store = extracted.get("store_name")
    store_pattern = 1.0 if isinstance(store, str) and len(store.strip()) >= 2 else 0.0
    store_keyword = 0.5 if any(k in joined_low for k in ("store", "mart", "shop", "supermarket")) else 0.2
    store_consistency = 1.0 if store else 0.0
    store_score = _score(store_pattern, store_keyword, store_consistency, mean_ocr, weights)

    date = extracted.get("date")
    date_pattern = 1.0 if date else 0.0
    date_keyword = 1.0 if "date" in joined_low else 0.3
    date_consistency = 1.0 if date else 0.0
    date_score = _score(date_pattern, date_keyword, date_consistency, mean_ocr, weights)

    total_amount = extracted.get("total_amount")
    total_pattern = 1.0 if isinstance(total_amount, (int, float)) and total_amount > 0 else 0.0
    total_keyword = 1.0 if any(k in joined_low for k in ("total", "amount due", "grand total")) else 0.3
    item_sum = sum(float(i.get("price", 0.0)) for i in extracted.get("items", []))
    if total_amount and item_sum > 0:
        rel_err = abs(item_sum - float(total_amount)) / max(float(total_amount), 1.0)
        total_consistency = 1.0 if rel_err < 0.12 else 0.5 if rel_err < 0.30 else 0.2
    else:
        total_consistency = 0.5 if total_amount else 0.0
    total_score = _score(total_pattern, total_keyword, total_consistency, mean_ocr, weights)

    items = extracted.get("items", [])
    items_pattern = 1.0 if isinstance(items, list) and len(items) > 0 else 0.0
    items_keyword = 0.8 if any(k in joined_low for k in ("qty", "item", "price")) else 0.4
    items_consistency = 1.0 if items else 0.0
    items_score = _score(items_pattern, items_keyword, items_consistency, mean_ocr, weights)

    return {
        "store_name": FieldConfidence(store, round(store_score, 4), store_score < low_thr),
        "date": FieldConfidence(date, round(date_score, 4), date_score < low_thr),
        "items": FieldConfidence(items, round(items_score, 4), items_score < low_thr),
        "total_amount": FieldConfidence(total_amount, round(total_score, 4), total_score < low_thr),
        "ocr_mean_confidence": FieldConfidence(round(mean_ocr, 4), round(mean_ocr, 4), mean_ocr < low_thr),
    }

