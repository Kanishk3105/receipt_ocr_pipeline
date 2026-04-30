from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dateutil import parser as date_parser

from .ocr_engine import OCRLine

DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b", re.IGNORECASE),
]
MONEY_RE = re.compile(r"(?<!\d)(?:\$|USD|INR|Rs\.?|₹)?\s*([0-9]{1,6}(?:[.,][0-9]{1,2})?)")
TOTAL_KEYWORDS = ("total", "amount due", "grand total", "balance due", "net amount")


@dataclass
class ExtractedReceipt:
    store_name: str | None
    date: str | None
    items: list[dict[str, Any]]
    total_amount: float | None
    raw_text: str


def _parse_money(token: str) -> float | None:
    # If the only separator is a comma followed by 1-2 digits, treat it as the
    # decimal point (common in MY/EU receipts: "66,17" means 66.17, not 6617).
    if "," in token and "." not in token:
        if re.fullmatch(r"\d+,\d{1,2}", token):
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")
    else:
        token = token.replace(",", "")
    try:
        val = float(token)
        if val <= 0:
            return None
        if val > 100000:
            return None
        return val
    except ValueError:
        return None


def _line_y_top(line: OCRLine) -> float:
    return min(p[1] for p in line.bbox) if line.bbox else 0.0


def _line_height(line: OCRLine) -> float:
    if not line.bbox:
        return 0.0
    ys = [p[1] for p in line.bbox]
    return max(ys) - min(ys)


def _y_bounds(lines: list[OCRLine]) -> tuple[float, float]:
    if not lines:
        return 0.0, 1.0
    tops = [_line_y_top(l) for l in lines]
    return min(tops), max(tops)


def _extract_store_name(lines: list[OCRLine]) -> str | None:
    if not lines:
        return None
    y_min, y_max = _y_bounds(lines)
    y_range = max(y_max - y_min, 1.0)

    best_text: str | None = None
    best_score = -1.0
    for line in lines:
        text = line.text.strip()
        if len(text) < 3:
            continue
        if MONEY_RE.fullmatch(text):
            continue
        if any(p.search(text) for p in DATE_PATTERNS):
            continue
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if letters < 3 or digits > letters:
            continue

        norm_y = (_line_y_top(line) - y_min) / y_range
        height_score = min(_line_height(line) / 50.0, 1.0)
        score = (1.0 - norm_y) * 2.0 + line.confidence * 1.5 + height_score
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def _extract_date(text: str) -> str | None:
    for patt in DATE_PATTERNS:
        match = patt.search(text)
        if not match:
            continue
        candidate = match.group(0)
        try:
            dt = date_parser.parse(candidate, dayfirst=True, fuzzy=True)
        except (ValueError, OverflowError):
            continue
        if not (2010 <= dt.year <= 2030):
            continue
        return dt.date().isoformat()
    return None


_GRAND_KEYWORDS = ("grand total", "amount due", "balance due", "net amount", "total amount")
_NEGATIVE_KEYWORDS = ("subtotal", "sub total", "saving", "discount", "change", "tender", "rounding")


def _line_amounts(line: OCRLine) -> list[float]:
    out = []
    for m in MONEY_RE.findall(line.text):
        v = _parse_money(m)
        if v is not None:
            out.append(v)
    return out


def _extract_total(lines: list[OCRLine]) -> float | None:
    if not lines:
        return None

    median_h = sorted(_line_height(l) for l in lines)[len(lines) // 2] or 20.0
    y_window = max(median_h * 1.5, 25.0)

    def _amount_near(target_line: OCRLine) -> float | None:
        ty = _line_y_top(target_line)
        candidates: list[float] = []
        for other in lines:
            if abs(_line_y_top(other) - ty) > y_window:
                continue
            for v in _line_amounts(other):
                if 0.5 <= v < 10000:
                    candidates.append(v)
        return max(candidates) if candidates else None

    grand_best: float | None = None
    plain_best: float | None = None
    for line in lines:
        txt_low = line.text.lower()
        if any(neg in txt_low for neg in _NEGATIVE_KEYWORDS):
            continue
        is_grand = any(k in txt_low for k in _GRAND_KEYWORDS)
        is_plain = any(k in txt_low for k in TOTAL_KEYWORDS)
        if not (is_grand or is_plain):
            continue
        amount = _amount_near(line)
        if amount is None:
            continue
        if is_grand:
            grand_best = amount if grand_best is None else max(grand_best, amount)
        else:
            plain_best = amount if plain_best is None else max(plain_best, amount)

    if grand_best is not None:
        return round(grand_best, 2)
    if plain_best is not None:
        return round(plain_best, 2)

    y_min, y_max = _y_bounds(lines)
    y_thresh = y_min + 0.6 * (y_max - y_min)
    bottom_vals: list[float] = []
    for line in lines:
        if _line_y_top(line) < y_thresh:
            continue
        for v in _line_amounts(line):
            if 0.5 <= v < 10000:
                bottom_vals.append(v)
    return round(max(bottom_vals), 2) if bottom_vals else None


def _extract_items(lines: list[OCRLine]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in lines:
        txt = line.text.strip()
        if len(txt) < 3:
            continue
        if any(k in txt.lower() for k in TOTAL_KEYWORDS):
            continue
        money_matches = MONEY_RE.findall(txt)
        if not money_matches:
            continue
        price_token = money_matches[-1]
        price = _parse_money(price_token)
        if price is None:
            continue
        if price > 10000:
            continue
        # Remove final price token to approximate item name.
        name = re.sub(r"(\$|USD|INR|Rs\.?|₹)?\s*[0-9]+(?:[.,][0-9]{1,2})?\s*$", "", txt).strip(" -:")
        if not name or len(name) < 2:
            continue
        # Filter likely IDs/noise-only names.
        letters = sum(ch.isalpha() for ch in name)
        digits = sum(ch.isdigit() for ch in name)
        if letters < 2:
            continue
        if digits > letters * 2:
            continue
        items.append({"name": name, "price": round(price, 2)})
    return items


def extract_fields(lines: list[OCRLine]) -> ExtractedReceipt:
    joined_text = "\n".join([line.text for line in lines])
    return ExtractedReceipt(
        store_name=_extract_store_name(lines),
        date=_extract_date(joined_text),
        items=_extract_items(lines),
        total_amount=_extract_total(lines),
        raw_text=joined_text,
    )

