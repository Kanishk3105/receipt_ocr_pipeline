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
    clean = token.replace(",", "")
    try:
        val = float(clean)
        if val <= 0:
            return None
        if val > 100000:
            return None
        return val
    except ValueError:
        return None


def _first_non_empty(lines: list[OCRLine]) -> str | None:
    for line in lines:
        text = line.text.strip()
        if len(text) >= 2 and not MONEY_RE.fullmatch(text):
            return text
    return None


def _extract_date(text: str) -> str | None:
    for patt in DATE_PATTERNS:
        match = patt.search(text)
        if not match:
            continue
        candidate = match.group(0)
        try:
            dt = date_parser.parse(candidate, dayfirst=False, fuzzy=True)
            return dt.date().isoformat()
        except (ValueError, OverflowError):
            continue
    return None


def _extract_total(lines: list[OCRLine]) -> float | None:
    best: float | None = None
    for line in lines:
        txt_low = line.text.lower()
        money = MONEY_RE.findall(line.text)
        if not money:
            continue
        amounts = [_parse_money(m) for m in money]
        amounts = [a for a in amounts if a is not None]
        if not amounts:
            continue
        max_amount = max(amounts)
        if any(key in txt_low for key in TOTAL_KEYWORDS):
            best = max(best, max_amount) if best is not None else max_amount
    if best is not None:
        return round(best, 2)

    all_vals: list[float] = []
    for line in lines:
        for m in MONEY_RE.findall(line.text):
            v = _parse_money(m)
            if v is not None and v < 10000:
                all_vals.append(v)
    return round(max(all_vals), 2) if all_vals else None


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
        store_name=_first_non_empty(lines),
        date=_extract_date(joined_text),
        items=_extract_items(lines),
        total_amount=_extract_total(lines),
        raw_text=joined_text,
    )

