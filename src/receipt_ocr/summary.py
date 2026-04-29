from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def generate_summary(receipts: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    total_spend = 0.0
    spend_per_store: dict[str, float] = defaultdict(float)
    low_confidence_receipts: list[str] = []

    for receipt in receipts:
        receipt_id = receipt["receipt_id"]
        field_scores = receipt.get("fields", {})
        is_low = any(v.get("low_confidence", False) for v in field_scores.values())
        if is_low:
            low_confidence_receipts.append(receipt_id)

        total_val = field_scores.get("total_amount", {}).get("value")
        if isinstance(total_val, (int, float)):
            total_spend += float(total_val)
            store_name = field_scores.get("store_name", {}).get("value") or "UNKNOWN"
            spend_per_store[str(store_name)] += float(total_val)

    summary = {
        "number_of_receipts": len(receipts),
        "total_spend": round(total_spend, 2),
        "spend_per_store": {k: round(v, 2) for k, v in sorted(spend_per_store.items())},
        "low_confidence_receipts": low_confidence_receipts,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        writer.writerow(["number_of_receipts", summary["number_of_receipts"]])
        writer.writerow(["total_spend", summary["total_spend"]])
        writer.writerow(["low_confidence_receipts_count", len(low_confidence_receipts)])
        writer.writerow([])
        writer.writerow(["store_name", "spend"])
        for store, spend in summary["spend_per_store"].items():
            writer.writerow([store, spend])

    return summary

