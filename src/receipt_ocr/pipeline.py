from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from .confidence import score_receipt_fields
from .config import PipelineConfig
from .discovery import discover_images
from .extraction import extract_fields
from .logging_utils import setup_logger
from .ocr_engine import OCRFailure, OCREngine
from .preprocessing import preprocess_image
from .summary import generate_summary


def _serialize_line(line: Any) -> dict[str, Any]:
    return {"text": line.text, "confidence": line.confidence, "bbox": line.bbox}


def _checkpoint_path(output_root: Path) -> Path:
    return output_root / "checkpoint.json"


def _failed_receipts_path(output_root: Path) -> Path:
    return output_root / "failed_receipts.json"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_ids": [], "last_successful_receipt": None, "last_processed_receipt": None}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    data.setdefault("processed_ids", [])
    data.setdefault("last_successful_receipt", None)
    data.setdefault("last_processed_receipt", None)
    return data


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(checkpoint, fp, indent=2)


def _load_failed_receipts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return list(data) if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        # If the file is corrupted from an interrupted run, we keep the pipeline moving.
        return []


def _save_failed_receipts_atomic(path: Path, payload: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _append_failed_receipt(path: Path, payload: dict[str, Any]) -> None:
    existing = _load_failed_receipts(path)
    existing.append(payload)
    _save_failed_receipts_atomic(path, existing)


def _safe_release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_receipt_outputs(receipts_dir: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in sorted(receipts_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fp:
                receipts.append(json.load(fp))
        except Exception:  # noqa: BLE001
            continue
    return receipts


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    cfg = PipelineConfig.from_yaml(config_path)
    dataset_root = Path(cfg.get("dataset_root")).resolve()
    output_root = Path(cfg.get("output_root", "./outputs")).resolve()
    logs_dir = output_root / "logs"
    pre_dir = output_root / "preprocessed"
    raw_ocr_dir = output_root / "raw_ocr"
    receipts_dir = output_root / "receipts"
    checkpoint_file = _checkpoint_path(output_root)
    failed_file = _failed_receipts_path(output_root)

    # Stability: keep compute deterministic and avoid GPU allocations when CUDA isn't available.
    try:
        cv2.setNumThreads(0)
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001
        pass
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logger = setup_logger(logs_dir)
    logger.info("Starting receipt OCR pipeline")

    report = discover_images(dataset_root, cfg.get("image_extensions", []))
    logger.info(
        "Discovery complete | images=%d non_images=%d extensions=%s duplicates=%d",
        len(report.all_images),
        report.non_image_files,
        report.by_extension,
        len(report.duplicates),
    )

    save_pre = bool(cfg.nested("runtime", "save_preprocessed_images", default=False))
    max_images = cfg.nested("runtime", "max_images", default=None)
    ocr = OCREngine(cfg.get("ocr", {}))

    # Robustness: if a previous run left a file where a directory should be,
    # rename it to a backup and re-create the directory.
    if raw_ocr_dir.exists() and not raw_ocr_dir.is_dir():
        backup = raw_ocr_dir.with_name(f"{raw_ocr_dir.name}.backup_{time.time_ns()}")
        try:
            raw_ocr_dir.replace(backup)
            logger.warning("raw_ocr output path was a file; renamed to %s", backup.name)
        except Exception:  # noqa: BLE001
            logger.exception("raw_ocr output path is not a directory and cannot be renamed")
    raw_ocr_dir.mkdir(parents=True, exist_ok=True)

    if receipts_dir.exists() and not receipts_dir.is_dir():
        backup = receipts_dir.with_name(f"{receipts_dir.name}.backup_{time.time_ns()}")
        try:
            receipts_dir.replace(backup)
            logger.warning("receipts output path was a file; renamed to %s", backup.name)
        except Exception:  # noqa: BLE001
            logger.exception("receipts output path is not a directory and cannot be renamed")
    receipts_dir.mkdir(parents=True, exist_ok=True)

    if save_pre:
        if pre_dir.exists() and not pre_dir.is_dir():
            backup = pre_dir.with_name(f"{pre_dir.name}.backup_{time.time_ns()}")
            try:
                pre_dir.replace(backup)
                logger.warning("preprocessed output path was a file; renamed to %s", backup.name)
            except Exception:  # noqa: BLE001
                logger.exception("preprocessed output path is not a directory and cannot be renamed")
        pre_dir.mkdir(parents=True, exist_ok=True)

    image_list = report.all_images[: int(max_images)] if max_images else report.all_images
    checkpoint = _load_checkpoint(checkpoint_file)
    processed_ids = set(checkpoint.get("processed_ids", []))
    last_successful_receipt = checkpoint.get("last_successful_receipt")
    last_processed_receipt = checkpoint.get("last_processed_receipt")
    for path in receipts_dir.glob("*.json"):
        processed_ids.add(path.stem)

    # Resume requirement: continue after the last processed receipt (success or failure).
    resume_after = last_processed_receipt or last_successful_receipt
    if resume_after:
        for i, image_path in enumerate(image_list):
            if image_path.stem == resume_after:
                image_list = image_list[i + 1 :]
                logger.info(
                    "Resuming after last_processed_receipt | %s (remaining=%d/%d)",
                    resume_after,
                    len(image_list),
                    len(report.all_images[: int(max_images)] if max_images else report.all_images),
                )
                break
    logger.info(
        "Resume state loaded | already_processed=%d last_successful_receipt=%s last_processed_receipt=%s gpu_enabled=%s",
        len(processed_ids),
        last_successful_receipt,
        last_processed_receipt,
        ocr.using_gpu,
    )

    for idx, image_path in enumerate(image_list, start=1):
        receipt_id = image_path.stem
        if receipt_id in processed_ids:
            logger.info("Skipping %d/%d | %s | status=already_processed", idx, len(image_list), image_path.name)
            continue
        logger.info("Processing %d/%d | %s", idx, len(image_list), image_path.name)
        processed = None
        ocr_lines = None
        extracted = None
        field_scores = None
        receipt_json = None
        try:
            processed = preprocess_image(image_path, cfg.get("preprocessing", {}))
            if save_pre:
                cv2.imwrite(str(pre_dir / f"{receipt_id}.png"), processed)

            ocr_lines, ocr_meta = ocr.run(processed)
            logger.info(
                "OCR complete | file=%s retries=%d original_size=%sx%s attempted_sizes=%s capped=%s",
                image_path.name,
                ocr_meta.retry_count,
                ocr_meta.original_size[0],
                ocr_meta.original_size[1],
                ocr_meta.attempted_sizes,
                ocr_meta.capped,
            )
            with (raw_ocr_dir / f"{receipt_id}.json").open("w", encoding="utf-8") as fp:
                json.dump([_serialize_line(line) for line in ocr_lines], fp, indent=2, ensure_ascii=False)

            extracted = extract_fields(ocr_lines)
            extracted_dict = {
                "store_name": extracted.store_name,
                "date": extracted.date,
                "items": extracted.items,
                "total_amount": extracted.total_amount,
            }

            field_scores = score_receipt_fields(extracted_dict, ocr_lines, cfg.get("confidence", {}))
            receipt_json = {
                "receipt_id": receipt_id,
                "source_path": str(image_path),
                "fields": {
                    key: {
                        "value": val.value,
                        "confidence": val.confidence,
                        "low_confidence": val.low_confidence,
                    }
                    for key, val in field_scores.items()
                },
                "raw_text": extracted.raw_text,
            }

            with (receipts_dir / f"{receipt_id}.json").open("w", encoding="utf-8") as fp:
                json.dump(receipt_json, fp, indent=2, ensure_ascii=False)
            processed_ids.add(receipt_id)
            checkpoint["processed_ids"] = sorted(processed_ids)
            checkpoint["last_successful_receipt"] = receipt_id
            checkpoint["last_processed_receipt"] = receipt_id
            _save_checkpoint(checkpoint_file, checkpoint)
            logger.info("Completed | file=%s status=success", image_path.name)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, OCRFailure):
                logger.exception("Completed | file=%s status=failed reason=%s", image_path.name, exc)
                logger.error(
                    "OCR failure details | file=%s retry_count=%d attempted_sizes=%s ocr_errors=%s",
                    image_path.name,
                    exc.retry_count,
                    exc.attempted_sizes,
                    exc.errors,
                )
                try:
                    _append_failed_receipt(
                        failed_file,
                        {
                            "receipt_id": receipt_id,
                            "source_path": str(image_path),
                            "reason": str(exc),
                            "retry_count": exc.retry_count,
                            "attempted_sizes": exc.attempted_sizes,
                            "ocr_errors": exc.errors,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to write failed_receipts.json (file=%s)", image_path.name)
            else:
                logger.exception("Completed | file=%s status=failed reason=%s", image_path.name, exc)
                try:
                    _append_failed_receipt(
                        failed_file,
                        {
                            "receipt_id": receipt_id,
                            "source_path": str(image_path),
                            "reason": str(exc),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to write failed_receipts.json (file=%s)", image_path.name)
            # Checkpoint robustness: even failed images are considered "processed" for resume.
            processed_ids.add(receipt_id)
            checkpoint["processed_ids"] = sorted(processed_ids)
            checkpoint["last_processed_receipt"] = receipt_id
            try:
                _save_checkpoint(checkpoint_file, checkpoint)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to write checkpoint after failure (file=%s)", image_path.name)
            continue
        finally:
            del processed
            del ocr_lines
            del extracted
            del field_scores
            del receipt_json
            _safe_release_memory()

    all_receipts = _load_receipt_outputs(receipts_dir)
    summary = generate_summary(all_receipts, output_root)
    discovery_path = output_root / "discovery_report.json"
    with discovery_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "image_count": len(report.all_images),
                "non_image_files": report.non_image_files,
                "extensions": report.by_extension,
                "duplicate_groups": {k: [str(p) for p in v] for k, v in report.duplicates.items()},
            },
            fp,
            indent=2,
        )
    logger.info("Pipeline completed | processed=%d", len(all_receipts))
    return {
        "processed_receipts": len(all_receipts),
        "summary": summary,
        "output_root": str(output_root),
    }

