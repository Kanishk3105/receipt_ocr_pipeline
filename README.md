# Receipt OCR Pipeline

Production-grade receipt OCR pipeline that automatically discovers dataset images, preprocesses them, runs OCR, extracts structured receipt fields, assigns confidence scores, and generates expense summaries.

## Project Structure

```text
receipt_ocr_pipeline/
  config/
    default_config.yaml
  outputs/
    discovery_report.json
    summary.json
    summary.csv
    receipts/
      <receipt_id>.json
    raw_ocr/
      <receipt_id>.json
    logs/
      pipeline.log
  src/receipt_ocr/
    config.py
    logging_utils.py
    discovery.py
    preprocessing.py
    ocr_engine.py
    extraction.py
    confidence.py
    summary.py
    pipeline.py
  run_pipeline.py
  requirements.txt
```

## What the Pipeline Does

1. Recursively scans dataset root for image files.
2. Ignores non-image files safely.
3. Computes extension counts and duplicate groups (SHA-256 hash based).
4. Preprocesses each receipt (denoise, deskew, orientation correction, contrast, threshold).
5. Runs EasyOCR and stores raw line-level OCR with confidence.
6. Extracts fields:
   - `store_name`
   - `date`
   - `items` and `price`
   - `total_amount`
7. Scores field-level confidence based on OCR quality, pattern strength, keywords, and consistency checks.
8. Flags low-confidence fields (`< 0.70`).
9. Saves one JSON per receipt plus summary JSON/CSV.
10. Logs all processing and skips broken files without crashing.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py --config config/default_config.yaml
```

## Output Format (Per Receipt)

```json
{
  "receipt_id": "example",
  "source_path": "...",
  "fields": {
    "store_name": { "value": "ABC Store", "confidence": 0.91, "low_confidence": false },
    "date": { "value": "2024-02-01", "confidence": 0.87, "low_confidence": false },
    "items": {
      "value": [{ "name": "Milk", "price": 2.5 }],
      "confidence": 0.79,
      "low_confidence": false
    },
    "total_amount": { "value": 2.5, "confidence": 0.95, "low_confidence": false },
    "ocr_mean_confidence": { "value": 0.84, "confidence": 0.84, "low_confidence": false }
  },
  "raw_text": "..."
}
```

## Configuration Notes

Edit `config/default_config.yaml` to tune preprocessing and confidence behavior:
- `preprocessing.*` for image cleanup knobs.
- `ocr.*` for EasyOCR settings.
- `confidence.*` for scoring threshold and weights.
- `runtime.max_images` for quick dry-runs.

## Assumptions

- Input receipts are image files with supported extensions.
- Item lines may not be perfectly extractable for all layouts, so uncertain fields return `null`/empty list.
- Total amount fallback uses highest visible numeric amount when explicit total keywords are unavailable.
