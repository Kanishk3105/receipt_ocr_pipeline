from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.receipt_ocr.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receipt OCR pipeline runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(Path(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

