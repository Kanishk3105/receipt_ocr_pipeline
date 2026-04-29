from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    values: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        return cls(values=data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def nested(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.values
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

