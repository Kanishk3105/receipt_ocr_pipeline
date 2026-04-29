from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DiscoveryReport:
    all_images: list[Path]
    duplicates: dict[str, list[Path]]
    by_extension: dict[str, int]
    non_image_files: int


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_images(dataset_root: Path, image_exts: Iterable[str]) -> DiscoveryReport:
    norm_exts = {e.lower() for e in image_exts}
    all_images: list[Path] = []
    by_extension: dict[str, int] = {}
    non_image_files = 0
    hash_map: dict[str, list[Path]] = {}

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in norm_exts:
            all_images.append(path)
            by_extension[ext] = by_extension.get(ext, 0) + 1
            try:
                h = _hash_file(path)
                hash_map.setdefault(h, []).append(path)
            except OSError:
                # unreadable files are still kept in list and handled later
                pass
        else:
            non_image_files += 1

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return DiscoveryReport(
        all_images=sorted(all_images),
        duplicates=duplicates,
        by_extension=by_extension,
        non_image_files=non_image_files,
    )

