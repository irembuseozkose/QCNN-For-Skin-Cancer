from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageSample:
    path: Path
    label_name: str
    label_id: int


def scan_image_dataset(data_dir: str | Path) -> Tuple[pd.DataFrame, dict[str, int]]:
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under: {data_dir}")

    label_map = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
    rows: List[dict] = []

    for class_dir in class_dirs:
        label_name = class_dir.name
        label_id = label_map[label_name]

        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMG_EXTENSIONS:
                rows.append(
                    {
                        "filepath": str(file_path.resolve()),
                        "label_name": label_name,
                        "label_id": label_id,
                    }
                )

    if not rows:
        raise ValueError(f"No image files found under: {data_dir}")

    df = pd.DataFrame(rows)
    return df, label_map