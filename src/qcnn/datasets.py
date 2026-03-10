from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def load_split(processed_dir: str | Path, split: str):
    """
    split: 'train', 'val', 'test'
    """
    processed_dir = Path(processed_dir)
    file_path = processed_dir / f"{split}.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    data = np.load(file_path)
    x = data["x"]
    y = data["y"]
    return x, y


def load_all_splits(processed_dir: str | Path):
    processed_dir = Path(processed_dir)

    x_train, y_train = load_split(processed_dir, "train")
    x_val, y_val = load_split(processed_dir, "val")
    x_test, y_test = load_split(processed_dir, "test")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def load_metadata(processed_dir: str | Path):
    processed_dir = Path(processed_dir)
    meta_path = processed_dir / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata