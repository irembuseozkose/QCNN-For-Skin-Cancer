from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PreprocessConfig:
    raw_root: Path = Path("data/raw")  # içinde Train/ Test/ var
    processed_dir: Path = Path("data/processed")
    manifest_path: Path = Path("data/processed/manifest.csv")
    label2id_path: Path = Path("data/processed/label2id.json")

    # Kaggle dizin adları (SENİN VERİNDE BÖYLE)
    train_dirname: str = "Train"
    test_dirname: str = "Test"

    # train içinden val ayırma
    val_size: float = 0.15
    seed: int = 42

    # Resize cache
    save_resized: bool = True
    resized_dir: Path = Path("data/processed/resized_224")
    img_size: Tuple[int, int] = (224, 224)

    skip_broken_images: bool = True


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _is_readable_image(p: str) -> bool:
    try:
        with Image.open(p) as img:
            img.verify()
        return True
    except Exception:
        return False


def _collect_from_folder(root: Path, folder_name: str, split_tag: str) -> pd.DataFrame:
    """
    root/folder_name/classA/*.jpg
    split_tag manifest içinde 'train'/'test' olarak tutulur.
    """
    split_dir = root / folder_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing: {split_dir.resolve()}")

    rows = []
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders under {split_dir.resolve()}")

    for class_dir in sorted(class_dirs):
        label = class_dir.name
        for fp in class_dir.rglob("*"):
            if _is_image(fp):
                rows.append({"path": str(fp), "label": label, "split": split_tag})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {split_dir.resolve()}")
    return df


def build_manifest(cfg: PreprocessConfig) -> pd.DataFrame:
    # 1) Train/Test oku (senin klasör adların: Train, Test)
    train_df = _collect_from_folder(cfg.raw_root, cfg.train_dirname, "train")
    test_df  = _collect_from_folder(cfg.raw_root, cfg.test_dirname,  "test")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 2) Bozuk dosyaları ele (önerilir)
    if cfg.skip_broken_images:
        ok = [_is_readable_image(p) for p in full_df["path"].tolist()]
        full_df = full_df[np.array(ok)].reset_index(drop=True)

    # 3) Label encoding
    labels = sorted(full_df["label"].unique().tolist())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(labels)}
    full_df["label_id"] = full_df["label"].map(label2id).astype(int)

    # 4) Val split (SADECE train içinden; test'e dokunma)
    train_only = full_df[full_df["split"] == "train"].copy()
    train_split, val_split = train_test_split(
        train_only,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=train_only["label_id"],
    )
    train_split = train_split.copy()
    val_split = val_split.copy()
    train_split["split"] = "train"
    val_split["split"] = "val"

    test_split = full_df[full_df["split"] == "test"].copy()
    manifest = pd.concat([train_split, val_split, test_split], ignore_index=True)

    # 5) Kaydet
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.manifest_path, index=False)
    cfg.label2id_path.write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def make_resize_cache(cfg: PreprocessConfig) -> pd.DataFrame:
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {cfg.manifest_path.resolve()}")

    df = pd.read_csv(cfg.manifest_path)
    cfg.resized_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    keep = []

    for i, row in df.iterrows():
        src = Path(row["path"])
        dst = cfg.resized_dir / f"{row['split']}_{row['label']}_{i}{src.suffix.lower()}"
        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(cfg.img_size, resample=Image.BILINEAR)
            img.save(dst, quality=95)
            out_paths.append(str(dst))
            keep.append(True)
        except Exception:
            out_paths.append("")
            keep.append(False)

    df = df[np.array(keep)].reset_index(drop=True)
    df["path_resized"] = np.array(out_paths)[np.array(keep)]
    df.to_csv(cfg.manifest_path, index=False)
    return df