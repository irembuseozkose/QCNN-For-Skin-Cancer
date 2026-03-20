from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


ColorMode = Literal["grayscale", "rgb"]
EncodingMode = Literal["angle", "amplitude"]


@dataclass
class PreprocessConfig:
    train_dir: str | Path          # FIX: ayrı train klasörü
    test_dir: str | Path           # FIX: ayrı test klasörü
    output_dir: str | Path

    image_size: tuple[int, int] = (16, 16)
    color_mode: ColorMode = "grayscale"

    # FIX: val, train içinden ayrılır — test'e dokunulmaz
    val_size: float = 0.15
    random_state: int = 42

    normalize_pixels: bool = True
    flatten: bool = True

    use_pca: bool = False
    n_components: int | None = None

    encoding_mode: EncodingMode = "amplitude"
    angle_scale: float = np.pi

    save_intermediate_arrays: bool = False


# ── Görüntü yükleme ──────────────────────────────────────────────────────────

def load_and_resize_image(
    image_path: str | Path,
    image_size: tuple[int, int],
    color_mode: ColorMode = "grayscale",
) -> np.ndarray:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        if color_mode == "grayscale":
            img = img.convert("L")
        elif color_mode == "rgb":
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported color_mode: {color_mode}")
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.float32)


def normalize_image_array(x: np.ndarray) -> np.ndarray:
    return x / 255.0


def flatten_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


# ── Encoding hazırlık ─────────────────────────────────────────────────────────

def prepare_for_angle_encoding(x: np.ndarray, angle_scale: float = np.pi) -> np.ndarray:
    return x * angle_scale


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def prepare_for_amplitude_encoding(x: np.ndarray) -> np.ndarray:
    return l2_normalize_rows(x)


# ── PCA ───────────────────────────────────────────────────────────────────────

def apply_pca(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=42)
    return (
        pca.fit_transform(x_train),
        pca.transform(x_val),
        pca.transform(x_test),
        pca,
    )


# ── Split yardımcıları ────────────────────────────────────────────────────────

def split_train_val(
    df: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    FIX: Test zaten ayrı geldiği için sadece train → train + val bölünür.
    Stratify ile sınıf dağılımı korunur.
    """
    df_train, df_val = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label_id"],
        random_state=random_state,
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def dataframe_to_arrays(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    color_mode: ColorMode,
    normalize_pixels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for _, row in df.iterrows():
        img = load_and_resize_image(row["filepath"], image_size, color_mode)
        if normalize_pixels:
            img = normalize_image_array(img)
        images.append(img)
        labels.append(int(row["label_id"]))

    return (
        np.stack(images, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


# ── Amplitude encoding padding ────────────────────────────────────────────────

def next_power_of_two(n: int) -> int:
    k = 1
    while k < n:
        k <<= 1
    return k


def pad_to_power_of_two(x: np.ndarray) -> np.ndarray:
    n_features = x.shape[1]
    target = next_power_of_two(n_features)
    if target == n_features:
        return x
    pad_width = target - n_features
    print(f"[preprocess] Padding {n_features} → {target} (+{pad_width} zeros)")
    return np.pad(x, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def ensure_power_of_two_features(x: np.ndarray) -> None:
    n = x.shape[1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(
            f"Amplitude encoding: feature dim must be power of 2, got {n}"
        )


# ── Kaydetme ──────────────────────────────────────────────────────────────────

def save_preprocessed_data(
    output_dir: str | Path,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val:   np.ndarray, y_val:   np.ndarray,
    x_test:  np.ndarray, y_test:  np.ndarray,
    metadata: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(output_dir / "val.npz",   x=x_val,   y=y_val)
    np.savez_compressed(output_dir / "test.npz",  x=x_test,  y=y_test)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def run_preprocessing(cfg: PreprocessConfig) -> dict:
    from src.qcnn.data import scan_image_dataset

    # FIX: Train ve Test klasörlerini ayrı ayrı tara
    print("[preprocess] Scanning train dir ...")
    df_all_train, label_map = scan_image_dataset(cfg.train_dir)

    print("[preprocess] Scanning test dir ...")
    df_test_raw, label_map_test = scan_image_dataset(cfg.test_dir)

    # Test klasöründe train'de olmayan sınıf varsa uyar ve yeniden eşleştir
    if set(label_map.keys()) != set(label_map_test.keys()):
        print(
            f"[preprocess] WARNING: train sınıfları != test sınıfları\n"
            f"  train: {sorted(label_map.keys())}\n"
            f"  test : {sorted(label_map_test.keys())}"
        )
        df_test_raw["label_id"] = df_test_raw["label_name"].map(label_map)
        missing = df_test_raw["label_id"].isna().sum()
        if missing > 0:
            raise ValueError(
                f"{missing} test örneği train label_map'te karşılık bulamadı."
            )
        df_test_raw["label_id"] = df_test_raw["label_id"].astype(int)

    # Train'den validation ayır
    df_train, df_val = split_train_val(
        df_all_train,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    print(
        f"[preprocess] train: {len(df_train)} | "
        f"val: {len(df_val)} | "
        f"test: {len(df_test_raw)}"
    )

    # Görüntüleri dizilere çevir
    x_train, y_train = dataframe_to_arrays(df_train,    cfg.image_size, cfg.color_mode, cfg.normalize_pixels)
    x_val,   y_val   = dataframe_to_arrays(df_val,      cfg.image_size, cfg.color_mode, cfg.normalize_pixels)
    x_test,  y_test  = dataframe_to_arrays(df_test_raw, cfg.image_size, cfg.color_mode, cfg.normalize_pixels)

    if cfg.save_intermediate_arrays:
        raw_dir = Path(cfg.output_dir) / "intermediate_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.save(raw_dir / "x_train_images.npy", x_train)
        np.save(raw_dir / "x_val_images.npy",   x_val)
        np.save(raw_dir / "x_test_images.npy",  x_test)

    if cfg.flatten:
        x_train = flatten_images(x_train)
        x_val   = flatten_images(x_val)
        x_test  = flatten_images(x_test)

    # PCA (opsiyonel)
    pca_model = None
    explained_variance_ratio = None

    if cfg.use_pca:
        if cfg.n_components is None:
            raise ValueError("n_components must be set when use_pca=True")
        x_train, x_val, x_test, pca_model = apply_pca(
            x_train, x_val, x_test, cfg.n_components
        )
        explained_variance_ratio = float(np.sum(pca_model.explained_variance_ratio_))
        print(f"[preprocess] PCA explained variance: {explained_variance_ratio:.4f}")

    # Encoding'e göre son işlem
    if cfg.encoding_mode == "angle":
        x_train = prepare_for_angle_encoding(x_train, cfg.angle_scale)
        x_val   = prepare_for_angle_encoding(x_val,   cfg.angle_scale)
        x_test  = prepare_for_angle_encoding(x_test,  cfg.angle_scale)

    elif cfg.encoding_mode == "amplitude":
        if x_train.ndim != 2:
            raise ValueError("Amplitude encoding expects flattened 2D feature matrix.")

        n_features = x_train.shape[1]
        if (n_features & (n_features - 1)) != 0:
            x_train = pad_to_power_of_two(x_train)
            x_val   = pad_to_power_of_two(x_val)
            x_test  = pad_to_power_of_two(x_test)

        ensure_power_of_two_features(x_train)

        x_train = prepare_for_amplitude_encoding(x_train)
        x_val   = prepare_for_amplitude_encoding(x_val)
        x_test  = prepare_for_amplitude_encoding(x_test)
    else:
        raise ValueError(f"Unsupported encoding_mode: {cfg.encoding_mode}")

    metadata = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(cfg).items()
        },
        "label_map": label_map,
        "num_classes": len(label_map),
        "train_size": int(len(y_train)),
        "val_size":   int(len(y_val)),
        "test_size":  int(len(y_test)),
        "x_train_shape": list(x_train.shape),
        "x_val_shape":   list(x_val.shape),
        "x_test_shape":  list(x_test.shape),
        "explained_variance_ratio_sum": explained_variance_ratio,
    }

    save_preprocessed_data(
        output_dir=cfg.output_dir,
        x_train=x_train, y_train=y_train,
        x_val=x_val,     y_val=y_val,
        x_test=x_test,   y_test=y_test,
        metadata=metadata,
    )

    if pca_model is not None:
        pca_dir = Path(cfg.output_dir) / "artifacts"
        pca_dir.mkdir(parents=True, exist_ok=True)
        np.save(pca_dir / "pca_components.npy", pca_model.components_)
        np.save(pca_dir / "pca_mean.npy",       pca_model.mean_)

    return metadata