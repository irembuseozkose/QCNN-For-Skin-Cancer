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
    data_dir: str | Path
    output_dir: str | Path

    image_size: tuple[int, int] = (16, 16)
    color_mode: ColorMode = "grayscale"

    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    normalize_pixels: bool = True
    flatten: bool = True

    use_pca: bool = False
    n_components: int | None = None

    encoding_mode: EncodingMode = "angle"
    angle_scale: float = np.pi

    save_intermediate_arrays: bool = False


def load_and_resize_image(
    image_path: str | Path,
    image_size: tuple[int, int],
    color_mode: ColorMode = "grayscale",
) -> np.ndarray:
    """
    Görüntüyü okuyup yeniden boyutlandırır.
    grayscale -> (H, W)
    rgb       -> (H, W, 3)
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        if color_mode == "grayscale":
            img = img.convert("L")
        elif color_mode == "rgb":
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported color_mode: {color_mode}")

        img = img.resize(image_size, Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)

    return arr


def normalize_image_array(x: np.ndarray) -> np.ndarray:
    """
    [0, 255] -> [0, 1]
    """
    return x / 255.0


def flatten_images(x: np.ndarray) -> np.ndarray:
    """
    (N, H, W) or (N, H, W, C) -> (N, D)
    """
    return x.reshape(x.shape[0], -1)


def prepare_for_angle_encoding(x: np.ndarray, angle_scale: float = np.pi) -> np.ndarray:
    """
    Input [0,1] varsayımıyla:
        x -> [0, pi] veya seçilen aralığa taşınır
    """
    return x * angle_scale


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def prepare_for_amplitude_encoding(x: np.ndarray) -> np.ndarray:
    """
    Her örneği L2 normalize eder.
    Amplitude encoding için gereklidir.
    """
    return l2_normalize_rows(x)


def apply_pca(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=42)

    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_val_pca, x_test_pca, pca


def make_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Önce test ayrılır, sonra train içinden validation ayrılır.
    Stratify kullanılır.
    """
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label_id"],
        random_state=random_state,
    )

    val_ratio_relative = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio_relative,
        stratify=df_train_val["label_id"],
        random_state=random_state,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def dataframe_to_arrays(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    color_mode: ColorMode,
    normalize_pixels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for _, row in df.iterrows():
        img = load_and_resize_image(
            image_path=row["filepath"],
            image_size=image_size,
            color_mode=color_mode,
        )
        if normalize_pixels:
            img = normalize_image_array(img)

        images.append(img)
        labels.append(int(row["label_id"]))

    x = np.stack(images, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return x, y


def ensure_power_of_two_features(x: np.ndarray) -> None:
    """
    Amplitude encoding için feature boyutunun 2^n olması gerekir.
    """
    n_features = x.shape[1]
    if n_features <= 0 or (n_features & (n_features - 1)) != 0:
        raise ValueError(
            f"Amplitude encoding requires feature dimension to be power of 2, got {n_features}"
        )


def save_preprocessed_data(
    output_dir: str | Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metadata: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(output_dir / "val.npz", x=x_val, y=y_val)
    np.savez_compressed(output_dir / "test.npz", x=x_test, y=y_test)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def run_preprocessing(cfg: PreprocessConfig) -> dict:
    from src.qcnn.data import scan_image_dataset

    df, label_map = scan_image_dataset(cfg.data_dir)

    df_train, df_val, df_test = make_splits(
        df=df,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    x_train, y_train = dataframe_to_arrays(
        df_train,
        image_size=cfg.image_size,
        color_mode=cfg.color_mode,
        normalize_pixels=cfg.normalize_pixels,
    )
    x_val, y_val = dataframe_to_arrays(
        df_val,
        image_size=cfg.image_size,
        color_mode=cfg.color_mode,
        normalize_pixels=cfg.normalize_pixels,
    )
    x_test, y_test = dataframe_to_arrays(
        df_test,
        image_size=cfg.image_size,
        color_mode=cfg.color_mode,
        normalize_pixels=cfg.normalize_pixels,
    )

    if cfg.save_intermediate_arrays:
        raw_dir = Path(cfg.output_dir) / "intermediate_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.save(raw_dir / "x_train_images.npy", x_train)
        np.save(raw_dir / "x_val_images.npy", x_val)
        np.save(raw_dir / "x_test_images.npy", x_test)

    if cfg.flatten:
        x_train = flatten_images(x_train)
        x_val = flatten_images(x_val)
        x_test = flatten_images(x_test)

    pca_model = None
    explained_variance_ratio = None

    if cfg.use_pca:
        if cfg.n_components is None:
            raise ValueError("n_components must be set when use_pca=True")

        x_train, x_val, x_test, pca_model = apply_pca(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            n_components=cfg.n_components,
        )
        explained_variance_ratio = float(np.sum(pca_model.explained_variance_ratio_))

    if cfg.encoding_mode == "angle":
        x_train = prepare_for_angle_encoding(x_train, angle_scale=cfg.angle_scale)
        x_val = prepare_for_angle_encoding(x_val, angle_scale=cfg.angle_scale)
        x_test = prepare_for_angle_encoding(x_test, angle_scale=cfg.angle_scale)

    elif cfg.encoding_mode == "amplitude":
        if x_train.ndim != 2:
            raise ValueError("Amplitude encoding expects flattened 2D feature matrix.")
        ensure_power_of_two_features(x_train)
        ensure_power_of_two_features(x_val)
        ensure_power_of_two_features(x_test)

        x_train = prepare_for_amplitude_encoding(x_train)
        x_val = prepare_for_amplitude_encoding(x_val)
        x_test = prepare_for_amplitude_encoding(x_test)

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
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "x_train_shape": list(x_train.shape),
        "x_val_shape": list(x_val.shape),
        "x_test_shape": list(x_test.shape),
        "explained_variance_ratio_sum": explained_variance_ratio,
    }

    save_preprocessed_data(
        output_dir=cfg.output_dir,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )

    if pca_model is not None:
        pca_dir = Path(cfg.output_dir) / "artifacts"
        pca_dir.mkdir(parents=True, exist_ok=True)
        np.save(pca_dir / "pca_components.npy", pca_model.components_)
        np.save(pca_dir / "pca_mean.npy", pca_model.mean_)

    return metadata