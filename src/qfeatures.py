from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class QFeatureConfig:
    in_dir: Path = Path("data/processed/features_raw")
    out_dir: Path = Path("data/processed/features_q")

    n_components: int = 8     # 8 qubit
    seed: int = 42
    clip_z: float = 3.0


def _load(in_dir: Path, split: str):
    X = np.load(in_dir / f"X_{split}.npy").astype(np.float32)
    y = np.load(in_dir / f"y_{split}.npy").astype(np.int64)
    return X, y


def make_quantum_features(cfg: QFeatureConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _load(cfg.in_dir, "train")
    X_val, y_val     = _load(cfg.in_dir, "val")
    X_test, y_test   = _load(cfg.in_dir, "test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit sadece train
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    pca = PCA(n_components=cfg.n_components, random_state=cfg.seed)
    X_train_p = pca.fit_transform(X_train_s)    # fit sadece train
    X_val_p   = pca.transform(X_val_s)
    X_test_p  = pca.transform(X_test_s)

    cz = float(cfg.clip_z)
    X_train_c = np.clip(X_train_p, -cz, cz)
    X_val_c   = np.clip(X_val_p, -cz, cz)
    X_test_c  = np.clip(X_test_p, -cz, cz)

    pi = np.pi
    X_train_q = (X_train_c / cz) * pi
    X_val_q   = (X_val_c / cz) * pi
    X_test_q  = (X_test_c / cz) * pi

    np.save(cfg.out_dir / "X_train.npy", X_train_q.astype(np.float32))
    np.save(cfg.out_dir / "y_train.npy", y_train)
    np.save(cfg.out_dir / "X_val.npy", X_val_q.astype(np.float32))
    np.save(cfg.out_dir / "y_val.npy", y_val)
    np.save(cfg.out_dir / "X_test.npy", X_test_q.astype(np.float32))
    np.save(cfg.out_dir / "y_test.npy", y_test)

    # reproducibility
    np.save(cfg.out_dir / "scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(cfg.out_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))
    np.save(cfg.out_dir / "pca_components.npy", pca.components_.astype(np.float32))
    np.save(cfg.out_dir / "pca_mean.npy", pca.mean_.astype(np.float32))