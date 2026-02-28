from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from qiskit_machine_learning.connectors import TorchConnector

from src.qcnn.ansatz import QCNNConfig, build_qcnn
from src.qcnn.model import MulticlassQNNConfig, build_sampler_qnn


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/processed/features_q")

    n_qubits: int = 8
    n_blocks: int = 2
    measured_qubits: List[int] = None  # default [0,1,2,3]

    n_classes: int = 9
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-2

    # Stabilite
    eps: float = 1e-10

    # Reproducibility
    seed: int = 42


def load_features(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)

    X_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)

    X_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(data_dir / "y_test.npy").astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def build_torch_qcnn(cfg: TrainConfig):
    # 1) QCNN circuit
    qcfg = QCNNConfig(
        n_qubits=cfg.n_qubits,
        n_blocks=cfg.n_blocks,
        measured_qubits=cfg.measured_qubits,
    )
    qc, x_params, w_params = build_qcnn(qcfg)

    # 2) SamplerQNN (16 output)
    mcfg = MulticlassQNNConfig(
    measured_qubits=cfg.measured_qubits,
    n_classes=cfg.n_classes,
    )
    qnn, measured = build_sampler_qnn(qc, x_params, w_params, mcfg)

    # 3) TorchConnector
    # initial_weights boyutu = len(weight_params)
    init_w = 0.01 * np.random.default_rng(cfg.seed).standard_normal(len(w_params)).astype(np.float32)
    model = TorchConnector(qnn, initial_weights=init_w)

    return model, measured


def probs16_to_probs9_torch(probs16):
    return probs16[:, :9]


def nll_from_probs(probs: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    probs: (B, C) olasılıklar
    y: (B,) sınıf id
    loss = -mean(log(p_true))
    """
    probs = torch.clamp(probs, min=eps, max=1.0)
    p_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
    return -torch.mean(torch.log(p_true))


@torch.no_grad()
def accuracy_from_probs(probs: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(probs, dim=1)
    return float((pred == y).float().mean().item())


def train_one_epoch(model, loader: DataLoader, optimizer, eps: float) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for X, y in loader:
        optimizer.zero_grad()

        # model(X) => (B,16) probabilities (TorchConnector output)
        probs16 = model(X)
        probs9 = probs16_to_probs9_torch(probs16)

        loss = nll_from_probs(probs9, y, eps=eps)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += accuracy_from_probs(probs9.detach(), y)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, eps: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for X, y in loader:
        probs16 = model(X)
        probs9 = probs16_to_probs9_torch(probs16)

        loss = nll_from_probs(probs9, y, eps=eps)
        total_loss += float(loss.item())
        total_acc += accuracy_from_probs(probs9, y)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / max(n_batches, 1),
    }


def fit(cfg: TrainConfig) -> Tuple[Any, Dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_features(cfg.data_dir)
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, cfg.batch_size)

    model, measured = build_torch_qcnn(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train": [], "val": [], "measured_qubits": measured}

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, eps=cfg.eps)
        va = evaluate(model, val_loader, eps=cfg.eps)

        history["train"].append(tr)
        history["val"].append(va)

        print(f"Epoch {epoch:03d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} "
              f"| val loss {va['loss']:.4f} acc {va['acc']:.4f}")

    # test evaluate (tek sefer)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    te = evaluate(model, test_loader, eps=cfg.eps)
    history["test"] = te
    print(f"TEST | loss {te['loss']:.4f} acc {te['acc']:.4f}")

    return model, history