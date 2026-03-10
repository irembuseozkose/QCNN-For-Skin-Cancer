from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import random
from typing import Dict, Tuple

import numpy as np

from src.qcnn.encoding import EncodingConfig
from src.qcnn.model import QCNNModel


@dataclass
class TrainConfig:
    data_dir: Path

    n_qubits: int = 8
    encoding_mode: str = "amplitude"
    rotation_gate: str = "ry"
    add_barriers: bool = False

    n_classes: int = 9

    epochs: int = 30
    batch_size: int = 8
    seed: int = 42

    # SPSA ayarları
    lr: float = 1e-2
    c: float = 1e-1

    # classical head init scale
    head_init_scale: float = 1e-2

    # debug / hız için
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    checkpoint_dir: Path = Path("outputs/checkpoints")
    history_path: Path = Path("outputs/reports/train_history.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_split(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = data_dir / f"{split}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    data = np.load(file_path)
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.int64)
    return x, y


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_vals = np.exp(logits)
    return exp_vals / np.sum(exp_vals)


def cross_entropy_from_logits(logits: np.ndarray, y_true: int, eps: float = 1e-12) -> float:
    probs = softmax(logits)
    return -math.log(float(probs[y_true]) + eps)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


class HybridQCNNClassifier:
    """
    Quantum backbone:
        x -> QCNN -> 4-boyutlu olasılık vektörü

    Classical head:
        q_features(4,) -> logits(n_classes,)

    Parametre vektörü:
        [theta_qcnn | W_flat | b]
    """

    def __init__(self, qcnn_model: QCNNModel, n_classes: int, head_init_scale: float = 1e-2):
        self.qcnn_model = qcnn_model
        self.n_classes = n_classes

        self.q_dim = 2 ** len(self.qcnn_model.final_active_qubits)
        self.n_q_params = self.qcnn_model.n_trainable_params
        self.n_head_params = n_classes * self.q_dim + n_classes
        self.total_params = self.n_q_params + self.n_head_params

        w = np.random.randn(n_classes, self.q_dim) * head_init_scale
        b = np.zeros(n_classes, dtype=np.float64)

        self.init_params = np.concatenate(
            [
                np.random.uniform(-0.1, 0.1, size=self.n_q_params),
                w.reshape(-1),
                b,
            ]
        ).astype(np.float64)

    def unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64).reshape(-1)

        if len(params) != self.total_params:
            raise ValueError(
                f"Expected {self.total_params} total parameters, got {len(params)}"
            )

        theta_q = params[: self.n_q_params]

        start = self.n_q_params
        end = start + self.n_classes * self.q_dim
        w = params[start:end].reshape(self.n_classes, self.q_dim)

        b = params[end:end + self.n_classes]

        return theta_q, w, b

    def quantum_features(self, x: np.ndarray, theta_q: np.ndarray) -> np.ndarray:
        return self.qcnn_model.predict_probabilities_statevector(x, theta_q)

    def logits(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        theta_q, w, b = self.unpack_params(params)
        q_feat = self.quantum_features(x, theta_q)
        return w @ q_feat + b

    def predict_proba(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x, params))

    def predict_class(self, x: np.ndarray, params: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(x, params)))

    def loss_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, params: np.ndarray) -> float:
        losses = []
        for x, y in zip(x_batch, y_batch):
            logits = self.logits(x, params)
            loss = cross_entropy_from_logits(logits, int(y))
            losses.append(loss)
        return float(np.mean(losses))

    def evaluate_dataset(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        params: np.ndarray,
    ) -> Dict[str, float]:
        losses = []
        preds = []

        for x, y in zip(x_data, y_data):
            logits = self.logits(x, params)
            losses.append(cross_entropy_from_logits(logits, int(y)))
            preds.append(int(np.argmax(softmax(logits))))

        preds = np.asarray(preds, dtype=np.int64)
        return {
            "loss": float(np.mean(losses)),
            "accuracy": accuracy_score(y_data, preds),
        }


def sample_batch(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.random.choice(len(x), size=batch_size, replace=False)
    return x[idx], y[idx]


def spsa_step(
    model: HybridQCNNClassifier,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    params: np.ndarray,
    lr: float,
    c: float,
) -> Tuple[np.ndarray, float]:
    """
    SPSA gradient estimate:
        g_i ≈ (L(theta + cΔ) - L(theta - cΔ)) / (2 c Δ_i)
    """
    delta = np.random.choice([-1.0, 1.0], size=params.shape[0])

    params_plus = params + c * delta
    params_minus = params - c * delta

    loss_plus = model.loss_on_batch(x_batch, y_batch, params_plus)
    loss_minus = model.loss_on_batch(x_batch, y_batch, params_minus)

    g_hat = (loss_plus - loss_minus) / (2.0 * c * delta)
    new_params = params - lr * g_hat

    current_loss_est = model.loss_on_batch(x_batch, y_batch, new_params)
    return new_params, float(current_loss_est)


def maybe_limit_samples(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(x):
        return x, y

    idx = np.random.choice(len(x), size=max_samples, replace=False)
    return x[idx], y[idx]


def save_checkpoint(
    checkpoint_path: Path,
    params: np.ndarray,
    summary: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        params=params,
        summary_json=json.dumps(summary, ensure_ascii=False),
    )


def save_history(history_path: Path, history: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def fit(cfg: TrainConfig):
    set_seed(cfg.seed)

    x_train, y_train = load_split(cfg.data_dir, "train")
    x_val, y_val = load_split(cfg.data_dir, "val")
    x_test, y_test = load_split(cfg.data_dir, "test")

    x_train, y_train = maybe_limit_samples(x_train, y_train, cfg.max_train_samples)
    x_val, y_val = maybe_limit_samples(x_val, y_val, cfg.max_val_samples)

    encoding_cfg = EncodingConfig(
        n_qubits=cfg.n_qubits,
        encoding_mode=cfg.encoding_mode,
        rotation_gate=cfg.rotation_gate,
        add_barriers=cfg.add_barriers,
    )

    qcnn_model = QCNNModel(encoding_cfg)
    hybrid_model = HybridQCNNClassifier(
        qcnn_model=qcnn_model,
        n_classes=cfg.n_classes,
        head_init_scale=cfg.head_init_scale,
    )

    params = hybrid_model.init_params.copy()
    best_params = params.copy()
    best_val_loss = float("inf")

    steps_per_epoch = max(1, len(x_train) // cfg.batch_size)

    history = {
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(cfg).items()
        },
        "model_summary": qcnn_model.summary(),
        "epochs": [],
    }

    print("Training started")
    print(f"Train samples: {len(x_train)}")
    print(f"Val samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"QCNN params: {hybrid_model.n_q_params}")
    print(f"Head params: {hybrid_model.n_head_params}")
    print(f"Total params: {hybrid_model.total_params}")

    for epoch in range(1, cfg.epochs + 1):
        batch_losses = []

        for _ in range(steps_per_epoch):
            x_batch, y_batch = sample_batch(x_train, y_train, cfg.batch_size)
            params, batch_loss = spsa_step(
                model=hybrid_model,
                x_batch=x_batch,
                y_batch=y_batch,
                params=params,
                lr=cfg.lr,
                c=cfg.c,
            )
            batch_losses.append(batch_loss)

        train_metrics = hybrid_model.evaluate_dataset(x_train, y_train, params)
        val_metrics = hybrid_model.evaluate_dataset(x_val, y_val, params)

        epoch_record = {
            "epoch": epoch,
            "batch_loss_mean": float(np.mean(batch_losses)),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"batch_loss={epoch_record['batch_loss_mean']:.4f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_params = params.copy()

            save_checkpoint(
                cfg.checkpoint_dir / "best_model.npz",
                best_params,
                {
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "model_summary": qcnn_model.summary(),
                },
            )

    test_metrics = hybrid_model.evaluate_dataset(x_test, y_test, best_params)

    history["final"] = {
        "best_val_loss": float(best_val_loss),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
    }

    save_checkpoint(
        cfg.checkpoint_dir / "last_model.npz",
        params,
        {
            "final_train_complete": True,
            "model_summary": qcnn_model.summary(),
        },
    )
    save_history(cfg.history_path, history)

    print("Training finished")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    return hybrid_model, best_params, history