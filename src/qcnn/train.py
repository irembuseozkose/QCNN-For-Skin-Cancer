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

    epochs: int = 50           # FIX: 20→50, kuantum modeller daha fazla epoch gerektirir
    batch_size: int = 16       # FIX: 8→16, SPSA gürültüsünü azaltır
    seed: int = 42

    # FIX: SPSA ayarları — literatür önerileri
    # a_coeff: learning rate katsayısı (SPSA notasyonu: a_k = lr / (k+1+A)^alpha)
    # c_coeff: perturbation katsayısı (c_k = c / (k+1)^gamma)
    lr: float = 0.15           # FIX: 1e-2 → 0.15 (SPSA için çok daha uygun)
    c: float = 0.10            # FIX: 1e-1 → 0.10 (küçük pertürbasyon)
    spsa_alpha: float = 0.602  # SPSA teorik optimal
    spsa_gamma: float = 0.101  # SPSA teorik optimal
    spsa_A: float = 10.0       # Stability sabiti (toplam epoch'un ~%10'u)

    # classical head init scale
    head_init_scale: float = 1e-2

    # Sınıf dengesizliği için class weights kullan
    use_class_weights: bool = True

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


def cross_entropy_from_logits(
    logits: np.ndarray,
    y_true: int,
    class_weight: float = 1.0,
    eps: float = 1e-12,
) -> float:
    probs = softmax(logits)
    # FIX: class_weight ile ağırlıklı cross-entropy
    return -class_weight * math.log(float(probs[y_true]) + eps)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    FIX: HAM-10000 gibi dengesiz veri setleri için zorunlu.
    sklearn'ün 'balanced' formülü: w_i = N / (n_classes * count_i)
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)  # sıfır bölmeyi önle
    n = len(y)
    weights = n / (n_classes * counts)
    return weights


class HybridQCNNClassifier:
    """
    Quantum backbone:
        x -> QCNN -> 4-boyutlu olasılık vektörü (2^2 = 4, 2 final qubit)

    FIX: Classical head genişletildi:
        q_features(4,) -> hidden(16,) -> logits(n_classes,)
    
    Tek doğrusal katman 4→9 sınıflandırma için çok zayıf.
    2 katmanlı küçük MLP quantum özelliklerini daha iyi kullanır.
    """

    def __init__(
        self,
        qcnn_model: QCNNModel,
        n_classes: int,
        head_init_scale: float = 1e-2,
        hidden_dim: int = 16,   # FIX: gizli katman boyutu
    ):
        self.qcnn_model = qcnn_model
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.q_dim = 2 ** len(self.qcnn_model.final_active_qubits)
        self.n_q_params = self.qcnn_model.n_trainable_params

        # FIX: 2 katmanlı head: q_dim -> hidden_dim -> n_classes
        self.n_head_params = (
            self.q_dim * hidden_dim + hidden_dim +     # W1, b1
            hidden_dim * n_classes + n_classes          # W2, b2
        )
        self.total_params = self.n_q_params + self.n_head_params

        # Init: küçük rastgele QCNN params + küçük head params
        np.random.seed(None)  # seed dışarıdan set_seed ile zaten ayarlı
        w1 = np.random.randn(hidden_dim, self.q_dim) * head_init_scale
        b1 = np.zeros(hidden_dim)
        w2 = np.random.randn(n_classes, hidden_dim) * head_init_scale
        b2 = np.zeros(n_classes)

        self.init_params = np.concatenate([
            np.random.uniform(-0.01, 0.01, size=self.n_q_params),  # FIX: küçük init
            w1.reshape(-1), b1,
            w2.reshape(-1), b2,
        ]).astype(np.float64)

    def unpack_params(
        self, params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        if len(params) != self.total_params:
            raise ValueError(
                f"Expected {self.total_params} total parameters, got {len(params)}"
            )

        theta_q = params[: self.n_q_params]
        offset = self.n_q_params

        w1 = params[offset: offset + self.q_dim * self.hidden_dim].reshape(
            self.hidden_dim, self.q_dim
        )
        offset += self.q_dim * self.hidden_dim
        b1 = params[offset: offset + self.hidden_dim]
        offset += self.hidden_dim

        w2 = params[offset: offset + self.hidden_dim * self.n_classes].reshape(
            self.n_classes, self.hidden_dim
        )
        offset += self.hidden_dim * self.n_classes
        b2 = params[offset: offset + self.n_classes]

        return theta_q, w1, b1, w2, b2

    def quantum_features(self, x: np.ndarray, theta_q: np.ndarray) -> np.ndarray:
        return self.qcnn_model.predict_probabilities_statevector(x, theta_q)

    def logits(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        theta_q, w1, b1, w2, b2 = self.unpack_params(params)
        q_feat = self.quantum_features(x, theta_q)

        # FIX: 2 katmanlı MLP forward pass
        h = np.tanh(w1 @ q_feat + b1)   # tanh aktivasyon
        return w2 @ h + b2

    def predict_proba(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x, params))

    def predict_class(self, x: np.ndarray, params: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(x, params)))

    def loss_on_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        params: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> float:
        losses = []
        for x, y in zip(x_batch, y_batch):
            logits = self.logits(x, params)
            weight = float(class_weights[int(y)]) if class_weights is not None else 1.0
            loss = cross_entropy_from_logits(logits, int(y), class_weight=weight)
            losses.append(loss)
        return float(np.mean(losses))

    def evaluate_dataset(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        params: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> Dict[str, float]:
        losses = []
        preds = []

        for x, y in zip(x_data, y_data):
            logits = self.logits(x, params)
            weight = float(class_weights[int(y)]) if class_weights is not None else 1.0
            losses.append(cross_entropy_from_logits(logits, int(y), class_weight=weight))
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
    class_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FIX: Stratified sampling — dengesiz veri setlerinde her batch'te
    nadir sınıflar daha fazla örneklenir (weighted random sampling).
    """
    if class_weights is not None:
        sample_probs = class_weights[y]
        sample_probs = sample_probs / sample_probs.sum()
        idx = np.random.choice(len(x), size=batch_size, replace=False, p=sample_probs)
    else:
        idx = np.random.choice(len(x), size=batch_size, replace=False)
    return x[idx], y[idx]


def spsa_step(
    model: HybridQCNNClassifier,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    params: np.ndarray,
    lr_k: float,   # FIX: adım bazlı azalan lr
    c_k: float,    # FIX: adım bazlı azalan pertürbasyon
    class_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """
    FIX: Gerçek SPSA formülasyonu — adım bazlı azalan lr ve c kullanır.
    
    Orijinal kod sabit lr ve c kullanıyordu; bu SPSA'nın yakınsama
    garantisini bozar. Spall (1992) teorisine göre a_k ve c_k
    monoton azalmalıdır.
    """
    delta = np.random.choice([-1.0, 1.0], size=params.shape[0])

    params_plus = params + c_k * delta
    params_minus = params - c_k * delta

    loss_plus = model.loss_on_batch(x_batch, y_batch, params_plus, class_weights)
    loss_minus = model.loss_on_batch(x_batch, y_batch, params_minus, class_weights)

    g_hat = (loss_plus - loss_minus) / (2.0 * c_k * delta)

    # FIX: Gradient clipping — exploding gradient sorununu önler
    g_norm = np.linalg.norm(g_hat)
    if g_norm > 10.0:
        g_hat = g_hat * (10.0 / g_norm)

    new_params = params - lr_k * g_hat
    current_loss = (loss_plus + loss_minus) / 2.0   # FIX: hesaplamayı yeniden kullan
    return new_params, float(current_loss)


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

    # FIX: Class weights hesapla
    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(y_train, cfg.n_classes)
        print("Class weights:", {i: f"{w:.3f}" for i, w in enumerate(class_weights)})

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
    total_steps = cfg.epochs * steps_per_epoch

    history = {
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(cfg).items()
        },
        "model_summary": qcnn_model.summary(),
        "epochs": [],
    }

    print("=" * 60)
    print("Training started")
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    print(f"QCNN params: {hybrid_model.n_q_params}")
    print(f"Head params: {hybrid_model.n_head_params}")
    print(f"Total params: {hybrid_model.total_params}")
    print(f"Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
    print("=" * 60)

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        batch_losses = []

        for _ in range(steps_per_epoch):
            global_step += 1

            # FIX: SPSA decay schedule — Spall (1992) optimal parametreler
            k = global_step
            lr_k = cfg.lr / (k + 1 + cfg.spsa_A) ** cfg.spsa_alpha
            c_k  = cfg.c  / (k + 1) ** cfg.spsa_gamma

            x_batch, y_batch = sample_batch(
                x_train, y_train, cfg.batch_size, class_weights
            )
            params, batch_loss = spsa_step(
                model=hybrid_model,
                x_batch=x_batch,
                y_batch=y_batch,
                params=params,
                lr_k=lr_k,
                c_k=c_k,
                class_weights=class_weights,
            )
            batch_losses.append(batch_loss)

        train_metrics = hybrid_model.evaluate_dataset(
            x_train, y_train, params, class_weights
        )
        val_metrics = hybrid_model.evaluate_dataset(
            x_val, y_val, params, class_weights
        )

        epoch_record = {
            "epoch": epoch,
            "batch_loss_mean": float(np.mean(batch_losses)),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr_last": float(lr_k),
            "c_last": float(c_k),
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"batch={epoch_record['batch_loss_mean']:.4f} | "
            f"tr_loss={train_metrics['loss']:.4f} | "
            f"tr_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"lr={lr_k:.5f}"
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

    test_metrics = hybrid_model.evaluate_dataset(
        x_test, y_test, best_params, class_weights
    )

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

    print("=" * 60)
    print("Training finished")
    print(f"Best val loss:   {best_val_loss:.4f}")
    print(f"Test loss:       {test_metrics['loss']:.4f}")
    print(f"Test accuracy:   {test_metrics['accuracy']:.4f}")
    print("=" * 60)

    return hybrid_model, best_params, history