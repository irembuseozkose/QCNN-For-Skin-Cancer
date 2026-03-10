from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import Initialize


EncodingMode = Literal["angle", "amplitude"]


@dataclass
class EncodingConfig:
    n_qubits: int
    encoding_mode: EncodingMode = "angle"
    rotation_gate: Literal["rx", "ry", "rz"] = "ry"
    add_barriers: bool = False


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _validate_1d_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)

    if features.ndim != 1:
        raise ValueError(
            f"Encoding expects a single sample as 1D vector, got shape={features.shape}"
        )

    return features


def _validate_angle_features(features: np.ndarray, n_qubits: int) -> np.ndarray:
    features = _validate_1d_features(features)

    if len(features) != n_qubits:
        raise ValueError(
            f"Angle encoding requires feature_dim == n_qubits. "
            f"Got feature_dim={len(features)}, n_qubits={n_qubits}"
        )

    return features


def _validate_amplitude_features(features: np.ndarray, n_qubits: int) -> np.ndarray:
    features = _validate_1d_features(features)
    expected_dim = 2 ** n_qubits

    if len(features) != expected_dim:
        raise ValueError(
            f"Amplitude encoding requires feature_dim == 2^n_qubits. "
            f"Got feature_dim={len(features)}, expected_dim={expected_dim}"
        )

    norm = np.linalg.norm(features)
    if not np.isclose(norm, 1.0, atol=1e-8):
        raise ValueError(
            f"Amplitude encoding requires L2-normalized input. Got norm={norm:.10f}"
        )

    return features


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(features)

    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")

    return features / norm


def angle_encoding(
    features: np.ndarray,
    n_qubits: int,
    rotation_gate: Literal["rx", "ry", "rz"] = "ry",
    add_barriers: bool = False,
) -> QuantumCircuit:
    features = _validate_angle_features(features, n_qubits)

    qc = QuantumCircuit(n_qubits, name="AngleEncoding")

    for i, value in enumerate(features):
        theta = float(value)

        if rotation_gate == "rx":
            qc.rx(theta, i)
        elif rotation_gate == "ry":
            qc.ry(theta, i)
        elif rotation_gate == "rz":
            qc.rz(theta, i)
        else:
            raise ValueError(f"Unsupported rotation_gate: {rotation_gate}")

    if add_barriers:
        qc.barrier()

    return qc


def amplitude_encoding(
    features: np.ndarray,
    n_qubits: int,
    add_barriers: bool = False,
) -> QuantumCircuit:
    features = _validate_amplitude_features(features, n_qubits)

    qc = QuantumCircuit(n_qubits, name="AmplitudeEncoding")
    init = Initialize(features)
    qc.append(init, qc.qubits)

    if add_barriers:
        qc.barrier()

    return qc


def build_encoding_circuit(
    features: np.ndarray,
    cfg: EncodingConfig,
) -> QuantumCircuit:
    if cfg.encoding_mode == "angle":
        return angle_encoding(
            features=features,
            n_qubits=cfg.n_qubits,
            rotation_gate=cfg.rotation_gate,
            add_barriers=cfg.add_barriers,
        )

    if cfg.encoding_mode == "amplitude":
        return amplitude_encoding(
            features=features,
            n_qubits=cfg.n_qubits,
            add_barriers=cfg.add_barriers,
        )

    raise ValueError(f"Unsupported encoding_mode: {cfg.encoding_mode}")


def build_parametric_angle_encoding(
    n_qubits: int,
    rotation_gate: Literal["rx", "ry", "rz"] = "ry",
    prefix: str = "x",
    add_barriers: bool = False,
) -> tuple[QuantumCircuit, ParameterVector]:
    params = ParameterVector(prefix, n_qubits)
    qc = QuantumCircuit(n_qubits, name="ParamAngleEncoding")

    for i in range(n_qubits):
        if rotation_gate == "rx":
            qc.rx(params[i], i)
        elif rotation_gate == "ry":
            qc.ry(params[i], i)
        elif rotation_gate == "rz":
            qc.rz(params[i], i)
        else:
            raise ValueError(f"Unsupported rotation_gate: {rotation_gate}")

    if add_barriers:
        qc.barrier()

    return qc, params


def check_encoding_compatibility(
    features: np.ndarray,
    cfg: EncodingConfig,
) -> None:
    if cfg.encoding_mode == "angle":
        _validate_angle_features(features, cfg.n_qubits)
        return

    if cfg.encoding_mode == "amplitude":
        _validate_amplitude_features(features, cfg.n_qubits)
        return

    raise ValueError(f"Unsupported encoding_mode: {cfg.encoding_mode}")


def load_processed_split(
    processed_dir: str | Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    data/processed/features_q/train.npz gibi dosyalardan x ve y yükler.
    """
    processed_dir = Path(processed_dir)
    split_path = processed_dir / f"{split}.npz"

    if not split_path.exists():
        raise FileNotFoundError(f"Processed split not found: {split_path}")

    data = np.load(split_path)
    x = data["x"]
    y = data["y"]

    return x, y


def get_sample_from_processed_data(
    processed_dir: str | Path,
    split: str,
    index: int,
) -> tuple[np.ndarray, int]:
    """
    Örn:
        x0, y0 = get_sample_from_processed_data("data/processed/features_q", "train", 0)
    """
    x, y = load_processed_split(processed_dir, split)

    if index < 0 or index >= len(x):
        raise IndexError(f"Index out of range: {index}")

    sample_x = np.asarray(x[index], dtype=np.float64)
    sample_y = int(y[index])

    return sample_x, sample_y


def build_encoding_circuit_from_processed_data(
    processed_dir: str | Path,
    split: str,
    index: int,
    cfg: EncodingConfig,
) -> tuple[QuantumCircuit, int, np.ndarray]:
    """
    Preprocessed veriden tek örnek seçer, encoding circuit üretir.

    Dönenler:
        qc         -> encoding devresi
        label      -> örneğin etiketi
        features   -> devreye verilen feature vektörü
    """
    features, label = get_sample_from_processed_data(processed_dir, split, index)

    check_encoding_compatibility(features, cfg)
    qc = build_encoding_circuit(features, cfg)

    return qc, label, features


def build_encoding_circuits_for_indices(
    x_data: np.ndarray,
    indices: list[int],
    cfg: EncodingConfig,
) -> list[QuantumCircuit]:
    """
    Bellekte zaten yüklü x_data üzerinden birden fazla encoding devresi üretir.
    Eğitimde tüm batch için circuit listesi üretmek istersen kullanılabilir.
    """
    circuits: list[QuantumCircuit] = []

    for idx in indices:
        features = np.asarray(x_data[idx], dtype=np.float64)
        check_encoding_compatibility(features, cfg)
        qc = build_encoding_circuit(features, cfg)
        circuits.append(qc)

    return circuits


def infer_feature_dim_from_processed_data(
    processed_dir: str | Path,
    split: str = "train",
) -> int:
    x, _ = load_processed_split(processed_dir, split)

    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D array (N, D), got shape={x.shape}")

    return int(x.shape[1])


def infer_n_qubits_for_amplitude_from_processed_data(
    processed_dir: str | Path,
    split: str = "train",
) -> int:
    feature_dim = infer_feature_dim_from_processed_data(processed_dir, split)

    if not _is_power_of_two(feature_dim):
        raise ValueError(
            f"Feature dimension must be a power of 2 for amplitude encoding. Got {feature_dim}"
        )

    return int(np.log2(feature_dim))