from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.qcnn.encoding import EncodingConfig, build_encoding_circuit
from src.qcnn.ansatz import build_parametric_qcnn_8q


@dataclass
class QCNNModelConfig:
    n_qubits: int = 8
    add_barriers: bool = False


class QCNNModel:
    """
    QCNN modeli:
        feature vector
        -> encoding
        -> parametrik QCNN ansatz
        -> final active qubits ölçümü
    """

    def __init__(
        self,
        encoding_cfg: EncodingConfig,
        model_cfg: QCNNModelConfig | None = None,
    ) -> None:
        self.encoding_cfg = encoding_cfg
        self.model_cfg = model_cfg or QCNNModelConfig(
            n_qubits=encoding_cfg.n_qubits,
            add_barriers=encoding_cfg.add_barriers,
        )

        if self.model_cfg.n_qubits != 8:
            raise ValueError(
                "This initial QCNNModel implementation currently supports 8 qubits."
            )

        self.ansatz_circuit, self.ansatz_param_dict, self.final_active_qubits = (
            build_parametric_qcnn_8q(add_barriers=self.model_cfg.add_barriers)
        )

        self.parameter_slices = self._build_parameter_slices()
        self.n_trainable_params = sum(
            len(param_vec) for param_vec in self.ansatz_param_dict.values()
        )

    def _build_parameter_slices(self) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        start = 0

        for name in ["conv1", "pool1", "conv2", "pool2", "conv3"]:
            length = len(self.ansatz_param_dict[name])
            slices[name] = slice(start, start + length)
            start += length

        return slices

    def split_parameter_vector(self, theta: Sequence[float]) -> dict[str, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)

        if len(theta) != self.n_trainable_params:
            raise ValueError(
                f"Expected {self.n_trainable_params} trainable parameters, got {len(theta)}"
            )

        return {
            name: theta[self.parameter_slices[name]]
            for name in self.parameter_slices
        }

    def bind_ansatz_parameters(self, theta: Sequence[float]) -> QuantumCircuit:
        theta_parts = self.split_parameter_vector(theta)

        bind_map = {}
        for name, param_vec in self.ansatz_param_dict.items():
            values = theta_parts[name]
            for param, value in zip(param_vec, values):
                bind_map[param] = float(value)

        return self.ansatz_circuit.assign_parameters(bind_map, inplace=False)

    def build_circuit(
        self,
        x: np.ndarray,
        theta: Sequence[float],
    ) -> QuantumCircuit:
        """
        Tek örnek için tam quantum devre:
            encoding(x) + bound_ansatz(theta)

        Not:
        Amplitude encoding kullanılıyorsa, Qiskit'in Initialize katı norm kontrolü
        yüzünden burada tekrar güvenli normalize yapılır.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if self.encoding_cfg.encoding_mode == "amplitude":
            norm = np.linalg.norm(x)
            if norm <= 1e-12:
                raise ValueError(
                    "Cannot build amplitude circuit from a near-zero vector."
                )
            x = x / norm

        encoding_circuit = build_encoding_circuit(x, self.encoding_cfg)
        ansatz_bound = self.bind_ansatz_parameters(theta)

        full_circuit = encoding_circuit.compose(ansatz_bound)
        return full_circuit

    def build_measured_circuit(
        self,
        x: np.ndarray,
        theta: Sequence[float],
    ) -> QuantumCircuit:
        circuit = self.build_circuit(x, theta)

        measured = QuantumCircuit(circuit.num_qubits, len(self.final_active_qubits))
        measured.compose(circuit, inplace=True)

        for c_idx, q_idx in enumerate(self.final_active_qubits):
            measured.measure(q_idx, c_idx)

        return measured

    def predict_probabilities_statevector(
        self,
        x: np.ndarray,
        theta: Sequence[float],
    ) -> np.ndarray:
        circuit = self.build_circuit(x, theta)
        state = Statevector.from_instruction(circuit)

        probs_dict = state.probabilities_dict(qargs=self.final_active_qubits)

        n_out = 2 ** len(self.final_active_qubits)
        probs = np.zeros(n_out, dtype=np.float64)

        for bitstring, prob in probs_dict.items():
            idx = int(bitstring, 2)
            probs[idx] = prob

        return probs

    def predict_class_statevector(
        self,
        x: np.ndarray,
        theta: Sequence[float],
    ) -> int:
        probs = self.predict_probabilities_statevector(x, theta)
        return int(np.argmax(probs))

    def summary(self) -> dict:
        return {
            "n_qubits": self.model_cfg.n_qubits,
            "encoding_mode": self.encoding_cfg.encoding_mode,
            "rotation_gate": self.encoding_cfg.rotation_gate,
            "n_trainable_params": self.n_trainable_params,
            "final_active_qubits": self.final_active_qubits,
            "parameter_sizes": {
                name: len(vec) for name, vec in self.ansatz_param_dict.items()
            },
        }