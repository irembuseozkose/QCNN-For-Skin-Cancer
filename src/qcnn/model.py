from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable

# Sampler import (sürüm uyumlu)
try:
    from qiskit.primitives import Sampler  # bazı sürümlerde var
except Exception:
    from qiskit.primitives import StatevectorSampler as Sampler  # Qiskit 1.x sık

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit import QuantumCircuit


@dataclass
class MulticlassQNNConfig:
    measured_qubits: List[int] | None = None   # None -> [0,1,2,3]
    n_classes: int = 9


def _make_subset_interpret(measured_qubits: List[int]) -> Callable[[int], int]:
    """
    SamplerQNN'nin interpret fonksiyonu.
    Sampler outcome integer'ını alır, sadece measured_qubits bitlerini çıkarıp
    0..(2^k-1) aralığına map eder.

    Not: Qiskit integer bit ordering (LSB) varsayımı kullanılır:
    integer'daki bit q -> qubit q.
    """
    measured_qubits = list(measured_qubits)

    def interpret(x: int) -> int:
        out = 0
        for i, q in enumerate(measured_qubits):
            out |= ((x >> q) & 1) << i
        return out

    return interpret


def build_sampler_qnn(
    base_circuit: QuantumCircuit,
    input_params,
    weight_params,
    cfg: MulticlassQNNConfig,
) -> Tuple[SamplerQNN, List[int]]:
    """
    Bu sürümde ölçüm register'ı eklemiyoruz.
    Sampler 2^n_qubits dağılım döndürse bile interpret ile 2^k output'a indiriyoruz.
    """
    measured = cfg.measured_qubits or [0, 1, 2, 3]
    k = len(measured)
    output_shape = 2 ** k

    sampler = Sampler()
    interpret = _make_subset_interpret(measured)

    qnn = SamplerQNN(
        circuit=base_circuit,
        input_params=list(input_params),
        weight_params=list(weight_params),
        sampler=sampler,
        interpret=interpret,
        output_shape=output_shape,  # <-- kritik: artık 16
    )
    return qnn, measured