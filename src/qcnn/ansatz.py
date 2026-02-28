from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


@dataclass
class QCNNConfig:
    n_qubits: int = 8
    n_blocks: int = 2
    # multiclass için 4 qubit ölçüp 16 çıktı almak: 0..3 ölçülsün
    measured_qubits: List[int] = None


def angle_encoding(qc: QuantumCircuit, x: ParameterVector) -> None:
    if len(x) != qc.num_qubits:
        raise ValueError("x length must equal number of qubits.")
    for i, theta in enumerate(x):
        qc.ry(theta, i)


def conv2(qc: QuantumCircuit, a: int, b: int, w: ParameterVector, k: int) -> int:
    # 3 parametreli basit 2-qubit blok
    qc.cx(a, b)
    qc.ry(w[k + 0], a)
    qc.rz(w[k + 1], b)
    qc.cx(a, b)
    qc.ry(w[k + 2], b)
    return k + 3


def pool2(qc: QuantumCircuit, source: int, sink: int, w: ParameterVector, k: int) -> int:
    # 2 parametreli pooling benzeri blok
    qc.ry(w[k + 0], sink)
    qc.cx(source, sink)
    qc.rz(w[k + 1], sink)
    return k + 2


def build_qcnn(cfg: QCNNConfig) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Ölçümsüz QCNN devresi döndürür.
    Notebook'ta: qc.draw("mpl") ile görebilirsin.
    """
    if cfg.n_qubits != 8:
        raise ValueError("Bu iskelet 8 qubit hedefli. İstersen genellerim.")

    x = ParameterVector("x", cfg.n_qubits)

    # param sayısı: conv ring (8 çift *3 =24) + pool (4 çift *2 =8) = 32 / blok
    w = ParameterVector("w", 32 * cfg.n_blocks)

    qc = QuantumCircuit(cfg.n_qubits, name="QCNN")
    angle_encoding(qc, x)

    k = 0
    for _ in range(cfg.n_blocks):
        # conv ring
        ring = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)]
        for a, b in ring:
            k = conv2(qc, a, b, w, k)

        # pooling: 0..3 -> 4..7 gibi “azaltma” hissi
        pools = [(0,4),(1,5),(2,6),(3,7)]
        for s, t in pools:
            k = pool2(qc, s, t, w, k)

    return qc, x, w