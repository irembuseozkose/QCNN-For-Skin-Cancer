from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


@dataclass
class QCNNAnsatzConfig:
    n_qubits: int = 8
    add_barriers: bool = False


def conv_block(
    n_qubits: int,
    params: Sequence,
    add_barriers: bool = False,
    name: str = "ConvBlock",
) -> QuantumCircuit:
    """
    FIX: Orijinal implementasyonda her çift için 3 parametre (Ry, Ry, Rz)
    kullanılıyordu; bu yeterli ifade gücü sağlamıyor.
    
    Geliştirilmiş versiyon: Her qubit çifti için 6 parametre
        Ry(a) on q0, Rz(b) on q0
        Ry(c) on q1, Rz(d) on q1
        CX(q0, q1)
        Ry(e) on q0, Ry(f) on q1   <- entanglement sonrası rotasyon
    
    Bu; ZZ-feature map + SU(4) gate kapasitesine daha yakın bir
    expressibility sağlar ve barren plateau riskini azaltır.
    """
    if n_qubits < 2:
        raise ValueError("conv_block requires at least 2 qubits.")

    expected_params = 6 * n_qubits
    if len(params) != expected_params:
        raise ValueError(
            f"conv_block requires {expected_params} parameters, got {len(params)}"
        )

    qc = QuantumCircuit(n_qubits, name=name)
    p = 0

    # even pairs: (0,1), (2,3), ...
    for i in range(0, n_qubits - 1, 2):
        qc.ry(params[p],     i)
        qc.rz(params[p + 1], i)
        qc.ry(params[p + 2], i + 1)
        qc.rz(params[p + 3], i + 1)
        qc.cx(i, i + 1)
        qc.ry(params[p + 4], i)
        qc.ry(params[p + 5], i + 1)
        p += 6

    # odd pairs + wrap-around
    odd_pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    odd_pairs.append((n_qubits - 1, 0))

    for q0, q1 in odd_pairs:
        qc.ry(params[p],     q0)
        qc.rz(params[p + 1], q0)
        qc.ry(params[p + 2], q1)
        qc.rz(params[p + 3], q1)
        qc.cx(q0, q1)
        qc.ry(params[p + 4], q0)
        qc.ry(params[p + 5], q1)
        p += 6

    if add_barriers:
        qc.barrier()

    return qc


def pool_block(
    source_qubits: Sequence[int],
    sink_qubits: Sequence[int],
    params: Sequence,
    n_total_qubits: int,
    add_barriers: bool = False,
    name: str = "PoolBlock",
) -> QuantumCircuit:
    """
    FIX: Orijinal pooling bloğu yeterli parametre içeriyordu ancak
    CRY (controlled-RY) kullanmak daha iyi bilgi aktarımı sağlar.
    
    Geliştirilmiş versiyon: 4 parametre/çift
        Ry(a) on source
        CRY(b, source → sink)   <- koşullu rotasyon, kuantum korelasyonu korur
        Rz(c) on sink
        Ry(d) on sink           <- ek özgürlük derecesi
    """
    if len(source_qubits) != len(sink_qubits):
        raise ValueError("source_qubits and sink_qubits must have same length.")

    n_pairs = len(source_qubits)
    expected_params = 4 * n_pairs
    if len(params) != expected_params:
        raise ValueError(
            f"pool_block requires {expected_params} parameters, got {len(params)}"
        )

    qc = QuantumCircuit(n_total_qubits, name=name)
    p = 0

    for src, sink in zip(source_qubits, sink_qubits):
        if src == sink:
            raise ValueError("source and sink qubits must be different.")

        qc.ry(params[p], src)
        qc.cry(params[p + 1], src, sink)   # CRY: bilgiyi koşullu aktarır
        qc.rz(params[p + 2], sink)
        qc.ry(params[p + 3], sink)
        p += 4

    if add_barriers:
        qc.barrier()

    return qc


def make_conv_parameter_vector(n_qubits: int, prefix: str) -> ParameterVector:
    # FIX: 6 parametre/çift
    return ParameterVector(prefix, length=6 * n_qubits)


def make_pool_parameter_vector(n_pairs: int, prefix: str) -> ParameterVector:
    # FIX: 4 parametre/çift
    return ParameterVector(prefix, length=4 * n_pairs)


def build_qcnn_ansatz_8q(
    conv1_params: Sequence,
    pool1_params: Sequence,
    conv2_params: Sequence,
    pool2_params: Sequence,
    conv3_params: Sequence,
    add_barriers: bool = False,
) -> tuple[QuantumCircuit, list[int]]:
    """
    8 qubitlik QCNN:
    Conv1(8q) -> Pool1(8→4) -> Conv2(4q) -> Pool2(4→2) -> Conv3(2q)
    """
    qc = QuantumCircuit(8, name="QCNN_8Q")

    # Conv1 on all 8 qubits
    qc.compose(
        conv_block(
            n_qubits=8,
            params=conv1_params,
            add_barriers=add_barriers,
            name="Conv1",
        ),
        inplace=True,
    )

    # Pool1: [0,2,4,6] -> [1,3,5,7]
    qc.compose(
        pool_block(
            source_qubits=[0, 2, 4, 6],
            sink_qubits=[1, 3, 5, 7],
            params=pool1_params,
            n_total_qubits=8,
            add_barriers=add_barriers,
            name="Pool1",
        ),
        inplace=True,
    )
    active_after_pool1 = [1, 3, 5, 7]

    # Conv2 on active qubits
    conv2_local = conv_block(
        n_qubits=4,
        params=conv2_params,
        add_barriers=add_barriers,
        name="Conv2",
    )
    qc.compose(conv2_local, qubits=active_after_pool1, inplace=True)

    # Pool2: [1,5] -> [3,7]
    qc.compose(
        pool_block(
            source_qubits=[1, 5],
            sink_qubits=[3, 7],
            params=pool2_params,
            n_total_qubits=8,
            add_barriers=add_barriers,
            name="Pool2",
        ),
        inplace=True,
    )
    active_after_pool2 = [3, 7]

    # Conv3 on final 2 active qubits
    conv3_local = conv_block(
        n_qubits=2,
        params=conv3_params,
        add_barriers=add_barriers,
        name="Conv3",
    )
    qc.compose(conv3_local, qubits=active_after_pool2, inplace=True)

    final_active_qubits = [3, 7]
    return qc, final_active_qubits


def build_parametric_qcnn_8q(
    add_barriers: bool = False,
) -> tuple[QuantumCircuit, dict[str, ParameterVector], list[int]]:
    conv1 = make_conv_parameter_vector(8, prefix="conv1")
    pool1 = make_pool_parameter_vector(4, prefix="pool1")
    conv2 = make_conv_parameter_vector(4, prefix="conv2")
    pool2 = make_pool_parameter_vector(2, prefix="pool2")
    conv3 = make_conv_parameter_vector(2, prefix="conv3")

    qc, final_active_qubits = build_qcnn_ansatz_8q(
        conv1_params=conv1,
        pool1_params=pool1,
        conv2_params=conv2,
        pool2_params=pool2,
        conv3_params=conv3,
        add_barriers=add_barriers,
    )

    params = {
        "conv1": conv1,
        "pool1": pool1,
        "conv2": conv2,
        "pool2": pool2,
        "conv3": conv3,
    }

    return qc, params, final_active_qubits