from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from src.qcnn.encoding import EncodingConfig, build_encoding_circuit
from src.qcnn.ansatz import build_parametric_qcnn_8q


class QCNNModel:

    def __init__(self, encoding_cfg: EncodingConfig):

        self.encoding_cfg = encoding_cfg

        self.qcnn_circuit, self.qcnn_params, self.final_qubits = build_parametric_qcnn_8q()

        self.sampler = StatevectorSampler()

    def build_circuit(self, x: np.ndarray):

        encoding = build_encoding_circuit(x, self.encoding_cfg)

        full_circuit = encoding.compose(self.qcnn_circuit)

        return full_circuit

    def measure(self, circuit: QuantumCircuit):

        meas_circuit = circuit.copy()

        for q in self.final_qubits:
            meas_circuit.measure_all()

        return meas_circuit

    def forward(self, x: np.ndarray):

        circuit = self.build_circuit(x)

        job = self.sampler.run([circuit])
        result = job.result()

        probs = result.quasi_dists[0]

        return probs