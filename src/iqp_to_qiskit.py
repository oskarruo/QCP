from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
import numpy as np

# Note: init_gates and init_coeffs will not probably be used in this project,
# but are included here just in case because they were included in the original Iqpopt package.

# Note: The sampling part uses a perfect simulator (StatevectorSampler) at the moment. 

# params is a list/array of angles, one per gate.
# gates is a list of which qubits each gate acts on.

class IqpCircuitQiskit:
    def __init__(self, n_qubits, gates, init_gates=None, spin_sym=False, bitflip=False):
        self.n_qubits = n_qubits
        self.gates = gates
        self.init_gates = init_gates
        self.spin_sym = spin_sym
        self.bitflip = bitflip

        # Build generator matrix (IQPopt-style)
        self.generators = self._build_generators()

    
    # Build generator matrix
    def _build_generators(self):
        """
        Convert gates into binary generator matrix:
        shape = (n_gates, n_qubits)
        """
        n_gates = len(self.gates)
        generators = np.zeros((n_gates, self.n_qubits), dtype=np.float32)

        for k, gate in enumerate(self.gates):
            qubits = gate[0]  # because format is [[i]] or [[i,j]]
            for q in qubits:
                generators[k, q] = 1

        return generators

    def iqp_circuit(self, params, init_coefs=None):
        qc = QuantumCircuit(self.n_qubits)

        # Spin symmetry
        if self.spin_sym:
            qc.h(0)
            for i in range(1, self.n_qubits):
                qc.cx(0, i)

        # Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        # ZZ decomposition
        def apply_zz(i, j, theta):
            qc.cx(i, j)
            qc.rz(2 * theta, j)
            qc.cx(i, j)

        # Initial gates
        if self.init_gates is not None:
            for par, gate in zip(init_coefs, self.init_gates):
                qubits = gate[0]

                if len(qubits) == 1:
                    qc.rz(2 * par, qubits[0])

                elif len(qubits) == 2:
                    apply_zz(qubits[0], qubits[1], par)

                else:
                    raise ValueError(
                        "Hardware-efficient version supports only 1- and 2-qubit gates"
                    )

        # Trainable gates
        # Loop over each gate in the ansatz and its corresponding parameter
        for par, gate in zip(params, self.gates):
            qubits = gate[0]

            if len(qubits) == 1:
                qc.rz(2 * par, qubits[0])

            elif len(qubits) == 2:
                apply_zz(qubits[0], qubits[1], par)

            else:
                raise ValueError(
                    "Hardware-efficient version supports only 1- and 2-qubit gates"
                )

        # Final Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        return qc

    def sample(self, params, init_coefs=None, shots=1024):
        qc = self.iqp_circuit(params, init_coefs)

        qc.measure_all()

        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=shots)
        result = job.result()

        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()

        samples = []
        for bitstring, cnt in counts.items():
            arr = np.array([int(b) for b in reversed(bitstring)])
            samples.extend([arr] * cnt)

        return np.array(samples)

    def probs(self, params, init_coefs=None):
        qc = self.iqp_circuit(params, init_coefs)

        # Use Statevector directly
        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        return probs
    
    # IQPopt expectation values computation for the training loop
    def op_expval(self, params, ops, init_coefs=None):
        """
        Fast IQP expectation values:
        ⟨Z_S⟩ = ∏ cos(2θ_k)^(overlap)
        """

        params = np.asarray(params, dtype=np.float32)
        ops = np.atleast_2d(ops).astype(np.float32)

        # Combine parameters + generators (OPTIMIZED)
      
        if self.init_gates is not None:
            if init_coefs is None:
                raise ValueError("init_coefs must be provided if init_gates are used")

            init_coefs = np.asarray(init_coefs, dtype=np.float32)

            # build init generators ONCE per call (could be cached for further speedup)
            init_generators = np.zeros((len(self.init_gates), self.n_qubits), dtype=np.float32)

            for k, gate in enumerate(self.init_gates):
                qubits = gate[0]
                init_generators[k, qubits] = 1.0   # vectorized indexing (faster)

            generators = np.concatenate([self.generators, init_generators], axis=0)
            params = np.concatenate([params, init_coefs], axis=0)
        else:
            generators = self.generators

       
        # Matrix multiplication between the observables 
        # and gates (generators) counts how many qubits overlap.
        # Then % 2 gives parity: even → 0 → no effect, odd → 1 → contributes cosine
        overlap = ((ops @ generators.T) % 2).astype(np.int8)   # faster than % 2

        # Cosine terms
        cos_terms = np.cos(2.0 * params).astype(np.float32)

        # FAST product trick (NO np.power)
        # expvals = product_k cos_k^(overlap_ik)
        # = product_k [1 if 0 else cos_k]

        expvals = np.prod(
            np.where(overlap, cos_terms, 1.0),
            axis=1
        )

        # Spin symmetry
        if self.spin_sym:
            odd_mask = (ops.sum(axis=1) & 1).astype(bool)
            expvals[odd_mask] = 0.0

        return expvals if expvals.shape[0] > 1 else expvals[0]