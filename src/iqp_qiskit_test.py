import numpy as np
from iqp_to_qiskit import IqpCircuitQiskit
from utils import hardware_efficient_iqp_gates

# Test with hardware efficient gates
if __name__ == "__main__":
    n_qubits = 4
    gates = hardware_efficient_iqp_gates(n_qubits)
    iqp = IqpCircuitQiskit(n_qubits=n_qubits, gates=gates)

    params = np.random.rand(len(gates)) * np.pi
    samples = iqp.sample(params, shots=1000)
    probs = iqp.probs(params)

    ops = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    exp_vals = iqp.op_expval(params, ops, shots=1000)

    print("Samples:", samples)
    print("Probs:", probs)
    print("Expectation values:")
    for op, val in zip(ops, exp_vals):
        print(f"{op} -> {val}")
