import numpy as np
from iqp_to_qiskit import IqpCircuitQiskit
from utils import nearest_neighbour_IQP_ansatz

# Simple test of the circuit
if __name__ == "__main__":
    n_qubits = 4
    gates = nearest_neighbour_IQP_ansatz(n_qubits)
    iqp = IqpCircuitQiskit(n_qubits=n_qubits, gates=gates)

    params = np.random.rand(len(gates)) * np.pi
    samples = iqp.sample(params, shots=1000)
    probs = iqp.probs(params)

    print("Samples:", samples)
    print("Probs:", probs)
    
