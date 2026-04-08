import numpy as np
from iqp_to_qiskit import IqpCircuitQiskit
from utils import median_heuristic_fast, generate_simple_dataset
from ansatzes import nearest_neighbour_IQP_ansatz
from torch_training import TrainerTorch
from torch_methods import mmd_loss_torch
from qiskit import qpy

n_qubits = 50
n_ops = 200
gates = nearest_neighbour_IQP_ansatz(n_qubits)
circuit = IqpCircuitQiskit(n_qubits, gates)

X_train = generate_simple_dataset(n_qubits, n_samples=10000)

print(f"Training data: {len(X_train)} samples")

# Faster version that only uses a fixed subset of the data
sigma = median_heuristic_fast(X_train)

params_init = np.random.normal(0, 1 / np.sqrt(n_qubits), len(gates))
p = (1 - np.exp(-1 / (2 * sigma**2))) / 2
ops = np.random.binomial(1, p, size=(n_ops, n_qubits))

loss_kwargs = {
    "params": params_init,
    "circuit": circuit,
    "ground_truth": X_train,
    "sigma": sigma,
    "n_ops": n_ops,
    "ops": ops,
}

trainer = TrainerTorch(mmd_loss_torch, lr=0.01)
trainer.train(n_iters=100, loss_kwargs=loss_kwargs)

final_circuit = circuit.iqp_circuit(trainer.final_params)
final_circuit.measure_all()
with open("circuit.qpy", "wb") as file:
    qpy.dump([final_circuit], file)
