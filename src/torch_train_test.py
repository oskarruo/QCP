import numpy as np
from iqp_to_qiskit import IqpCircuitQiskit
from utils import median_heuristic, hardware_efficient_iqp_gates
from torch_training import TrainerTorch
from torch_methods import mmd_loss_torch

n_qubits = 6
gates = hardware_efficient_iqp_gates(n_qubits)
circuit = IqpCircuitQiskit(n_qubits, gates)

X_train = np.random.binomial(1, 0.5, size=(1000, n_qubits))
X_train = X_train[X_train.sum(axis=1) < 3]

print(f"Training data: {len(X_train)} samples")

sigma = median_heuristic(X_train)

params_init = np.random.normal(0, 1 / np.sqrt(n_qubits), len(gates))

loss_kwargs = {
    "params": params_init,
    "circuit": circuit,
    "ground_truth": X_train,
    "sigma": sigma,
    "n_ops": 200,
    "shots": 5000,
}

trainer = TrainerTorch(mmd_loss_torch, lr=0.01)
trainer.train(n_iters=100, loss_kwargs=loss_kwargs)

final_params = trainer.final_params
print("Final parameters")
for gate, theta in zip(gates, final_params):
    print(f"  {gate}  →  θ = {theta:.4f} rad")
