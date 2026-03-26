import torch
import numpy as np


class ExpvalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, circuit, ops, shots):
        params_np = params.detach().numpy()
        expvals = circuit.op_expval(params_np, ops, shots=shots)

        ctx.save_for_backward(params)
        ctx.circuit = circuit
        ctx.ops = ops
        ctx.shots = shots

        return torch.tensor(expvals, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        circuit = ctx.circuit
        ops = ctx.ops
        shots = ctx.shots

        params_np = params.detach().numpy()
        shift = np.pi / 2

        grads = []

        for i in range(len(params_np)):
            plus = params_np.copy()
            minus = params_np.copy()

            plus[i] += shift
            minus[i] -= shift

            f_plus = circuit.op_expval(plus, ops, shots=shots)
            f_minus = circuit.op_expval(minus, ops, shots=shots)

            grad_i = (f_plus - f_minus) / 2.0
            grads.append(grad_i)

        jacobian = torch.tensor(np.array(grads).T, dtype=torch.float32)
        grad_params = grad_output @ jacobian

        return grad_params, None, None, None


def expvals_torch(params, circuit, ops, shots):
    return ExpvalFunction.apply(params, circuit, ops, shots)


def mmd_loss_torch(params, circuit, ground_truth, sigma, n_ops, shots):
    n_qubits = circuit.n_qubits

    p = (1 - np.exp(-1 / (2 * sigma**2))) / 2
    ops = np.random.binomial(1, p, size=(n_ops, n_qubits))

    expvals = expvals_torch(params, circuit, ops, shots)

    data_vals = 1 - 2 * ((ground_truth @ ops.T) % 2)
    tr_data = torch.tensor(data_vals.mean(axis=0), dtype=torch.float32)

    return torch.mean((expvals - tr_data) ** 2)
