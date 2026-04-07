import torch
import numpy as np


class ExpvalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, circuit, ops):
        params_np = params.detach().cpu().numpy()
        expvals = circuit.op_expval(params_np, ops)

        ctx.save_for_backward(params)
        ctx.circuit = circuit
        ctx.ops = ops

        return torch.tensor(expvals, dtype=params.dtype, device=params.device)

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        circuit = ctx.circuit
        ops = ctx.ops

        params_np = params.detach().cpu().numpy()
        shift = np.pi / 4

        grads = []

        for i in range(len(params_np)):
            plus = params_np.copy()
            minus = params_np.copy()

            plus[i] += shift
            minus[i] -= shift

            f_plus = circuit.op_expval(plus, ops)
            f_minus = circuit.op_expval(minus, ops)

            grad_i = f_plus - f_minus
            grads.append(grad_i)

        jacobian = torch.tensor(
            np.array(grads).T, dtype=params.dtype, device=grad_output.device
        )
        grad_params = grad_output @ jacobian

        return grad_params, None, None, None

    def backward_not(ctx, grad_output):
        (params,) = ctx.saved_tensors
        circuit = ctx.circuit
        ops = ctx.ops
        params_np = params.detach().numpy()

        # Because the gate is 2*theta, the shift must be scaled
        # For a gate Exp(-i * a * theta), shift is pi / (2 * a)
        shift = np.pi / 4  # Adjusted for the '2' in your cos(2*theta)

        grads = []
        for i in range(len(params_np)):
            plus = params_np.copy()
            minus = params_np.copy()
            plus[i] += shift
            minus[i] -= shift

            f_plus = circuit.op_expval(plus, ops)
            f_minus = circuit.op_expval(minus, ops)

            # The factor at the end (1.0) depends on the '2' in 2*theta
            grad_i = f_plus - f_minus
            grads.append(grad_i)

        jacobian = torch.tensor(np.array(grads).T, dtype=torch.float32)
        grad_params = grad_output @ jacobian

        return grad_params, None, None


def expvals_torch(params, circuit, ops):
    return ExpvalFunction.apply(params, circuit, ops)


def mmd_loss_torch(params, circuit, ground_truth, sigma, n_ops, ops):

    expvals = expvals_torch(params, circuit, ops)

    data_vals = 1 - 2 * ((ground_truth @ ops.T) % 2)
    tr_data = torch.tensor(
        data_vals.mean(axis=0), dtype=expvals.dtype, device=expvals.device
    )

    return torch.mean((expvals - tr_data) ** 2)
