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

        return grad_params, None, None


def expvals_torch(params, circuit, ops):
    return ExpvalFunction.apply(params, circuit, ops)


def mmd_loss_torch(params, circuit, ground_truth, ops):
    expvals = expvals_torch(params, circuit, ops)

    m = len(ground_truth)
    if m < 2:
        raise ValueError("ground_truth must contain at least 2 samples")

    data_vals = 1 - 2 * ((ground_truth @ ops.T) % 2)
    tr_data = torch.tensor(
        data_vals.mean(axis=0), dtype=expvals.dtype, device=expvals.device
    )

    # Unbiased MMD² estimator.
    # expvals are computed exactly (no Monte Carlo variance), so only the data
    # squared term needs a bias correction: E[tr_data²] = E_Q[Z_s]² + (1-E_Q[Z_s]²)/m,
    # giving the unbiased estimate (m·tr_data² - 1)/(m-1) for E_Q[Z_s]².
    return torch.mean(expvals**2 - 2 * expvals * tr_data + (m * tr_data**2 - 1) / (m - 1))
