import numpy as np


def median_heuristic(X):
    """
    Computes an estimate of the median heuristic used to decide the bandwidth of the RBF kernels; see
    https://arxiv.org/abs/1707.07269
    :param X (array): Dataset of interest
    :return (float): median heuristic estimate
    """
    m = len(X)
    X = np.array(X)
    med = np.median(
        [np.sqrt(np.sum((X[i] - X[j]) ** 2)) for i in range(m) for j in range(m)]
    )
    return med


def hardware_efficient_iqp_gates(n_qubits):
    """
    Nearest-neighbour IQP ansatz.

    Returns:
        gates: list of gates in IQP format [[[i,j]], ...]
    """
    gates = []

    # Single-qubit Z terms
    for i in range(n_qubits):
        gates.append([[i]])

    # Nearest-neighbour ZZ interactions
    for i in range(n_qubits - 1):
        gates.append([[i, i + 1]])

    return gates
