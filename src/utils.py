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


def median_heuristic_fast(X, n_samples=500):
    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sub = X[idx]

    dists = np.linalg.norm(X_sub[:, None, :] - X_sub[None, :, :], axis=-1)
    return np.median(dists)
