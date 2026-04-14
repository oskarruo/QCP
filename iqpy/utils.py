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


def generate_structured_dataset(n_qubits, n_samples, rng=None):
    """
    Generates a structured dataset which is a mixture of two clusters:
    Cluster 1 is mostly 0s with a few 1s
    Cluster 2 is mostly alternating 0s and 1s with some noise
    """
    if rng is None:
        rng = np.random.default_rng(0)

    samples = []
    while len(samples) < n_samples:
        batch = n_samples * 3

        cluster1 = rng.binomial(1, 0.15, size=(batch, n_qubits))
        cluster1 = cluster1[cluster1.sum(axis=1) <= 2]

        base = np.array([i % 2 for i in range(n_qubits)])
        cluster2 = np.tile(base, (batch, 1))
        noise = rng.binomial(1, 0.1, size=(batch, n_qubits))
        cluster2 = np.abs(cluster2 - noise)

        combined = np.vstack([cluster1, cluster2])
        rng.shuffle(combined)
        samples.extend(combined.tolist())

    return np.array(samples[:n_samples])


def generate_simple_dataset(n_qubits, n_samples, rng=None):
    """
    Generates a simple dataset with few 1s.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    data = np.zeros((n_samples, n_qubits), dtype=np.int8)

    for i in range(n_samples):
        k = rng.integers(0, 3)

        if k > 0:
            idx = rng.choice(n_qubits, size=k, replace=False)
            data[i, idx] = 1

    return data


def generate_experiment_mixture_dataset(n_qubits, n_samples, rng=None):
    """
    Samples x in {0,1}^n from the experiment distribution D:
    - With probability 1/3: x = 0^n
    - With probability 1/3: each bit is i.i.d. Bernoulli(0.4)
    - With probability 1/3: each bit is i.i.d. Bernoulli(0.8)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    component = rng.integers(0, 3, size=n_samples)
    data = np.empty((n_samples, n_qubits), dtype=np.int8)

    mask0 = component == 0
    data[mask0] = 0

    mask1 = component == 1
    if np.any(mask1):
        data[mask1] = rng.binomial(1, 0.4, size=(mask1.sum(), n_qubits))

    mask2 = component == 2
    if np.any(mask2):
        data[mask2] = rng.binomial(1, 0.8, size=(mask2.sum(), n_qubits))

    return data
