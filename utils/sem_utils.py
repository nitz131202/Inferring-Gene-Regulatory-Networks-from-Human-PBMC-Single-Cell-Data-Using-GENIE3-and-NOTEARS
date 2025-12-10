# utils/sem_utils.py

import numpy as np
import networkx as nx

def sample_linear_sem(B, n_samples=500, noise_scale=1.0, seed=None):
    """
    Linear SEM:
        X_j = sum_i( B[i,j] * X_i ) + noise
    Assumes B encodes direction i -> j in B[i,j].
    """
    rng = np.random.default_rng(seed)
    p = B.shape[0]
    X = np.zeros((n_samples, p))

    dag = nx.DiGraph(B)
    topo = list(nx.topological_sort(dag))

    for k in range(n_samples):
        eps = rng.normal(scale=noise_scale, size=p)
        for j in topo:
            parents = np.where(B[:, j] != 0)[0]
            if len(parents) > 0:
                X[k, j] = X[k, parents].dot(B[parents, j]) + eps[j]
            else:
                X[k, j] = eps[j]
    return X


def sample_nonlinear_sem(B, n_samples=500, noise_scale=1.0, seed=None):
    """
    Nonlinear SEM:
        X_j = tanh( sum_i( w_ij * X_i ) ) + noise
    """
    rng = np.random.default_rng(seed)
    p = B.shape[0]
    X = np.zeros((n_samples, p))

    dag = nx.DiGraph(B)
    topo = list(nx.topological_sort(dag))

    # Random weights for nonlinear SEM
    W = B * rng.normal(0, 1, size=B.shape)

    for k in range(n_samples):
        eps = rng.normal(scale=noise_scale, size=p)
        for j in topo:
            parents = np.where(B[:, j] != 0)[0]
            if len(parents) > 0:
                X[k, j] = np.tanh(X[k, parents].dot(W[parents, j])) + eps[j]
            else:
                X[k, j] = eps[j]
    return X
