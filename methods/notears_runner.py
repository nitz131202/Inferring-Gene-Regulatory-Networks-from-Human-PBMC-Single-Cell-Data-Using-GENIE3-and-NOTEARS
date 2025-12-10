# methods/notears_runner.py

import numpy as np
from numpy.linalg import eigvals

def _h_func(W):
    """Acyclicity constraint."""
    return np.trace(np.linalg.matrix_power(W*W, 10))

def run_notears(X, lambda1=0.1, max_iter=100, lr=0.01):
    n, d = X.shape
    W = np.zeros((d, d))

    for iteration in range(max_iter):
        grad = X.T @ (X - X @ W) / n + lambda1 * np.sign(W)
        W = W - lr * grad
        
        # Project diagonal to zero
        np.fill_diagonal(W, 0)
    return W
