# utils/metrics.py

import numpy as np

def threshold_adj(A, thresh=0.01):
    """Turn weighted adjacency into binary adjacency."""
    return (np.abs(A) > thresh).astype(int)

def shd(A_true, A_hat):
    """Structural Hamming Distance."""
    return np.sum(A_true != A_hat)

def precision_recall_f1(A_true, A_hat):
    tp = np.sum((A_true == 1) & (A_hat == 1))
    fp = np.sum((A_true == 0) & (A_hat == 1))
    fn = np.sum((A_true == 1) & (A_hat == 0))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (fn + tp + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def orientation_accuracy(A_true, A_hat):
    """
    Among edges predicted correctly (ignoring direction),
    how many directions are also correct?
    """
    # Skeleton match
    A_true_skel = ((A_true + A_true.T) > 0).astype(int)
    A_hat_skel  = ((A_hat + A_hat.T) > 0).astype(int)

    common = np.where((A_true_skel == 1) & (A_hat_skel == 1))
    total = len(common[0])
    if total == 0: return 0

    correct = 0
    for i, j in zip(*common):
        if A_true[i, j] == 1 and A_hat[i, j] == 1:
            correct += 1
    return correct / total
