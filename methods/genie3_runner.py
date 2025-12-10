# methods/genie3_runner.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm   # progress bar

def run_genie3(X, n_trees=500, seed=None, max_depth=None):
    """
    Pure Python GENIE3 reimplementation using RandomForestRegressor.
    X: numpy array (n_samples, n_genes)
    Returns adjacency matrix A where A[i, j] = importance of gene i -> j
    """
    n_samples, p = X.shape
    A = np.zeros((p, p))

    print(f"Running GENIE3 on {p} genes with {n_trees} trees...")
    print("Progress:")

    # tqdm adds a live progress bar with ETA
    for j in tqdm(range(p), desc="Fitting models", ncols=100):
        # Predict gene j using all other genes
        X_other = np.delete(X, j, axis=1)
        y = X[:, j]

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_other, y)

        # Feature importances correspond to predictors -> target j
        importances = rf.feature_importances_

        # Insert back into p-dimensional space
        full_imp = np.zeros(p)
        full_imp[np.arange(p) != j] = importances

        A[:, j] = full_imp

    print("GENIE3 finished.")
    return A
