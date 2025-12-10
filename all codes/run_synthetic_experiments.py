# run_synthetic_experiments.py

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.dag_utils import (
    generate_er_dag,
    generate_scale_free_dag,
    dag_to_adjacency
)

from utils.sem_utils import (
    sample_linear_sem,
    sample_nonlinear_sem
)

from utils.metrics import (
    threshold_adj,
    shd, precision_recall_f1, orientation_accuracy
)

from methods.genie3_runner import run_genie3
from methods.notears_runner import run_notears


# ------------------------------
# Experiment Parameters
# ------------------------------
p = 50              # nodes
n_samples = 500
noise = 1.0
thresh = 0.05

results = []


# ------------------------------
# Main experiment loop
# ------------------------------
for graph_type in ["ER", "SF"]:
    if graph_type == "ER":
        dag = generate_er_dag(p, edge_prob=0.1, seed=0)
    else:
        dag = generate_scale_free_dag(p, seed=0)

    B = dag_to_adjacency(dag)

    for sem in ["linear", "nonlinear"]:
        if sem == "linear":
            X = sample_linear_sem(B, n_samples=n_samples, noise_scale=noise)
        else:
            X = sample_nonlinear_sem(B, n_samples=n_samples, noise_scale=noise)

        # ===========================
        # GENIE3
        # ===========================
        A_genie = run_genie3(X, n_trees=500)
        A_genie_bin = threshold_adj(A_genie, thresh)

        s = shd(B, A_genie_bin)
        pr, re, f1 = precision_recall_f1(B, A_genie_bin)
        oa = orientation_accuracy(B, A_genie_bin)

        results.append(["GENIE3", graph_type, sem, s, pr, re, f1, oa])

        # ===========================
        # NOTEARS
        # ===========================
        A_notears = run_notears(X, lambda1=0.1)
        A_notears_bin = threshold_adj(A_notears, thresh)

        s = shd(B, A_notears_bin)
        pr, re, f1 = precision_recall_f1(B, A_notears_bin)
        oa = orientation_accuracy(B, A_notears_bin)

        results.append(["NOTEARS", graph_type, sem, s, pr, re, f1, oa])


# ------------------------------
# Save results
# ------------------------------
df = pd.DataFrame(results, columns=[
    "Method", "Graph", "SEM",
    "SHD", "Precision", "Recall", "F1", "OrientAcc"
])

df.to_csv("data/synthetic/synthetic_results.csv", index=False)
print(df)
