import pandas as pd
import numpy as np

from methods.genie3_runner import run_genie3
from methods.notears_runner import run_notears

# Load expression data
df = pd.read_csv("data/pbmc/pbmc3k_filtered.csv")
X = df.values
genes = df.columns


# GENIE3
print("Running GENIE3...")
A_genie = run_genie3(X, n_trees=50)
np.save("data/pbmc/GENIE3_adj.npy", A_genie)

# Threshold: top 5% edges
thr = np.percentile(A_genie[A_genie > 0], 95)
A_genie_bin = (A_genie >= thr).astype(int)
np.save("data/pbmc/GENIE3_binary.npy", A_genie_bin)

print("GENIE3 Finished:")
print("Adjacency shape:", A_genie.shape)
print("Binary edges:", A_genie_bin.sum())



# NOTEARS
print("Running NOTEARS...")
A_notears = run_notears(X, lambda1=0.1)
np.save("data/pbmc/NOTEARS_adj.npy", A_notears)

thr2 = np.percentile(np.abs(A_notears[A_notears != 0]), 95)
A_notears_bin = (np.abs(A_notears) >= thr2).astype(int)
np.save("data/pbmc/NOTEARS_binary.npy", A_notears_bin)

print("NOTEARS Finished:")
print("Adjacency shape:", A_notears.shape)
print("Binary edges:", A_notears_bin.sum())

print("Inference Complete.")
