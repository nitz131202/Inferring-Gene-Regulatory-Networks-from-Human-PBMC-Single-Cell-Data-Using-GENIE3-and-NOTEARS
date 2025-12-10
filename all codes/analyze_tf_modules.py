# analyze_tf_modules.py

import numpy as np
import pandas as pd


# 1. Load gene names
print("Loading PBMC HVG matrix (gene names)...")
genes = pd.read_csv("data/pbmc/pbmc3k_filtered.csv", nrows=1).columns.tolist()

print(f"Total genes in matrix: {len(genes)}")


# 2. Load GENIE3 adjacency (weighted)
print("\nLoading GENIE3 adjacency...")
A = np.load("data/pbmc/GENIE3_adj.npy")

print("GENIE3 adjacency shape:", A.shape)


# 3. Define TF families of interest
stat_family = ["STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6"]
irf_family  = ["IRF1", "IRF3", "IRF7", "IRF8"]
nfkb_family = ["NFKB1", "NFKB2", "RELA", "RELB"]

modules = {
    "STAT": stat_family,
    "IRF": irf_family,
    "NFKB": nfkb_family
}

# Map gene names to indices
gene_to_idx = {g: i for i, g in enumerate(genes)}


# 4. Analyze each TF family
for module_name, tf_list in modules.items():
    print(f"\n===== {module_name} FAMILY =====")

    for tf in tf_list:
        if tf not in gene_to_idx:
            print(f"⚠ {tf}: NOT present in the 500 HVG list.")
            continue

        idx = gene_to_idx[tf]

        # All genes j where A[idx, j] > 0
        target_indices = np.where(A[idx, :] > 0)[0]
        targets = [genes[i] for i in target_indices]

        print(f"\nTF: {tf}")
        print(f" - Number of predicted targets: {len(targets)}")
        print(f" - Top 10 targets: {targets[:10]}")

print("\nTF module analysis complete.")
