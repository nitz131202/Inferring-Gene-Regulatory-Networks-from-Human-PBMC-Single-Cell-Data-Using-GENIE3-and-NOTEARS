import numpy as np
import pandas as pd

print("Loading inferred GRNs...")

# Load inferred adjacency matrices
A_genie = np.load("data/pbmc/GENIE3_adj.npy")
A_notears = np.load("data/pbmc/NOTEARS_adj.npy")

print(f"GENIE3 adjacency shape: {A_genie.shape}")
print(f"NOTEARS adjacency shape: {A_notears.shape}")


# Load gene names from PBMC filtered matrix
print("\nExtracting gene names from filtered PBMC matrix...")

df = pd.read_csv("data/pbmc/pbmc3k_filtered.csv", nrows=2)
genes = list(df.columns)

print(f"Extracted {len(genes)} genes.")

# Load TRRUST database


print("\nLoading TRRUST...")

trrust = pd.read_csv(
    "data/trtrust/trrust_rawdata.human.tsv",
    sep="\t",
    header=None,
    names=["TF", "Target", "Mode", "PMID"]
)

print(f"Total TRRUST edges: {len(trrust)}")

# Filter TRRUST edges to PBMC genes
gene_set = set(genes)
trrust_filtered = trrust[
    trrust["TF"].isin(gene_set) &
    trrust["Target"].isin(gene_set)
]

print(f"TRRUST edges in PBMC gene set: {len(trrust_filtered)}")


#  TRRUST adjacency matrix


p = len(genes)
gene_to_idx = {g: i for i, g in enumerate(genes)}
T = np.zeros((p, p))

for _, row in trrust_filtered.iterrows():
    i = gene_to_idx[row["TF"]]
    j = gene_to_idx[row["Target"]]
    T[i, j] = 1

print("Built TRRUST matrix.")

# Evaluation

def evaluate(A, T, name):
    A_bin = (A > 0).astype(int)

    tp = np.sum((A_bin == 1) & (T == 1))
    fp = np.sum((A_bin == 1) & (T == 0))
    fn = np.sum((A_bin == 0) & (T == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    print(f"\n====== {name} ======")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# Run evaluation
evaluate(A_genie, T, "GENIE3")
evaluate(A_notears, T, "NOTEARS")

print("\nValidation complete.")
