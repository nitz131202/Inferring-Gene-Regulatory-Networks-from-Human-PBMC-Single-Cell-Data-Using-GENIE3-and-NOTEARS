# evaluate_dorothea.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ========= LOAD GRNs =========

print("\nLoading inferred GRNs...")
genie3_adj = np.load("data/pbmc/GENIE3_adj.npy")
notears_adj = np.load("data/pbmc/NOTEARS_adj.npy")
genie3_bin = np.load("data/pbmc/GENIE3_binary.npy")
notears_bin = np.load("data/pbmc/NOTEARS_binary.npy")
print("GENIE3 and NOTEARS GRNs loaded.")

# ========= LOAD PBMC GENES =========

print("Loading gene names from filtered PBMC matrix...")
df = pd.read_csv("data/pbmc/pbmc3k_filtered.csv")
genes = list(df.columns)
print(f"Loaded {len(genes)} genes.\n")

# ========= LOAD DOROTHEA =========

print("Loading DoRothEA (local database.csv)...")
dorothea = pd.read_csv("data/dorothea/database.csv")

# Keep only TF, target, effect
dorothea = dorothea[["TF", "target", "effect"]]

# Filter to pairs that are in PBMC HVG list
doro_filtered = dorothea[
    (dorothea["TF"].isin(genes)) & 
    (dorothea["target"].isin(genes))
]

print(f"Total DoRothEA interactions: {len(dorothea)}")
print(f"DoRothEA interactions in PBMC HVGs: {len(doro_filtered)}\n")

# Map gene → index
gene_to_idx = {gene: i for i, gene in enumerate(genes)}

# Build DoRothEA adjacency matrix
N = len(genes)
doro_mat = np.zeros((N, N), dtype=int)

for _, row in doro_filtered.iterrows():
    tf = row["TF"]
    tgt = row["target"]
    if tf in gene_to_idx and tgt in gene_to_idx:
        doro_mat[gene_to_idx[tf], gene_to_idx[tgt]] = 1

# ========= EVALUATION FUNCTION =========

def evaluate(pred_bin, gold_bin, name):
    TP = np.sum((pred_bin == 1) & (gold_bin == 1))
    FP = np.sum((pred_bin == 1) & (gold_bin == 0))
    FN = np.sum((pred_bin == 0) & (gold_bin == 1))

    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"\n====== {name} ======")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return TP, FP, FN, prec, rec, f1

# ========= RUN EVALUATION =========

g_TP, g_FP, g_FN, g_prec, g_rec, g_f1 = evaluate(genie3_bin, doro_mat, "GENIE3")
n_TP, n_FP, n_FN, n_prec, n_rec, n_f1 = evaluate(notears_bin, doro_mat, "NOTEARS")

print("\nEvaluation complete.")

# ========= MAKE PLOTS =========

os.makedirs("plots", exist_ok=True)

models = ["GENIE3", "NOTEARS"]

# Precision
plt.figure(figsize=(5,5))
plt.bar(models, [g_prec, n_prec])
plt.ylabel("Precision")
plt.title("DoRothEA Precision")
plt.savefig("plots/dorothea_precision.png", dpi=300)
plt.close()

# Recall
plt.figure(figsize=(5,5))
plt.bar(models, [g_rec, n_rec])
plt.ylabel("Recall")
plt.title("DoRothEA Recall")
plt.savefig("plots/dorothea_recall.png", dpi=300)
plt.close()

# F1 Score
plt.figure(figsize=(5,5))
plt.bar(models, [g_f1, n_f1])
plt.ylabel("F1 Score")
plt.title("DoRothEA F1 Score")
plt.savefig("plots/dorothea_f1.png", dpi=300)
plt.close()

# TP vs FN
import numpy as np
x = np.arange(2)
width = 0.35

plt.figure(figsize=(6,5))
plt.bar(x - width/2, [g_TP, n_TP], width, label="TP")
plt.bar(x + width/2, [g_FN, n_FN], width, label="FN")
plt.xticks(x, models)
plt.ylabel("Count")
plt.title("DoRothEA: TP vs FN")
plt.legend()
plt.savefig("plots/dorothea_tp_fn.png", dpi=300)
plt.close()

print("\nSaved plots to: plots/")
