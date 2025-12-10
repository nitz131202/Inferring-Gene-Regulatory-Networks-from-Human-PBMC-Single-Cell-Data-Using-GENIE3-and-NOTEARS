import scanpy as sc
import pandas as pd
import numpy as np

# Load raw PBMC 3k
adata = sc.read("data/pbmc/pbmc3k_raw.h5ad")

# Basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)

# Normalize total counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=500)
adata = adata[:, adata.var['highly_variable']]

# Convert sparse matrix to dense matrix safely
if hasattr(adata.X, "toarray"):
    X = adata.X.toarray()
else:
    X = np.array(adata.X)

# Save to CSV
df = pd.DataFrame(X, columns=adata.var_names)
df.to_csv("data/pbmc/pbmc3k_filtered.csv", index=False)

print("Saved filtered PBMC matrix:", df.shape)
