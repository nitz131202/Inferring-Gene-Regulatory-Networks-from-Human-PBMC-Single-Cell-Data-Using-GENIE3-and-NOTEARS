import scanpy as sc

print("Downloading PBMC 3k dataset...")
adata = sc.datasets.pbmc3k()
adata.write("data/pbmc/pbmc3k_raw.h5ad")
print("Saved: data/pbmc/pbmc3k_raw.h5ad")
