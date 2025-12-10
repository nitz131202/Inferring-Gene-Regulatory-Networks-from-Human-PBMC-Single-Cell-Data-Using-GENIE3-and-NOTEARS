# Inferring Gene Regulatory Networks from Human PBMC Single-Cell Data Using GENIE3 and NOTEARS

This repository contains the code and analyses for my BMI 775 final project, where I compare two structure-learning approaches for gene regulatory network (GRN) inference:

- **GENIE3** — tree-based dependency network using Random Forests  
- **NOTEARS** — continuous optimization for learning DAG structures  

I evaluate both methods on:
- Synthetic graphs (Erdős–Rényi and scale-free, linear and nonlinear SEMs), and  
- Real **PBMC 3k** single-cell RNA-seq data, with biological validation using **TRRUST** and **DoRothEA**.

---

## Repository Structure

```text
.
├── download_pbmc3k.py          # Download PBMC 3k dataset (10x)
├── preprocess_pbmc3k.py        # Filtering, normalization, HVG selection, save PBMC matrix
├── run_synthetic_experiments.py# Synthetic benchmarks: generate graphs, simulate data, run GENIE3 & NOTEARS
├── plot_all_synthetic.py       # Generate synthetic plots: F1, SHD, precision/recall, orientation accuracy
├── infer_grn_human.py          # Run GENIE3 + NOTEARS on PBMC 3k HVG matrix, save adjacency + binary matrices
├── evaluate_trrust.py          # Evaluate GRNs against TRRUST v2 (TF–target interactions)
├── evaluate_dorothea.py        # Evaluate GRNs against DoRothEA regulons
├── analyze_tf_modules.py       # Inspect TF modules (STAT, IRF, NF-κB) in PBMC GRN
├── bar_plots.py                # Make bar plots for TRRUST/DoRothEA evaluation metrics
├── methods/
│   ├── genie3_runner.py        # Pure Python implementation of GENIE3 with RandomForestRegressor
│   ├── notears_runner.py       # Wrapper for running NOTEARS on data
│   └── grandag_runner.py       # Placeholder / future work for GraN-DAG (not used in final pipeline)
├── utils/
│   ├── load_dorothea.py        # (If present) helpers to load DoRothEA or TF lists
│   └── other utility scripts   # plotting, helper functions
├── data/
│   ├── pbmc/                   # PBMC raw and filtered matrices, adjacency .npy files
│   └── dorothea/               # Local DoRothEA database.csv (not committed if large)
└── results/                    # Plots and metrics for synthetic + real data (optional, can be regenerated)
