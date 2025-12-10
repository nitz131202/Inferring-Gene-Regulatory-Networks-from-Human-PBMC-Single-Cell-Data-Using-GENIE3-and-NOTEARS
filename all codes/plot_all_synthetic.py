# plot_all_synthetic.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
df = pd.read_csv("/Users/sreenithyam/Desktop/BMI 775_Final Project/data/synthetic/synthetic_results.csv")

# Better labels
df["SEM"] = df["SEM"].map({"linear": "Linear SEM", "nonlinear": "Nonlinear SEM"})

sns.set(style="whitegrid", font_scale=1.3)

# -------------------------------------------------------------
# SHD Plot
# -------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=df,
    x="SEM",
    y="SHD",
    hue="Method"
)
plt.title("Structural Hamming Distance (SHD)")
plt.ylabel("SHD")
plt.xlabel("SEM Type")
plt.tight_layout()
plt.savefig("data/synthetic/plot_shd.png")
plt.show()

# -------------------------------------------------------------
# F1 Score Plot
# -------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=df,
    x="SEM",
    y="F1",
    hue="Method"
)
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xlabel("SEM Type")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("data/synthetic/plot_f1.png")
plt.show()

# -------------------------------------------------------------
# Precision & Recall Side-by-Side
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14,6))

sns.barplot(
    data=df,
    x="SEM",
    y="Precision",
    hue="Method",
    ax=axes[0]
)
axes[0].set_title("Precision")
axes[0].set_ylim(0, 1)

sns.barplot(
    data=df,
    x="SEM",
    y="Recall",
    hue="Method",
    ax=axes[1]
)
axes[1].set_title("Recall")
axes[1].set_ylim(0, 1)

for ax in axes:
    ax.set_xlabel("SEM Type")
    ax.legend(title="Method")

plt.tight_layout()
plt.savefig("data/synthetic/plot_precision_recall.png")
plt.show()

# -------------------------------------------------------------
# Orientation Accuracy Plot
# -------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=df,
    x="SEM",
    y="OrientAcc",
    hue="Method"
)
plt.title("Orientation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("SEM Type")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("data/synthetic/plot_orientation.png")
plt.show()

# -------------------------------------------------------------
# F1 by Graph Type (ER vs SF)
# -------------------------------------------------------------
g = sns.catplot(
    data=df,
    x="SEM",
    y="F1",
    hue="Method",
    col="Graph",
    kind="bar",
    height=5,
    aspect=1
)
g.fig.suptitle("F1 Scores by Graph Type (ER vs SF)", y=1.05)
plt.savefig("data/synthetic/plot_f1_by_graph.png")
plt.show()

print("All plots saved to data/synthetic/")
