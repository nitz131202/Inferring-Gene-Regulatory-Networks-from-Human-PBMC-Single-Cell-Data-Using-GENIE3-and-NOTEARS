import matplotlib.pyplot as plt
import numpy as np

# TRRUST evaluation results
methods = ["GENIE3", "NOTEARS"]

tp = np.array([2, 0])
fp = np.array([193265, 7569])
fn = np.array([1, 3])

# Precision, Recall, F1 from earlier
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = (2 * precision * recall) / (precision + recall + 1e-8)


# BAR PLOT 1 — TP, FN

plt.figure(figsize=(7, 5))
plt.bar(methods, tp, label="True Positives", color="green")
plt.bar(methods, fn, bottom=tp, label="False Negatives", color="red")
plt.title("TRRUST — True vs False Negatives")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("trrust_tp_fn.png")
plt.close()


# BAR PLOT 2 — Precision

plt.figure(figsize=(7, 5))
plt.bar(methods, precision, color="purple")
plt.title("TRRUST — Precision")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("trrust_precision.png")
plt.close()


# BAR PLOT 3 — Recall

plt.figure(figsize=(7, 5))
plt.bar(methods, recall, color="orange")
plt.title("TRRUST — Recall")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("trrust_recall.png")
plt.close()


# BAR PLOT 4 — F1

plt.figure(figsize=(7, 5))
plt.bar(methods, f1, color="blue")
plt.title("TRRUST — F1 Score")
plt.ylabel("F1")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("trrust_f1.png")
plt.close()

print("Saved plots:")
print(" - trrust_tp_fn.png")
print(" - trrust_precision.png")
print(" - trrust_recall.png")
print(" - trrust_f1.png")
