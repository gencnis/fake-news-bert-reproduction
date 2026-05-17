import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = "stage3/outputs/stage3_results.csv"
PLOTS_DIR = Path("stage3/outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

label_map = {
    "stage2_baseline": "DistilBERT\nBaseline",
    "stage2_roberta": "RoBERTa",
    "stage2_weighted": "Aggressive\nWeighted",
    "stage3_threshold_tuning": "Threshold\nTuning",
    "stage3_mild_weighted": "Mild\nWeighted",
    "stage3_distilbert_5epoch": "DistilBERT\n5 Epoch",
    "stage3_distilbert_lr1e5": "DistilBERT\nLR 1e-5",
    "stage3_bert_base": "BERT-base",
}

df["plot_label"] = df["experiment"].map(label_map)

# Clean table for paper
table_cols = [
    "experiment",
    "model",
    "change",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
]

table_df = df[table_cols].copy()
metric_cols = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
table_df[metric_cols] = table_df[metric_cols].round(4)

table_df.to_csv("stage3/outputs/stage3_results_clean_table.csv", index=False)

print("\nClean result table:")
print(table_df.to_string(index=False))

# Plot 1: Test F1 comparison
plt.figure(figsize=(11, 5))
plt.bar(df["plot_label"], df["test_f1"])
plt.title("Test F1-score Comparison Across Experiments")
plt.xlabel("Experiment")
plt.ylabel("Test F1-score")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, max(df["test_f1"]) + 0.08)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "test_f1_comparison.png", dpi=300)
plt.close()

# Plot 2: Precision and Recall comparison
x = range(len(df))
width = 0.35

plt.figure(figsize=(11, 5))
plt.bar([i - width / 2 for i in x], df["test_precision"], width=width, label="Precision")
plt.bar([i + width / 2 for i in x], df["test_recall"], width=width, label="Recall")
plt.title("Test Precision and Recall Comparison")
plt.xlabel("Experiment")
plt.ylabel("Score")
plt.xticks(list(x), df["plot_label"], rotation=25, ha="right")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "precision_recall_comparison.png", dpi=300)
plt.close()

# Plot 3: Accuracy, Precision, Recall, F1 table-like grouped plot
metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]

plt.figure(figsize=(12, 6))
bar_width = 0.2
x = range(len(df))

for idx, metric in enumerate(metrics):
    positions = [i + (idx - 1.5) * bar_width for i in x]
    plt.bar(positions, df[metric], width=bar_width, label=metric.replace("test_", "").capitalize())

plt.title("Test Metric Comparison Across Experiments")
plt.xlabel("Experiment")
plt.ylabel("Score")
plt.xticks(list(x), df["plot_label"], rotation=25, ha="right")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "all_test_metrics_comparison.png", dpi=300)
plt.close()

print("\nSaved files:")
print("stage3/outputs/stage3_results_clean_table.csv")
print("stage3/outputs/plots/test_f1_comparison.png")
print("stage3/outputs/plots/precision_recall_comparison.png")
print("stage3/outputs/plots/all_test_metrics_comparison.png")