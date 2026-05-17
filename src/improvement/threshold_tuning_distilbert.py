from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import pandas as pd

MODEL_PATH = "outputs/distilbert_gptlabel/best_model"
DATA_PATH = "data/gpt_label_tokenized"

dataset = load_from_disk(DATA_PATH)

if "token_type_ids" in dataset["train"].column_names:
    dataset = dataset.remove_columns(["token_type_ids"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

args = TrainingArguments(
    output_dir="stage3/outputs/threshold_tmp",
    per_device_eval_batch_size=8,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args
)

def get_probabilities(split_name):
    predictions = trainer.predict(dataset[split_name])
    logits = predictions.predictions
    labels = predictions.label_ids

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    positive_probs = probs[:, 1]

    return positive_probs, labels

def evaluate_with_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )

    return accuracy, precision, recall, f1

# Validation threshold search
val_probs, val_labels = get_probabilities("validation")

thresholds = np.arange(0.10, 0.91, 0.01)
rows = []

for threshold in thresholds:
    accuracy, precision, recall, f1 = evaluate_with_threshold(
        val_probs,
        val_labels,
        threshold
    )

    rows.append({
        "threshold": round(float(threshold), 2),
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    })

df = pd.DataFrame(rows)
best_row = df.sort_values("val_f1", ascending=False).iloc[0]
best_threshold = float(best_row["threshold"])

print("\nBest validation threshold:")
print(best_row)

# Test evaluation with best threshold
test_probs, test_labels = get_probabilities("test")
test_accuracy, test_precision, test_recall, test_f1 = evaluate_with_threshold(
    test_probs,
    test_labels,
    best_threshold
)

print("\nTest results with best threshold:")
print("threshold:", best_threshold)
print("test_accuracy:", test_accuracy)
print("test_precision:", test_precision)
print("test_recall:", test_recall)
print("test_f1:", test_f1)

# Save threshold search table
df.to_csv("stage3/outputs/threshold_search_distilbert.csv", index=False)

summary = pd.DataFrame([{
    "experiment": "stage3_threshold_tuning",
    "model": "DistilBERT",
    "change": f"threshold tuning, threshold={best_threshold}",
    "target_label": "gpt label",
    "val_accuracy": best_row["val_accuracy"],
    "val_precision": best_row["val_precision"],
    "val_recall": best_row["val_recall"],
    "val_f1": best_row["val_f1"],
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "notes": "Threshold selected on validation set by best F1"
}])

summary.to_csv("stage3/outputs/threshold_tuning_summary.csv", index=False)

print("\nSaved:")
print("stage3/outputs/threshold_search_distilbert.csv")
print("stage3/outputs/threshold_tuning_summary.csv")