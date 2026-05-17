from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/gpt_label_tokenized"
OUTPUT_DIR = "stage3/outputs/distilbert_lr1e5"

dataset = load_from_disk(DATA_PATH)

if "token_type_ids" in dataset["train"].column_names:
    dataset = dataset.remove_columns(["token_type_ids"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nValidation results:")
val_results = trainer.evaluate(dataset["validation"])
print(val_results)

print("\nTest results:")
test_results = trainer.evaluate(dataset["test"])
print(test_results)

summary = pd.DataFrame([{
    "experiment": "stage3_distilbert_lr1e5",
    "model": "DistilBERT",
    "change": "learning rate reduced from 2e-5 to 1e-5",
    "target_label": "gpt label",
    "val_accuracy": val_results["eval_accuracy"],
    "val_precision": val_results["eval_precision"],
    "val_recall": val_results["eval_recall"],
    "val_f1": val_results["eval_f1"],
    "test_accuracy": test_results["eval_accuracy"],
    "test_precision": test_results["eval_precision"],
    "test_recall": test_results["eval_recall"],
    "test_f1": test_results["eval_f1"],
    "notes": "Lower learning rate tested for more stable fine-tuning"
}])

summary.to_csv("stage3/outputs/distilbert_lr1e5_summary.csv", index=False)
trainer.save_model(f"{OUTPUT_DIR}/best_model")

print("\nSaved:")
print("stage3/outputs/distilbert_lr1e5_summary.csv")
print(f"{OUTPUT_DIR}/best_model")