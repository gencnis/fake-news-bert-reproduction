from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/gpt_label_tokenized"
OUTPUT_DIR = "stage3/outputs/distilbert_mild_weighted"

dataset = load_from_disk(DATA_PATH)

if "token_type_ids" in dataset["train"].column_names:
    dataset = dataset.remove_columns(["token_type_ids"])

labels = dataset["train"]["labels"]
class_counts = np.bincount(labels)
total = sum(class_counts)

# Original inverse-frequency weights
raw_weights = np.array([total / c for c in class_counts], dtype=np.float32)

# Mild weights: square-root of inverse-frequency weights
mild_weights = np.sqrt(raw_weights)
class_weights = torch.tensor(mild_weights, dtype=torch.float)

print("Class counts:", class_counts)
print("Raw weights:", raw_weights)
print("Mild sqrt weights:", class_weights)

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
class MildWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

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
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
)

trainer = MildWeightedTrainer(
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
    "experiment": "stage3_mild_weighted",
    "model": "DistilBERT",
    "change": "mild weighted loss sqrt class weights",
    "target_label": "gpt label",
    "val_accuracy": val_results["eval_accuracy"],
    "val_precision": val_results["eval_precision"],
    "val_recall": val_results["eval_recall"],
    "val_f1": val_results["eval_f1"],
    "test_accuracy": test_results["eval_accuracy"],
    "test_precision": test_results["eval_precision"],
    "test_recall": test_results["eval_recall"],
    "test_f1": test_results["eval_f1"],
    "notes": "Mild sqrt class weights to reduce aggressive recall-precision tradeoff"
}])

summary.to_csv("stage3/outputs/mild_weighted_summary.csv", index=False)
trainer.save_model(f"{OUTPUT_DIR}/best_model")

print("\nSaved:")
print("stage3/outputs/mild_weighted_summary.csv")
print(f"{OUTPUT_DIR}/best_model")