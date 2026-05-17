from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import torch

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/gpt_label_tokenized"
OUTPUT_DIR = "outputs/distilbert_gptlabel"

# Loads tokenized data
dataset = load_from_disk(DATA_PATH)

# DistilBERT does not use token_type_ids
if "token_type_ids" in dataset["train"].column_names:
    dataset = dataset.remove_columns(["token_type_ids"])

# Metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=labels, average="binary")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="binary")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Training args
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

trainer.save_model(f"{OUTPUT_DIR}/best_model")
print(f"\nModel saved to: {OUTPUT_DIR}/best_model")