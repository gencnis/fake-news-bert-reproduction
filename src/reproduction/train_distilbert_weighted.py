from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import torch
from torch import nn

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/gpt_label_tokenized"
OUTPUT_DIR = "outputs/distilbert_weighted"

dataset = load_from_disk(DATA_PATH)

# removes token_type_ids if exists
if "token_type_ids" in dataset["train"].column_names:
    dataset = dataset.remove_columns(["token_type_ids"])

# Computes class weights
labels = dataset["train"]["labels"]
class_counts = np.bincount(labels)
total = sum(class_counts)

weights = [total / c for c in class_counts]
class_weights = torch.tensor(weights, dtype=torch.float)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

# Metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": recall_metric.compute(predictions=preds, references=labels, average="binary")["recall"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
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
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nValidation results:")
print(trainer.evaluate(dataset["validation"]))

print("\nTest results:")
print(trainer.evaluate(dataset["test"]))