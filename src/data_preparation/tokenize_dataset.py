from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512

dataset = load_from_disk("data/gpt_label_splits")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    text = (example["title"] or "") + " [SEP] " + (example["content"] or "")
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    encoded["labels"] = example["gpt label"]
    return encoded

tokenized = dataset.map(preprocess, batched=False)

# Keeps only what Trainer needs
tokenized = tokenized.remove_columns(["title", "content", "gpt label", "nela label"])
tokenized.set_format("torch")

print(tokenized)
print("\nTrain sample keys:", tokenized["train"][0].keys())
print("Train sample input_ids shape:", len(tokenized["train"][0]["input_ids"]))
print("Train sample label:", tokenized["train"][0]["labels"])

tokenized.save_to_disk("data/gpt_label_tokenized")
print("\nSaved to: data/gpt_label_tokenized")