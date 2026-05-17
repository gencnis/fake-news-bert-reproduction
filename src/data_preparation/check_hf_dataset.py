from datasets import load_dataset
import os

print("HF_TOKEN exists:", bool(os.environ.get("HF_TOKEN")))

dataset = load_dataset(
    "newsmediabias/NELA-GT_GPT-Labels_Dataset",
    token=True
)

print(dataset)

for split in dataset.keys():
    print(f"\n=== {split} ===")
    print(dataset[split].column_names)
    print(dataset[split][0])