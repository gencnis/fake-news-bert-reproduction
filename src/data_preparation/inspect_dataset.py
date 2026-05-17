from datasets import load_dataset
import os
import pandas as pd

dataset = load_dataset(
    "newsmediabias/NELA-GT_GPT-Labels_Dataset",
    token=True
)

train_ds = dataset["train"]

print("HF_TOKEN exists:", bool(os.environ.get("HF_TOKEN")))
print("Number of rows:", len(train_ds))
print("Columns:", train_ds.column_names)

df = train_ds.to_pandas()

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution - nela label:")
print(df["nela label"].value_counts(dropna=False))

print("\nLabel distribution - gpt label:")
print(df["gpt label"].value_counts(dropna=False))

print("\nContent length stats:")
content_lengths = df["content"].astype(str).apply(len)
print(content_lengths.describe())