from datasets import load_dataset, DatasetDict

SEED = 42

dataset = load_dataset(
    "newsmediabias/NELA-GT_GPT-Labels_Dataset",
    token=True
)

ds = dataset["train"]

# Keeps only needed columns
ds = ds.select_columns(["title", "content", "gpt label", "nela label"])

# Converts labels to ClassLabel so stratified split works
ds = ds.class_encode_column("gpt label")
ds = ds.class_encode_column("nela label")

print("Features after encoding:")
print(ds.features)

# 80 / 20 split, stratified by gpt label
split_1 = ds.train_test_split(
    test_size=0.2,
    seed=SEED,
    stratify_by_column="gpt label"
)

train_ds = split_1["train"]
temp_ds = split_1["test"]

# 10 / 10 from temp, stratified
split_2 = temp_ds.train_test_split(
    test_size=0.5,
    seed=SEED,
    stratify_by_column="gpt label"
)

val_ds = split_2["train"]
test_ds = split_2["test"]

final_ds = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

print("\nFinal dataset:")
print(final_ds)

for split in final_ds:
    labels = final_ds[split]["gpt label"]
    zeros = sum(1 for x in labels if x == 0)
    ones = sum(1 for x in labels if x == 1)
    print(f"\n=== {split} ===")
    print("rows:", len(final_ds[split]))
    print("gpt label 0:", zeros)
    print("gpt label 1:", ones)

final_ds.save_to_disk("data/gpt_label_splits")
print("\nSaved to: data/gpt_label_splits")