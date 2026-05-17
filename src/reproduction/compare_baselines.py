import pandas as pd

results = [
    {
        "model": "DistilBERT",
        "target_label": "gpt label",
        "val_accuracy": 0.8651,
        "val_precision": 0.7119,
        "val_recall": 0.4286,
        "val_f1": 0.5350,
        "test_accuracy": 0.8708,
        "test_precision": 0.7302,
        "test_recall": 0.4646,
        "test_f1": 0.5679,
    },
    {
        "model": "RoBERTa",
        "target_label": "gpt label",
        "val_accuracy": 0.8484,
        "val_precision": 0.7222,
        "val_recall": 0.2653,
        "val_f1": 0.3881,
        "test_accuracy": 0.8598,
        "test_precision": 0.7949,
        "test_recall": 0.3131,
        "test_f1": 0.4493,
    },
    {
	"model": "DistilBERT_weighted",
	"target_label": "gpt label",
	"val_accuracy": 0.8429,
	"val_precision": 0.5596,
	"val_recall": 0.6224,
	"val_f1": 0.5894,
	"test_accuracy": 0.8173,
	"test_precision": 0.5000,
	"test_recall": 0.6465,
	"test_f1": 0.5639,
},
]

df = pd.DataFrame(results)

print("\nBaseline Comparison Table:")
print(df.to_string(index=False))

df.to_csv("outputs/baseline_comparison.csv", index=False)
print("\nSaved to: outputs/baseline_comparison.csv")