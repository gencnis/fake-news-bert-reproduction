# Fake News Detection with BERT-like Models: Reproduction and Improvement Experiments

This repository contains reproduction and improvement experiments for fake news detection using BERT-like transformer models on GPT-labeled news data.

The study is based on the paper:

**Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data**  
Shaina Raza, Drai Paulen-Patterson, Chen Ding  
Knowledge and Information Systems, 2025  

Paper: https://arxiv.org/abs/2412.14276  
Original repository: https://github.com/draip96/FakeNewsClassification

## Project Overview

The goal of this project is to reproduce a BERT-like fake news classification pipeline and then evaluate controlled improvement strategies based on the observed limitations of the reproduced results.

The experiments focus on:

- Reproducing BERT-like model baselines
- Evaluating the effect of class imbalance
- Testing threshold tuning
- Testing weighted loss variants
- Testing hyperparameter changes
- Comparing DistilBERT, RoBERTa, and BERT-base architectures
- Analyzing precision-recall tradeoff

## Dataset

The experiments use the dataset referenced through the original authors' repository:

`newsmediabias/NELA-GT_GPT-Labels_Dataset`

The accessed dataset version contains:

- 5,414 news samples
- `title`
- `content`
- `nela label`
- `gpt label`

In this study, `gpt label` is used as the target label.

The dataset is not included in this repository. It should be accessed through Hugging Face according to the original dataset access conditions.

## Repository Structure

src/
├── data_preparation/
│   ├── check_hf_dataset.py
│   ├── inspect_dataset.py
│   ├── make_splits.py
│   └── tokenize_dataset.py
│
├── reproduction/
│   ├── check_imports.py
│   ├── train_distilbert.py
│   ├── train_roberta.py
│   ├── train_distilbert_weighted.py
│   └── compare_baselines.py
│
└── improvement/
    ├── threshold_tuning_distilbert.py
    ├── train_distilbert_mild_weighted.py
    ├── train_distilbert_5epoch.py
    ├── train_distilbert_lr1e5.py
    ├── train_bert_base.py
    └── make_stage3_plots.py

results/
├── reproduction_results.csv
├── improvement_results.csv
├── final_results_clean_table.csv
└── threshold_search_distilbert.csv

figures/
├── test_f1_comparison.png
├── precision_recall_comparison.png
└── all_test_metrics_comparison.png

notes/
└── experiment_log.txt

## Reproduction Experiments

The reproduction experiments include:

- DistilBERT baseline
- RoBERTa baseline
- DistilBERT with weighted loss

The best reproduced baseline by test F1-score was:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|
| DistilBERT | 0.8708 | 0.7302 | 0.4646 | 0.5679 |

## Improvement Experiments

Based on the reproduction results, the main observed issue was low recall on the minority class due to class imbalance. The following improvement strategies were tested:

- Threshold tuning
- Mild weighted loss
- Increasing epoch count
- Lowering learning rate
- Testing BERT-base as an alternative BERT-like architecture

Summary of final results:

| Experiment | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|
| DistilBERT baseline | 0.8708 | 0.7302 | 0.4646 | 0.5679 |
| Threshold tuning | 0.8690 | 0.7188 | 0.4646 | 0.5644 |
| Mild weighted loss | 0.8376 | 0.5534 | 0.5758 | 0.5644 |
| DistilBERT 5 epoch | 0.8598 | 0.6575 | 0.4848 | 0.5581 |
| DistilBERT LR 1e-5 | 0.8506 | 0.6071 | 0.5152 | 0.5574 |
| BERT-base | 0.8727 | 0.7679 | 0.4343 | 0.5548 |

## Key Findings

The improvement experiments did not outperform the DistilBERT baseline in terms of test F1-score. However, they showed important metric-level tradeoffs:

- BERT-base achieved the highest test accuracy and precision.
- Aggressive weighted loss achieved the highest test recall.
- Mild weighted loss improved recall more moderately while preserving a better precision-recall balance.
- Threshold tuning slightly improved validation F1-score but did not improve test F1-score.
- Increasing epochs and lowering learning rate improved recall but reduced precision.

Overall, the results suggest that the main challenge is not only model architecture, but also class imbalance and precision-recall tradeoff.

## Environment

Main environment:

- Python 3.12
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas
- matplotlib

The experiments were run on:

- NVIDIA GeForce RTX 3070 Ti Laptop GPU
- 8 GB GPU memory

Dependencies are listed in `requirements.txt`.

## Notes on Reproducibility

Large artifacts are intentionally not included in this repository:

- dataset files
- model checkpoints
- `.safetensors`
- `.bin`
- local virtual environment files

The dataset should be downloaded according to its original access rules, and model checkpoints can be regenerated using the provided training scripts.

## Main Conclusion

The study shows that controlled improvement strategies can affect different performance dimensions, especially recall and precision. However, the reproduced DistilBERT baseline remained the strongest model in terms of test F1-score.

The results suggest that class imbalance, dataset version differences, preprocessing decisions, and hardware constraints are critical factors in reproducing and improving fake news detection results.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts in the following order.

### 1. Prepare and inspect the dataset

```bash
python src/data_preparation/check_hf_dataset.py
python src/data_preparation/inspect_dataset.py
python src/data_preparation/make_splits.py
python src/data_preparation/tokenize_dataset.py
```

### 2. Run reproduction experiments

```bash
python src/reproduction/train_distilbert.py
python src/reproduction/train_roberta.py
python src/reproduction/train_distilbert_weighted.py
python src/reproduction/compare_baselines.py
```

### 3. Run improvement experiments

```bash
python src/improvement/threshold_tuning_distilbert.py
python src/improvement/train_distilbert_mild_weighted.py
python src/improvement/train_distilbert_5epoch.py
python src/improvement/train_distilbert_lr1e5.py
python src/improvement/train_bert_base.py
python src/improvement/make_stage3_plots.py
```

> Note: The dataset is not included in this repository. Access to the Hugging Face dataset may require authentication depending on the dataset permissions.

## Author

Nisanur Genc  
MSc-level Pattern Recognition project  
GitHub: @gencnis
