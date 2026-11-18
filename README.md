# AI Evaluation: Conflict Event Classification (UCDP Philippines)
**CMSC 170: Introduction to Artificial Intelligence**  
**Team: Gem in AI**  
- Carlos Miguel Alquinto  
- Nathaniel Escaño (Group Leader)  
- Earl Ian Garduce  

---

## 1. Project Overview

This project implements and evaluates three distinct machine learning algorithms from scratch (using only standard Python data structures and mathematical operations) to classify armed conflict events in the Philippines:

1. **Gaussian Naive Bayes** (`gaussian_nb.py`)
2. **Linear Perceptron (One-vs-Rest Multiclass)** (`perceptron.py`)
3. **K-Nearest Neighbors ($k=3$, Euclidean Distance)** (`knn.py`)

### Problem Formulation
Using historical conflict data from the **Uppsala Conflict Data Program (UCDP)** hosted on the Humanitarian Data Exchange (HDX), the models predict the armed group (`side_b`) involved in an event across **three classes**:
- **Class 0**: Communist Party of the Philippines / New People's Army (`CPP`)
- **Class 1**: Mindanao Regional Armed Groups (`ASG`, `MILF`, `MNLF`, `BIFF`, etc.)
- **Class 2**: Other Non-State Actors / Excluded Groups (`Civilians`, `Government of Kuwait`, military factions)

### Input Features (8 numerical features)
- **Geographic**: `longitude`, `latitude`
- **Lethality**: `deaths_a`, `deaths_b`, `deaths_civilians`, `best` (highest casualty estimate)
- **Context**: `type_of_violence` (state-based, non-state, one-sided), `duration_days` (computed from `date_start` and `date_end`)

### Dataset Partitioning
The dataset contains **4,368 total events** divided using an **80/20 train/test split**:
- **Training Set**: 3,486 samples (80%)
- **Testing Set**: 882 samples (20%)

---

## 2. Prerequisites & Environment Setup

The core models and evaluation metrics are implemented in pure Python with zero third-party machine learning libraries (no scikit-learn). Generating the comparison plots requires `matplotlib` and `numpy`.

### Requirements
- Python 3.10+
- `matplotlib`
- `numpy`

### Installation
Run the following command in your terminal to install the necessary plotting dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. How to Run and Reproduce Results

### Option A: Complete Pipeline Execution (Recommended)
To run the full preprocessing, model training, evaluation, table printing, CSV exports, and plot generation in a single command:

```bash
python main.py
```
*(or `python3 main.py` on Linux/macOS)*

This script will:
1. Load and preprocess `conflict_data_phl.csv`.
2. Split the dataset into 80% train and 20% test sets.
3. Train **Gaussian Naive Bayes**, **Linear Perceptron**, and **KNN ($k=3$)**.
4. Save model predictions to `plots/nb_predictions.csv`, `plots/perceptron_predictions.csv`, and `plots/knn_predictions.csv`.
5. Compute and print formatted comparison tables (Overall Metrics, Per-Class Breakdown, and Confusion Matrices).
6. Export metric tables to `plots/summary_metrics.csv` and `plots/per_class_metrics.csv`.
7. Generate all three 300 DPI visualization figures inside the `plots/` directory.

---

### Option B: Compare Models & Generate Visualizations from Saved Predictions
If prediction CSVs already exist and you want to recalculate metrics, view tables, and regenerate plots without retraining:

```bash
python compare_models.py
```

---

### Option C: KNN Hyperparameter Tuning (Elbow Method)
To reproduce the experimental evaluation that determined $k=3$ as the optimal number of neighbors:

```bash
python bestk.py
```
This evaluates $k \in [1, 20]$ using Macro F1-Score and displays the elbow curve.

---

## 4. Expected Results (For Reproduction Verification)

When executing `python main.py`, your terminal output should match the official evaluation numbers below:

### Table 1: Overall Model Evaluation Metrics (882 Test Samples)

| Model | Correct / Total | Standard Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes** | 553 / 882 | 62.70% | 66.65% | 74.83% | 66.65% | 58.07% |
| **Linear Perceptron (OvR)** | 739 / 882 | 83.79% | 87.63% | 86.56% | 87.63% | 86.43% |
| **K-Nearest Neighbors ($k=3$)** | **817 / 882** | **92.63%** | **93.07%** | **93.42%** | **93.07%** | **93.22%** |

### Table 2: Confusion Matrices (Raw Counts & Class Recall)

#### Gaussian Naive Bayes
```
Actual \ Predicted |  0 (CPP)  | 1 (Mindanao) | 2 (Others) | Support
-------------------|-----------|--------------|------------|--------
0 (CPP)            | 423 (98.8%) |    4 (0.9%)  |   1 (0.2%) |   428
1 (Mindanao)       | 312 (93.7%) |   12 (3.6%)  |   9 (2.7%) |   333
2 (Others)         |   3 (2.5%)  |    0 (0.0%)  | 118 (97.5%)|   121
```

#### Linear Perceptron (One-vs-Rest)
```
Actual \ Predicted |  0 (CPP)  | 1 (Mindanao) | 2 (Others) | Support
-------------------|-----------|--------------|------------|--------
0 (CPP)            | 317 (74.1%) |  109 (25.5%) |   2 (0.5%) |   428
1 (Mindanao)       |  23 (6.9%)  |  304 (91.3%) |   6 (1.8%) |   333
2 (Others)         |   3 (2.5%)  |    0 (0.0%)  | 118 (97.5%)|   121
```

#### K-Nearest Neighbors ($k=3$)
```
Actual \ Predicted |  0 (CPP)  | 1 (Mindanao) | 2 (Others) | Support
-------------------|-----------|--------------|------------|--------
0 (CPP)            | 392 (91.6%) |   33 (7.7%)  |   3 (0.7%) |   428
1 (Mindanao)       |  21 (6.3%)  |  311 (93.4%) |   1 (0.3%) |   333
2 (Others)         |   5 (4.1%)  |    2 (1.7%)  | 114 (94.2%)|   121
```

---

## 5. Output Files & Visualizations

Running `main.py` or `compare_models.py` produces the following artifacts inside the `plots/` folder:

| File Name | Description |
| :--- | :--- |
| `plots/nb_predictions.csv` | Testing set actual labels vs Gaussian Naive Bayes predictions. |
| `plots/perceptron_predictions.csv` | Testing set actual labels vs Linear Perceptron predictions. |
| `plots/knn_predictions.csv` | Testing set actual labels vs KNN ($k=3$) predictions. |
| `plots/summary_metrics.csv` | Exported Table 1 metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1). |
| `plots/per_class_metrics.csv` | Exported Table 2 per-class precision, recall, support, and F1. |
| `plots/overall_metrics_comparison.png` | **Graph 1**: Grouped bar chart comparing Macro F1 vs Balanced Accuracy vs Standard Accuracy. |
| `plots/confusion_matrices_heatmap.png` | **Graph 2**: Side-by-side $3 \times 3$ confusion matrix heatmaps. |
| `plots/per_class_breakdown.png` | **Graph 3**: Per-class recall and F1-score breakdown. |
| `plots/bestkplot.png` | Elbow curve plot showing optimal $k$ selection for KNN. |

---

## 6. Repository Architecture

```
AI-Evaluation/
├── conflict_data_phl.csv         # Main UCDP conflict dataset
├── preprocess.py                 # Data loading, date parsing & 3-class target mapping
├── split.py                      # 80/20 train/test stratified splitter
├── gaussian_nb.py                # Gaussian Naive Bayes classifier
├── perceptron.py                 # Multiclass One-vs-Rest Linear Perceptron
├── knn.py                        # K-Nearest Neighbors classifier
├── recall_precision.py           # Precision, Recall & accuracy calculation functions
├── balanced_accuracy.py          # Balanced Accuracy metric implementation
├── macrof1.py                    # Macro-averaged F1 metric implementation
├── confusion_matrix.py           # Confusion matrix calculation & detailed ASCII printer
├── bestk.py                      # Elbow method script for determining optimal k
├── compare_models.py             # Table summary, CSV exporter & Matplotlib graph generator
├── main.py                       # Master execution script
├── README.md                     # Project documentation & reproduction instructions
├── requirements.txt               # Python package dependencies
└── plots/                        # Generated figures, tables, and predictions
    ├── overall_metrics_comparison.png
    ├── confusion_matrices_heatmap.png
    ├── per_class_breakdown.png
    ├── bestkplot.png
    ├── summary_metrics.csv
    ├── per_class_metrics.csv
    ├── nb_predictions.csv
    ├── perceptron_predictions.csv
    └── knn_predictions.csv
```
