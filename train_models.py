"""Train and serialize the 3 conflict classification models for instant demo startup.

Run this script once before presenting:
    python3 train_models.py

It uses the exact preprocessing and SEED=67 stratified 80/20 split from the
core project without modifying any existing model files, and saves the trained
models, held-out test split, human-readable UCDP event metadata, and verified
evaluation metrics into models/trained_models.pkl.
"""

import os
import pickle
import time

import balanced_accuracy
import macrof1
import preprocess
import recall_precision
import split
from gaussian_nb import NB_Classifier
from knn import KNN
from perceptron import Perceptron

CSV_PATH = "conflict_data_phl.csv"
OUTPUT_PATH = os.path.join("models", "trained_models.pkl")

# Raw CSV column indices for human-readable demo context
COL_ID = 0
COL_YEAR = 2
COL_SIDE_B = 17
COL_HEADLINE = 22
COL_WHERE_DESC = 26
COL_ADM1 = 27
COL_ADM2 = 28
COL_DATE_START = 39
COL_DATE_END = 40


def extract_metadata(csv_path):
    """Extract human-readable metadata for each data row in conflict_data_phl.csv."""
    raw_rows = preprocess.load_data(csv_path)[1:]  # skip header
    metadata = []
    for row in raw_rows:
        where_parts = []
        for col in (COL_WHERE_DESC, COL_ADM2, COL_ADM1):
            val = row[col].strip() if col < len(row) else ""
            if val and val not in where_parts:
                where_parts.append(val)
        location = ", ".join(where_parts[:2]) if where_parts else "Unknown location"
        province = row[COL_ADM1].strip() if COL_ADM1 < len(row) else ""
        headline = row[COL_HEADLINE].strip() if COL_HEADLINE < len(row) else ""
        date_start = row[COL_DATE_START][:10] if COL_DATE_START < len(row) else ""
        date_end = row[COL_DATE_END][:10] if COL_DATE_END < len(row) else ""
        metadata.append(
            {
                "id": row[COL_ID].strip() if COL_ID < len(row) else "",
                "year": row[COL_YEAR].strip() if COL_YEAR < len(row) else "",
                "side_b_raw": row[COL_SIDE_B].strip() if COL_SIDE_B < len(row) else "",
                "location": location,
                "province": province,
                "date_start": date_start,
                "date_end": date_end,
                "headline": headline,
            }
        )
    return metadata


def compute_metrics(actual, predicted):
    """Compute standard accuracy, balanced accuracy, macro F1, and per-class recall."""
    correct = recall_precision.count_correct(actual, predicted)
    total = len(actual)
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    bal_acc = balanced_accuracy.balanced_accuracy(actual, predicted) * 100.0
    mf1 = macrof1.macrof1(actual, predicted) * 100.0

    class_scores = recall_precision.class_recall_precision(actual, predicted)
    per_class = {}
    for cls in (0, 1, 2):
        rec = class_scores[cls]["recall"] * 100.0
        prec = class_scores[cls]["precision"] * 100.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = sum(1 for y in actual if y == cls)
        per_class[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "macro_f1": mf1,
        "per_class": per_class,
    }


def train_and_save(csv_path=CSV_PATH, output_path=OUTPUT_PATH):
    start_time = time.perf_counter()

    print(f"Loading and preprocessing {csv_path}...")
    processed_data = preprocess.preprocess(csv_path)
    raw_metadata = extract_metadata(csv_path)

    # Keep a snapshot of raw features to assert 1-to-1 metadata alignment after splitting
    feature_snapshot = [list(row[:-1]) for row in processed_data]

    # Perform the exact SEED=67 stratified split used in main.py
    train_data, test_data = split.split_dataset(processed_data)

    # Repeat the exact deterministic split on [orig_idx, raw_side_b] to align metadata
    indexed_rows = [[idx, row[-1]] for idx, row in enumerate(processed_data)]
    train_idx_rows, test_idx_rows = split.split_dataset(indexed_rows)

    assert len(train_data) == len(train_idx_rows), "Train split alignment mismatch!"
    assert len(test_data) == len(test_idx_rows), "Test split alignment mismatch!"

    train_meta = []
    for i, (orig_idx, _) in enumerate(train_idx_rows):
        assert feature_snapshot[orig_idx] == train_data[i][:-1], f"Train row {i} feature mismatch!"
        train_meta.append(raw_metadata[orig_idx])

    test_meta = []
    for i, (orig_idx, _) in enumerate(test_idx_rows):
        assert feature_snapshot[orig_idx] == test_data[i][:-1], f"Test row {i} feature mismatch!"
        test_meta.append(raw_metadata[orig_idx])

    # Map string side_b targets into classes 0 (CPP), 1 (Mindanao), 2 (Others)
    train_data = preprocess.classify_groups(train_data)
    test_data = preprocess.classify_groups(test_data)
    test_actual = preprocess.get_target(test_data)

    # Train all 3 models using their exact default/project hyperparameters
    print("Training Gaussian Naive Bayes...")
    t0 = time.perf_counter()
    nb_model = NB_Classifier()
    nb_model.fit(train_data)
    nb_train_s = time.perf_counter() - t0

    print("Training One-vs-Rest Linear Perceptron...")
    t0 = time.perf_counter()
    perceptron_model = Perceptron()
    perceptron_model.fit(train_data)
    perceptron_train_s = time.perf_counter() - t0

    print("Training K-Nearest Neighbors (k=3, p=2)...")
    t0 = time.perf_counter()
    knn_model = KNN(k=3, p=2)
    knn_model.fit(train_data)
    knn_train_s = time.perf_counter() - t0

    # Evaluate on the held-out 882-row test split to verify exact reproducibility
    print("Evaluating all 3 models on held-out test set...")
    nb_preds = nb_model.predict(test_data)
    perceptron_preds = perceptron_model.predict(test_data)
    knn_preds = knn_model.predict(test_data)

    metrics = {
        "nb": compute_metrics(test_actual, nb_preds),
        "perceptron": compute_metrics(test_actual, perceptron_preds),
        "knn": compute_metrics(test_actual, knn_preds),
    }

    bundle = {
        "version": 1,
        "seed": split.SEED,
        "dataset_path": csv_path,
        "total_rows": len(processed_data),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_times_sec": {
            "nb": nb_train_s,
            "perceptron": perceptron_train_s,
            "knn": knn_train_s,
        },
        "models": {
            "nb": nb_model,
            "perceptron": perceptron_model,
            "knn": knn_model,
        },
        "train_meta": train_meta,
        "test_data": test_data,
        "test_meta": test_meta,
        "test_predictions": {
            "nb": nb_preds,
            "perceptron": perceptron_preds,
            "knn": knn_preds,
        },
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.perf_counter() - start_time
    size_kb = os.path.getsize(output_path) / 1024.0

    print()
    print("=" * 72)
    print(f"Saved pre-trained models to: {output_path} ({size_kb:.1f} KB) in {elapsed:.2f}s")
    print(f"Split: {len(train_data)} train / {len(test_data)} test (SEED = {split.SEED})")
    print("-" * 72)
    print(f"{'Model':<28} | {'Correct':>9} | {'Accuracy':>8} | {'Bal. Acc':>8} | {'Macro F1':>8}")
    print("-" * 72)
    for label, key in (
        ("Gaussian Naive Bayes", "nb"),
        ("Linear Perceptron (OvR)", "perceptron"),
        ("K-Nearest Neighbors (k=3)", "knn"),
    ):
        m = metrics[key]
        print(
            f"{label:<28} | {m['correct']:>4}/{m['total']:<4} | "
            f"{m['accuracy']:>7.2f}% | {m['balanced_accuracy']:>7.2f}% | {m['macro_f1']:>7.2f}%"
        )
    print("=" * 72)
    print("Ready! Launch the live demo anytime with:  python3 demo_cli.py")


if __name__ == "__main__":
    train_and_save()
