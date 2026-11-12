import csv
import balanced_accuracy
import confusion_matrix
import macrof1
import recall_precision

CLASS_NAMES = {
    0: "0 (CPP)",
    1: "1 (Mindanao)",
    2: "2 (Others)"
}


def load_predictions_csv(filepath):
    actual = []
    predicted = []
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                actual.append(int(row[0]))
                predicted.append(int(row[1]))
    return actual, predicted


def build_overall_summary(test_actual, model_predictions):
    total_tests = len(test_actual)
    summary_rows = []

    for model_name, predictions in model_predictions:
        correct = recall_precision.count_correct(test_actual, predictions)
        std_acc = (correct / total_tests) * 100.0 if total_tests > 0 else 0.0
        bal_acc = balanced_accuracy.balanced_accuracy(test_actual, predictions) * 100.0
        mac_prec = recall_precision.precision(test_actual, predictions) * 100.0
        mac_rec = recall_precision.recall(test_actual, predictions) * 100.0
        mac_f1 = macrof1.macrof1(test_actual, predictions) * 100.0

        summary_rows.append({
            "model": model_name,
            "correct": correct,
            "total": total_tests,
            "correct_total_str": f"{correct} / {total_tests}",
            "std_acc": std_acc,
            "bal_acc": bal_acc,
            "mac_prec": mac_prec,
            "mac_rec": mac_rec,
            "mac_f1": mac_f1,
        })

    return summary_rows


def build_per_class_breakdown(test_actual, model_predictions):
    per_class_rows = []

    for model_name, predictions in model_predictions:
        raw_scores = recall_precision.get_scores(test_actual, predictions)
        class_rp = recall_precision.class_recall_precision(test_actual, predictions)

        for c in range(3):
            support = raw_scores[c]["tp"] + raw_scores[c]["fn"]
            prec = class_rp[c]["precision"]
            rec = class_rp[c]["recall"]
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_rows.append({
                "model": model_name,
                "class_id": c,
                "class_label": CLASS_NAMES[c],
                "support": support,
                "precision": prec * 100.0,
                "recall": rec * 100.0,
                "f1": f1 * 100.0,
            })

    return per_class_rows


def print_overall_table(summary_rows):
    header = (
        f"{'Model':<26} | {'Correct / Total':>15} | {'Std Acc (%)':>11} | "
        f"{'Balanced Acc (%)':>16} | {'Macro Prec (%)':>14} | {'Macro Rec (%)':>13} | {'Macro F1 (%)':>12}"
    )
    banner = "=" * len(header)
    separator = "-" * len(header)
    print(banner)
    print("TABLE 1: OVERALL MODEL EVALUATION METRICS")
    print(banner)
    print(header)
    print(separator)

    for row in summary_rows:
        print(
            f"{row['model']:<26} | {row['correct_total_str']:>15} | "
            f"{row['std_acc']:>10.2f}% | {row['bal_acc']:>15.2f}% | "
            f"{row['mac_prec']:>13.2f}% | {row['mac_rec']:>12.2f}% | "
            f"{row['mac_f1']:>11.2f}%"
        )
    print(separator)
    print()


def print_per_class_table(per_class_rows):
    header = (
        f"{'Model':<26} | {'Class Label':<14} | {'Support':>8} | "
        f"{'Precision (%)':>13} | {'Recall / Class Acc (%)':>22} | {'Class F1 (%)':>12}"
    )
    banner = "=" * len(header)
    separator = "-" * len(header)
    print(banner)
    print("TABLE 2: PER-CLASS PERFORMANCE BREAKDOWN (0 = CPP, 1 = Mindanao, 2 = Others)")
    print(banner)
    print(header)
    print(separator)

    for idx, row in enumerate(per_class_rows):
        print(
            f"{row['model']:<26} | {row['class_label']:<14} | {row['support']:>8} | "
            f"{row['precision']:>12.2f}% | {row['recall']:>21.2f}% | {row['f1']:>11.2f}%"
        )
        if (idx + 1) % 3 == 0 and (idx + 1) < len(per_class_rows):
            print(separator)

    print(separator)
    print()


def print_confusion_matrix_tables(test_actual, model_predictions):
    banner = "=" * 86
    print(banner)
    print("TABLE 3: CONFUSION MATRICES (RAW COUNTS & ROW-NORMALIZED CLASS RECALL %)")
    print(banner)
    for model_name, predictions in model_predictions:
        print(f"\nModel: {model_name}")
        matrix = confusion_matrix.confusion_matrix(test_actual, predictions, classes=[0, 1, 2])
        confusion_matrix.print_confusion_matrix_detailed(matrix)
    print()


def export_tables_to_csv(
    summary_rows,
    per_class_rows,
    summary_csv_path="summary_metrics.csv",
    per_class_csv_path="per_class_metrics.csv"
):
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Model",
            "Correct",
            "Total",
            "Standard_Accuracy_Pct",
            "Balanced_Accuracy_Pct",
            "Macro_Precision_Pct",
            "Macro_Recall_Pct",
            "Macro_F1_Pct"
        ])
        for row in summary_rows:
            writer.writerow([
                row["model"],
                row["correct"],
                row["total"],
                f"{row['std_acc']:.4f}",
                f"{row['bal_acc']:.4f}",
                f"{row['mac_prec']:.4f}",
                f"{row['mac_rec']:.4f}",
                f"{row['mac_f1']:.4f}",
            ])

    with open(per_class_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Model",
            "Class_ID",
            "Class_Label",
            "Support",
            "Precision_Pct",
            "Recall_Class_Accuracy_Pct",
            "Class_F1_Pct"
        ])
        for row in per_class_rows:
            writer.writerow([
                row["model"],
                row["class_id"],
                row["class_label"],
                row["support"],
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
            ])


def table_comparison(test_actual, model_predictions):
    summary_rows = build_overall_summary(test_actual, model_predictions)
    per_class_rows = build_per_class_breakdown(test_actual, model_predictions)

    print_overall_table(summary_rows)
    print_per_class_table(per_class_rows)
    print_confusion_matrix_tables(test_actual, model_predictions)

    export_tables_to_csv(summary_rows, per_class_rows)
    print("Exported comparison tables to 'summary_metrics.csv' and 'per_class_metrics.csv'.")


if __name__ == "__main__":
    nb_actual, nb_pred = load_predictions_csv("nb_predictions.csv")
    _, perc_pred = load_predictions_csv("perceptron_predictions.csv")
    _, knn_pred = load_predictions_csv("knn_predictions.csv")

    models = [
        ("Gaussian Naive Bayes", nb_pred),
        ("Linear Perceptron", perc_pred),
        ("K-Nearest Neighbors (k=3)", knn_pred),
    ]
    table_comparison(nb_actual, models)
