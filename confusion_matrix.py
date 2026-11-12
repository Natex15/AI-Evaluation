# compute confusion matrix using basic loops and lists
def confusion_matrix(actual, predicted, classes=None):
    # find unique classes if not provided
    if classes is None:
        classes = []
        # collect unique classes from actual labels
        for i in range(len(actual)):
            if actual[i] not in classes:
                classes.append(actual[i])
        # collect unique classes from predicted labels
        for i in range(len(predicted)):
            if predicted[i] not in classes:
                classes.append(predicted[i])
        # sort classes in ascending order
        classes.sort()

    # initialize 2d matrix with zeros using i and j
    matrix = []
    for i in range(len(classes)):
        row = []
        for j in range(len(classes)):
            row.append(0)
        matrix.append(row)

    # count occurrences for each actual and predicted pair
    for i in range(len(actual)):
        actual_val = actual[i]
        predicted_val = predicted[i]

        # find row index for actual class
        actual_idx = -1
        for j in range(len(classes)):
            if classes[j] == actual_val:
                actual_idx = j
                break

        # find column index for predicted class
        predicted_idx = -1
        for j in range(len(classes)):
            if classes[j] == predicted_val:
                predicted_idx = j
                break

        # increment cell count if both indices are found
        if actual_idx != -1 and predicted_idx != -1:
            matrix[actual_idx][predicted_idx] += 1

    return matrix

# print formatted confusion matrix table
def print_confusion_matrix(matrix, classes=None):
    # default classes list if not provided
    if classes is None:
        classes = []
        for i in range(len(matrix)):
            classes.append(i)

    # build header row for predicted labels
    header = "Actual / Pred"
    for j in range(len(classes)):
        header = header + f"{classes[j]:>8}"
    print(header)

    # print each actual class row with counts
    for i in range(len(matrix)):
        row_str = f"{classes[i]:<13}"
        for j in range(len(matrix[i])):
            row_str = row_str + f"{matrix[i][j]:>8}"
        print(row_str)


# print formatted confusion matrix table with raw counts and row-normalized percentages
def print_confusion_matrix_detailed(matrix):
    classes = [i for i in range(len(matrix))]
    class_names = {0: "0 (CPP)", 1: "1 (Mindanao)", 2: "2 (Others)"}

    labels = [class_names.get(c, str(c)) for c in classes]

    header = f"{'Actual \\ Predicted':<18}"
    for label in labels:
        header += f" | {label:>16}"
    header += f" | {'Support':>8}"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for i in range(len(matrix)):
        row_total = sum(matrix[i])
        row_str = f"{labels[i]:<18}"
        for j in range(len(matrix[i])):
            count = matrix[i][j]
            pct = (count / row_total * 100.0) if row_total > 0 else 0.0
            cell = f"{count} ({pct:5.1f}%)"
            row_str += f" | {cell:>16}"
        row_str += f" | {row_total:>8}"
        print(row_str)
    print(separator)

