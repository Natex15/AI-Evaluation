def balanced_accuracy(actual, predicted):
    # initialize counters for true positives and false negatives for each class
    true_positives = []
    false_negatives = []
    for i in range(3):
        true_positives.append(0)
        false_negatives.append(0)

    # count true positives and false negatives for class 0
    for i in range(len(actual)):
        actual_val = actual[i]
        predicted_val = predicted[i]
        if actual_val == 0:
            if predicted_val == 0:
                true_positives[0] = true_positives[0] + 1
            else:
                false_negatives[0] = false_negatives[0] + 1

    # count true positives and false negatives for class 1
    for i in range(len(actual)):
        actual_val = actual[i]
        predicted_val = predicted[i]
        if actual_val == 1:
            if predicted_val == 1:
                true_positives[1] = true_positives[1] + 1
            else:
                false_negatives[1] = false_negatives[1] + 1

    # count true positives and false negatives for class 2
    for i in range(len(actual)):
        actual_val = actual[i]
        predicted_val = predicted[i]
        if actual_val == 2:
            if predicted_val == 2:
                true_positives[2] = true_positives[2] + 1
            else:
                false_negatives[2] = false_negatives[2] + 1

    # compute recall for each class separately
    recalls = []
    for i in range(3):
        denom = true_positives[i] + false_negatives[i]
        if denom > 0:
            recall_score = true_positives[i] / denom
        else:
            recall_score = 0
        recalls.append(recall_score)

    # sum all class recalls
    recall_sum = 0
    for i in range(len(recalls)):
        recall_sum = recall_sum + recalls[i]

    # divide by the number of classes to get balanced accuracy
    balanced_acc = recall_sum / 3
    return balanced_acc
