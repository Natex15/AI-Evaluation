import recall_precision

def macrof1(actual, predicted):
    f1sum = 0
    class_scores = recall_precision.class_recall_precision(actual, predicted)
    for side_b in range(3):
        r = class_scores[side_b]["recall"]
        p = class_scores[side_b]["precision"]
        f1_score = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        f1sum += f1_score
    return f1sum / 3
