def get_scores(actual, predicted):
    scores = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(3)} # c = 0,1,2 ; tp = true positive, fp = false positive, fn = false negative
    for a, p in zip(actual, predicted):
        if a == p:
            scores[a]["tp"] += 1
        else:
            scores[a]["fn"] += 1
            scores[p]["fp"] += 1
    return scores

def recall(actual, predicted):
    recall_sum = 0
    scores = get_scores(actual, predicted)
    for side_b in range(3):
        denom = scores[side_b]["tp"] + scores[side_b]["fn"]
        recall_score = scores[side_b]["tp"] / denom if denom > 0 else 0
        recall_sum += recall_score
    return recall_sum / 3

def precision(actual, predicted):
    precision_sum = 0
    scores = get_scores(actual, predicted)
    for side_b in range(3):
        denom = scores[side_b]["tp"] + scores[side_b]["fp"]
        precision_score = scores[side_b]["tp"] / denom if denom > 0 else 0
        precision_sum += precision_score
    return precision_sum / 3

def class_recall_precision(actual, predicted):
    scores = get_scores(actual, predicted)
    results = {}
    for side_b in range(3):
        r_denom = scores[side_b]["tp"] + scores[side_b]["fn"]
        p_denom = scores[side_b]["tp"] + scores[side_b]["fp"]
        r_score = scores[side_b]["tp"] / r_denom if r_denom > 0 else 0
        p_score = scores[side_b]["tp"] / p_denom if p_denom > 0 else 0
        results[side_b] = {"recall": r_score, "precision": p_score}
    return results

def count_correct(actual, predicted):
    correct = 0
    for a, p in zip(actual, predicted):
        if a == p:
            correct += 1
    return correct

