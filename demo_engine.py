"""Glass-box inference and explanation engine for the 3 pre-trained models.

Loads models/trained_models.pkl produced by train_models.py and executes the
original untouched model methods (NB_Classifier.predict_one,
Perceptron.predict_one, KNN.predict) while exposing each model's internal
decision mechanics for live demonstration.
"""

import math
import os
import pickle
import time

DEFAULT_MODEL_PATH = os.path.join("models", "trained_models.pkl")

CLASS_IDS = (0, 1, 2)

CLASS_NAMES = {
    0: "CPP (CPP/NPA)",
    1: "Mindanao Groups",
    2: "Others",
}

CLASS_SHORT = {
    0: "CPP",
    1: "Mindanao",
    2: "Others",
}

VIOLENCE_TYPES = {
    1: "State-based conflict",
    2: "Non-state conflict",
    3: "One-sided violence",
}

FEATURE_SPECS = [
    {
        "key": "longitude",
        "label": "Longitude",
        "dtype": float,
        "hint": "PH range ~118.0 to 126.6 (e.g., Manila 120.98, Cotabato 124.25)",
    },
    {
        "key": "latitude",
        "label": "Latitude",
        "dtype": float,
        "hint": "PH range ~4.6 to 19.0 (Luzon ~13-18, Visayas ~9-12, Mindanao ~5-9.5)",
    },
    {
        "key": "deaths_a",
        "label": "Deaths Side A (Gov)",
        "dtype": int,
        "hint": "Philippine government force fatalities (>= 0)",
    },
    {
        "key": "deaths_b",
        "label": "Deaths Side B (Rebel)",
        "dtype": int,
        "hint": "Insurgent / rebel fatalities (>= 0)",
    },
    {
        "key": "deaths_civilians",
        "label": "Deaths Civilians",
        "dtype": int,
        "hint": "Civilian fatalities (>= 0)",
    },
    {
        "key": "best",
        "label": "Best Total Estimate",
        "dtype": int,
        "hint": "UCDP best estimate of total fatalities (>= 0)",
    },
    {
        "key": "type_of_violence",
        "label": "Type of Violence",
        "dtype": int,
        "hint": "1 = State-based, 2 = Non-state, 3 = One-sided",
    },
    {
        "key": "duration_days",
        "label": "Duration (Days)",
        "dtype": int,
        "hint": "Event duration in days (0 = single-day incident)",
    },
]

FEATURE_KEYS = [spec["key"] for spec in FEATURE_SPECS]


def load_bundle(model_path=DEFAULT_MODEL_PATH):
    """Load the serialized pre-trained bundle and return (bundle, load_ms)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pre-trained model file not found at '{model_path}'.\n"
            f"Run 'python3 train_models.py' once to train and save the models."
        )
    t0 = time.perf_counter()
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    load_ms = (time.perf_counter() - t0) * 1000.0
    return bundle, load_ms


def format_class(cls_id, short=False):
    mapping = CLASS_SHORT if short else CLASS_NAMES
    return f"Class {cls_id}: {mapping.get(cls_id, str(cls_id))}"


def _softmax_from_log_scores(log_scores):
    """Numerically stable softmax over {class_id: log_score}."""
    max_val = max(log_scores.values())
    exps = {c: math.exp(v - max_val) for c, v in log_scores.items()}
    total = sum(exps.values())
    return {c: (exps[c] / total) * 100.0 for c in log_scores}


def explain_nb(nb_model, features):
    """Run Gaussian Naive Bayes prediction and return log-posterior & feature breakdown."""
    prediction = nb_model.predict_one(features)

    log_priors = {}
    log_posteriors = {}
    feature_ll = {c: [] for c in CLASS_IDS}

    for c in CLASS_IDS:
        lp = math.log(nb_model.priors[c])
        log_priors[c] = lp
        total = lp
        for j in range(len(features)):
            ll = nb_model.calculate_log_likelihood(
                features[j],
                nb_model.means[c][j],
                nb_model.variances[c][j],
            )
            feature_ll[c].append(ll)
            total += ll
        log_posteriors[c] = total

    ranked = sorted(CLASS_IDS, key=lambda c: log_posteriors[c], reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    margin = log_posteriors[winner] - log_posteriors[runner_up]
    probs = _softmax_from_log_scores(log_posteriors)

    # Compare per-feature log-likelihood advantage of winner vs runner-up
    feature_breakdown = []
    for j, spec in enumerate(FEATURE_SPECS):
        delta = feature_ll[winner][j] - feature_ll[runner_up][j]
        feature_breakdown.append(
            {
                "index": j,
                "key": spec["key"],
                "value": features[j],
                "ll_by_class": {c: feature_ll[c][j] for c in CLASS_IDS},
                "delta_vs_runner_up": delta,
                "winner_var": nb_model.variances[winner][j],
                "runner_up_var": nb_model.variances[runner_up][j],
            }
        )

    dominant = max(feature_breakdown, key=lambda r: abs(r["delta_vs_runner_up"]))
    variance_spike = (
        dominant["key"] == "type_of_violence"
        and (dominant["winner_var"] < 1e-6 or dominant["runner_up_var"] < 1e-6)
    )

    return {
        "prediction": prediction,
        "winner": winner,
        "runner_up": runner_up,
        "margin": margin,
        "log_priors": log_priors,
        "log_posteriors": log_posteriors,
        "probabilities": probs,
        "feature_breakdown": feature_breakdown,
        "dominant_feature": dominant,
        "variance_spike_note": variance_spike,
    }


def explain_perceptron(perceptron_model, features):
    """Run OvR Linear Perceptron prediction and return hyperplane scores & contributions."""
    prediction = perceptron_model.predict_one(features)

    scores = {}
    contributions = {}
    for idx, cls_id in enumerate(perceptron_model.classes):
        bias = perceptron_model.biases[idx]
        weights = perceptron_model.weights[idx]
        terms = [weights[j] * features[j] for j in range(len(features))]
        scores[cls_id] = bias + sum(terms)
        contributions[cls_id] = {
            "bias": bias,
            "terms": terms,
        }

    ranked = sorted(perceptron_model.classes, key=lambda c: scores[c], reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    margin = scores[winner] - scores[runner_up]

    # Feature-level net contribution to (winner score - runner-up score)
    diff_rows = []
    for j, spec in enumerate(FEATURE_SPECS):
        w_term = contributions[winner]["terms"][j]
        r_term = contributions[runner_up]["terms"][j]
        diff_rows.append(
            {
                "key": spec["key"],
                "value": features[j],
                "winner_term": w_term,
                "runner_up_term": r_term,
                "net_push": w_term - r_term,
            }
        )
    diff_rows.sort(key=lambda r: abs(r["net_push"]), reverse=True)

    return {
        "prediction": prediction,
        "winner": winner,
        "runner_up": runner_up,
        "margin": margin,
        "scores": scores,
        "contributions": contributions,
        "top_drivers": diff_rows,
    }


def explain_knn(knn_model, train_meta, features):
    """Run KNN (k=3, p=2) prediction and return the exact k nearest training incidents."""
    prediction = knn_model.predict([features])[0]

    p = knn_model.p
    k = knn_model.k
    scored = []
    for train_idx, train_row in enumerate(knn_model.training_data):
        dist_sum = 0.0
        for j in range(8):
            dist_sum += abs(float(features[j]) - float(train_row[j])) ** p
        dist = dist_sum ** (1.0 / p)
        scored.append((dist, int(train_row[8]), train_idx))

    # Mirror knn.py's exact sort order on (distance, label)
    scored.sort(key=lambda item: (item[0], item[1]))
    top_k = scored[:k]

    # Mirror knn.py's tie-break behavior so we can report if a tie occurred
    active = [(d, lbl) for d, lbl, _ in top_k]
    initial_counts = knn_model.counter(active)
    tie_broken = False
    while (
        (max(initial_counts) == initial_counts[0] and max(initial_counts) == initial_counts[1])
        or (max(initial_counts) == initial_counts[1] and max(initial_counts) == initial_counts[2])
        or (max(initial_counts) == initial_counts[0] and max(initial_counts) == initial_counts[2])
    ):
        tie_broken = True
        active.pop()
        initial_counts = knn_model.counter(active)

    raw_counts = knn_model.counter([(d, lbl) for d, lbl, _ in top_k])

    neighbors = []
    for rank, (dist, lbl, train_idx) in enumerate(top_k, start=1):
        row = knn_model.training_data[train_idx]
        meta = train_meta[train_idx] if train_idx < len(train_meta) else {}
        neighbors.append(
            {
                "rank": rank,
                "train_idx": train_idx,
                "distance": dist,
                "label": lbl,
                "features": row[:8],
                "side_b_raw": meta.get("side_b_raw", CLASS_SHORT.get(lbl, "?")),
                "year": meta.get("year", ""),
                "location": meta.get("location", "Unknown"),
                "province": meta.get("province", ""),
            }
        )

    return {
        "prediction": prediction,
        "k": k,
        "p": p,
        "raw_counts": raw_counts,
        "final_counts": initial_counts,
        "tie_broken": tie_broken,
        "neighbors": neighbors,
    }


def predict_all(bundle, features):
    """Run all 3 pre-trained models on an 8-feature vector and return full explanations."""
    clean_features = [
        float(features[0]),
        float(features[1]),
        int(features[2]),
        int(features[3]),
        int(features[4]),
        int(features[5]),
        int(features[6]),
        int(features[7]),
    ]
    models = bundle["models"]
    nb_res = explain_nb(models["nb"], clean_features)
    perc_res = explain_perceptron(models["perceptron"], clean_features)
    knn_res = explain_knn(models["knn"], bundle["train_meta"], clean_features)

    preds = {
        "nb": nb_res["prediction"],
        "perceptron": perc_res["prediction"],
        "knn": knn_res["prediction"],
    }
    unanimous = len(set(preds.values())) == 1

    return {
        "features": clean_features,
        "predictions": preds,
        "unanimous": unanimous,
        "nb": nb_res,
        "perceptron": perc_res,
        "knn": knn_res,
    }


def _pick_best_candidate(candidates, test_meta):
    """Prefer test events with rich human-readable metadata (headline + location)."""
    if not candidates:
        return None

    def score(idx):
        meta = test_meta[idx]
        s = 0
        if meta.get("headline"):
            s += 3
        if meta.get("location") and meta["location"] != "Unknown location":
            s += 2
        if meta.get("province"):
            s += 1
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def build_curated_presets(bundle):
    """Select 5 real held-out test cases that showcase each model's strengths & failure modes."""
    test_data = bundle["test_data"]
    test_meta = bundle["test_meta"]
    preds = bundle["test_predictions"]

    presets = []
    used = set()

    def register(title, takeaway, candidates):
        filtered = [idx for idx in candidates if idx not in used]
        picked = _pick_best_candidate(filtered, test_meta)
        if picked is not None:
            used.add(picked)
            presets.append(
                {
                    "test_idx": picked,
                    "title": title,
                    "takeaway": takeaway,
                }
            )

    # 1. Classic Luzon/Visayas CPP event where all 3 models succeed
    register(
        "Classic CPP Guerrilla Clash (All 3 Models Correct)",
        "Northern/central coordinates + state-based skirmish: all 3 models agree on CPP.",
        [
            i
            for i, row in enumerate(test_data)
            if row[-1] == 0
            and row[1] >= 12.0
            and row[5] >= 2
            and preds["nb"][i] == 0
            and preds["perceptron"][i] == 0
            and preds["knn"][i] == 0
        ],
    )

    # 2. Mindanao insurgent clash where NB collapses to CPP while KNN & Perceptron succeed
    register(
        "Mindanao Insurgent Clash (Naive Bayes Collapses -> CPP)",
        "Demonstrates the 3.6% Mindanao recall collapse in NB caused by type_of_violence=1 smoothing (1e-9).",
        [
            i
            for i, row in enumerate(test_data)
            if row[-1] == 1
            and row[6] == 1
            and row[5] >= 2
            and preds["knn"][i] == 1
            and preds["perceptron"][i] == 1
            and preds["nb"][i] == 0
        ],
    )

    # 3. Southern CPP clash where Perceptron fails due to linear inseparability, KNN succeeds
    register(
        "Southern CPP Clash (Perceptron Fails, KNN Succeeds)",
        "A CPP clash in Mindanao coordinates defeats Perceptron's north-south hyperplane, while KNN uses local neighbors.",
        [
            i
            for i, row in enumerate(test_data)
            if row[-1] == 0
            and row[1] <= 9.5
            and preds["knn"][i] == 0
            and preds["perceptron"][i] == 1
        ],
    )

    # 4. Hard Mindanao case where ONLY KNN gets it right (both NB and Perceptron miss)
    register(
        "Complex Mindanao Incident (Only KNN Gets It Right)",
        "Both Gaussian NB and Linear Perceptron fail here; KNN succeeds via 3 nearest historical neighbors.",
        [
            i
            for i, row in enumerate(test_data)
            if row[-1] == 1
            and preds["knn"][i] == 1
            and preds["nb"][i] != 1
            and preds["perceptron"][i] != 1
        ],
    )

    # 5. Class 2 (Others) one-sided violence against civilians
    register(
        "One-Sided Violence vs. Civilians (Class 2: Others)",
        "type_of_violence=3 and civilian fatalities cleanly separate Class 2 across all models.",
        [
            i
            for i, row in enumerate(test_data)
            if row[-1] == 2
            and row[6] == 3
            and row[4] >= 1
            and preds["knn"][i] == 2
            and preds["perceptron"][i] == 2
            and preds["nb"][i] == 2
        ],
    )

    return presets
