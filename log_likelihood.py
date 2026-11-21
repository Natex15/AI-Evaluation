import math
import preprocess
import split
from gaussian_nb import NB_Classifier

FEATURE_NAMES = [
    "longitude",
    "latitude",
    "deaths_a",
    "deaths_b",
    "deaths_civilians",
    "best",
    "type_of_violence",
    "duration_days",
]


# compute median of a list of numbers
def compute_median(values):
    if len(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


# format mean or variance cleanly, preserving tiny smoothing values like 1e-9
def format_stat(value):
    if value is None:
        return "-"
    abs_val = abs(value)
    if abs_val > 0 and abs_val < 0.001:
        return f"{value:.2e}"
    if abs_val >= 100.0:
        return f"{value:.2f}"
    if abs_val > 0 and abs_val < 0.1:
        return f"{value:.5f}"
    return f"{value:.4f}"


# build table rows comparing Gaussian NB feature parameters and median log-likelihood gaps
def build_log_likelihood_rows(nb_model, test_data, cpp_class=0, mindanao_class=1):
    # filter test samples belonging to the target class (default: Class 1 Mindanao)
    target_samples = []
    for i in range(len(test_data)):
        if test_data[i][-1] == mindanao_class:
            target_samples.append(test_data[i])

    num_samples = len(target_samples)
    num_features = len(FEATURE_NAMES)

    # initialize lists to store per-feature log-likelihood differences (CPP - Mindanao)
    feature_diffs = []
    for j in range(num_features):
        feature_diffs.append([])

    total_diffs = []
    predicted_cpp_count = 0

    prior_cpp = nb_model.priors[cpp_class]
    prior_min = nb_model.priors[mindanao_class]
    prior_diff = math.log(prior_cpp) - math.log(prior_min)

    # evaluate log-likelihood differences across all Mindanao test events
    for i in range(num_samples):
        features = target_samples[i][:-1]
        log_prob_cpp = math.log(prior_cpp)
        log_prob_min = math.log(prior_min)

        for j in range(num_features):
            ll_cpp = nb_model.calculate_log_likelihood(
                features[j],
                nb_model.means[cpp_class][j],
                nb_model.variances[cpp_class][j],
            )
            ll_min = nb_model.calculate_log_likelihood(
                features[j],
                nb_model.means[mindanao_class][j],
                nb_model.variances[mindanao_class][j],
            )
            feature_diffs[j].append(ll_cpp - ll_min)
            log_prob_cpp += ll_cpp
            log_prob_min += ll_min

        total_diffs.append(log_prob_cpp - log_prob_min)

        if nb_model.predict_one(features) == cpp_class:
            predicted_cpp_count += 1

    # collect one row per feature
    rows = []
    for j in range(num_features):
        favors_cpp_count = 0
        for diff in feature_diffs[j]:
            if diff > 0:
                favors_cpp_count += 1

        favors_cpp_pct = (favors_cpp_count / num_samples * 100.0) if num_samples > 0 else 0.0
        rows.append({
            "term": FEATURE_NAMES[j],
            "cpp_mean_str": format_stat(nb_model.means[cpp_class][j]),
            "cpp_var_str": format_stat(nb_model.variances[cpp_class][j]),
            "min_mean_str": format_stat(nb_model.means[mindanao_class][j]),
            "min_var_str": format_stat(nb_model.variances[mindanao_class][j]),
            "median_diff": compute_median(feature_diffs[j]),
            "favors_cpp_str": f"{favors_cpp_pct:.1f}%",
        })

    # add class log-prior row
    prior_favors_pct = 100.0 if prior_diff > 0 else 0.0
    rows.append({
        "term": "Class Log-Prior ln P(c)",
        "cpp_mean_str": f"{prior_cpp * 100.0:.2f}%",
        "cpp_var_str": "-",
        "min_mean_str": f"{prior_min * 100.0:.2f}%",
        "min_var_str": "-",
        "median_diff": prior_diff,
        "favors_cpp_str": f"{prior_favors_pct:.1f}%",
    })

    # sort rows in descending order of median log-likelihood gap (CPP - Mindanao)
    rows.sort(key=lambda row: row["median_diff"], reverse=True)

    # build net log-posterior summary
    predicted_cpp_pct = (predicted_cpp_count / num_samples * 100.0) if num_samples > 0 else 0.0
    summary = {
        "num_samples": num_samples,
        "median_total_diff": compute_median(total_diffs),
        "predicted_cpp_count": predicted_cpp_count,
        "predicted_cpp_pct": predicted_cpp_pct,
    }

    return rows, summary


# print formatted table matching Table 5 in docs/index.html
def print_log_likelihood_table(nb_model, test_data, cpp_class=0, mindanao_class=1):
    rows, summary = build_log_likelihood_rows(nb_model, test_data, cpp_class, mindanao_class)

    header = (
        f"{'Feature / Term':<24} | {'CPP Mean':>10} | {'CPP Var':>10} | "
        f"{'Min Mean':>10} | {'Min Var':>10} | {'Median dLL (C0-C1)':>18} | {'Favors CPP':>12}"
    )
    banner = "=" * len(header)
    separator = "-" * len(header)

    print(banner)
    print(
        f"TABLE 5: GAUSSIAN NAIVE BAYES PARAMETERS & MEDIAN LOG-LIKELIHOOD GAP "
        f"(MINDANAO TEST EVENTS, n = {summary['num_samples']})"
    )
    print(banner)
    print(header)
    print(separator)

    for row in rows:
        print(
            f"{row['term']:<24} | {row['cpp_mean_str']:>10} | {row['cpp_var_str']:>10} | "
            f"{row['min_mean_str']:>10} | {row['min_var_str']:>10} | "
            f"{row['median_diff']:>+18.4f} | {row['favors_cpp_str']:>12}"
        )

    print(separator)
    net_note = "Sum across prior + all 8 features"
    net_favors = f"{summary['predicted_cpp_pct']:.1f}% -> CPP"
    print(
        f"{'Net Log-Posterior Gap':<24} | {net_note:<49} | "
        f"{summary['median_total_diff']:>+18.4f} | {net_favors:>12}"
    )
    print(separator)
    print()


if __name__ == "__main__":
    processed_data = preprocess.preprocess("conflict_data_phl.csv")
    train_data, test_data = split.split_dataset(processed_data)
    train_data = preprocess.classify_groups(train_data)
    test_data = preprocess.classify_groups(test_data)

    nb_model = NB_Classifier()
    nb_model.fit(train_data)

    print_log_likelihood_table(nb_model, test_data)

