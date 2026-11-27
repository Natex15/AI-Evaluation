"""Interactive Terminal CLI Demo for the 3 Pre-Trained Conflict Classification Models.

Usage:
    python3 demo_cli.py                      # Interactive demo menu
    python3 demo_cli.py --preset 2           # Run curated preset #2 directly
    python3 demo_cli.py --test-index 42      # Run held-out test row #42
    python3 demo_cli.py --random             # Run a random unseen test row
    python3 demo_cli.py --custom 124.25 7.20 2 3 0 5 1 0
    python3 demo_cli.py --summary            # Print pre-trained model scorecard
"""

import argparse
import os
import random
import sys

from demo_engine import (
    CLASS_IDS,
    CLASS_NAMES,
    CLASS_SHORT,
    DEFAULT_MODEL_PATH,
    FEATURE_SPECS,
    VIOLENCE_TYPES,
    build_curated_presets,
    format_class,
    load_bundle,
    predict_all,
)


class Palette:
    """Minimal ANSI color helper that cleanly disables when non-TTY or --no-color."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def wrap(self, text, code):
        if not self.enabled:
            return str(text)
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text):
        return self.wrap(text, "1")

    def dim(self, text):
        return self.wrap(text, "2")

    def cyan(self, text):
        return self.wrap(text, "36")

    def blue(self, text):
        return self.wrap(text, "34")

    def green(self, text):
        return self.wrap(text, "32")

    def red(self, text):
        return self.wrap(text, "31")

    def yellow(self, text):
        return self.wrap(text, "33")

    def magenta(self, text):
        return self.wrap(text, "35")


WIDTH = 92


def hr(char="-"):
    return char * WIDTH


def truncate(text, max_len):
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_banner(bundle, load_ms, pal):
    print()
    print(pal.cyan("=" * WIDTH))
    print(pal.bold("  PHILIPPINE CONFLICT CLASSIFICATION — LIVE 3-MODEL DEMO (PRE-TRAINED)"))
    print(
        pal.dim(
            f"  Loaded {bundle['train_size']} train / {bundle['test_size']} test "
            f"(SEED={bundle['seed']}) in {load_ms:.1f} ms | "
            f"NB · OvR Perceptron · KNN (k=3)"
        )
    )
    print(pal.cyan("=" * WIDTH))


def print_scorecard(bundle, pal):
    print()
    print(pal.bold("PRE-TRAINED MODEL SCORECARD (HELD-OUT TEST SET, n = " + str(bundle["test_size"]) + ")"))
    print(hr("="))
    print(
        f"{'Model':<26} | {'Correct':>9} | {'Accuracy':>8} | {'Bal. Acc':>8} | "
        f"{'Macro F1':>8} | {'CPP Rec':>7} | {'Min Rec':>7} | {'Oth Rec':>7}"
    )
    print(hr("-"))
    for label, key in (
        ("K-Nearest Neighbors (k=3)", "knn"),
        ("Linear Perceptron (OvR)", "perceptron"),
        ("Gaussian Naive Bayes", "nb"),
    ):
        m = bundle["metrics"][key]
        pc = m["per_class"]
        min_rec_str = f"{pc[1]['recall']:>6.2f}%"
        if key == "nb":
            min_rec_str = pal.red(min_rec_str)
        elif key == "knn":
            min_rec_str = pal.green(min_rec_str)
        print(
            f"{label:<26} | {m['correct']:>4}/{m['total']:<4} | "
            f"{m['accuracy']:>7.2f}% | {m['balanced_accuracy']:>7.2f}% | "
            f"{m['macro_f1']:>7.2f}% | {pc[0]['recall']:>6.2f}% | "
            f"{min_rec_str} | {pc[2]['recall']:>6.2f}%"
        )
    print(hr("="))


def _status_badge(predicted, actual, pal):
    if actual is None:
        return pal.dim(f"{'CUSTOM INPUT':<12}")
    if predicted == actual:
        return pal.green(f"{'CORRECT':<12}")
    return pal.red(f"{'WRONG':<12}")


def render_prediction_report(
    result,
    pal,
    actual_label=None,
    meta=None,
    test_idx=None,
    title=None,
    takeaway=None,
    brief=False,
):
    features = result["features"]

    print()
    print(pal.cyan("=" * WIDTH))
    if title:
        print(pal.bold(f"  {title}"))
    elif test_idx is not None:
        print(pal.bold(f"  HELD-OUT TEST SAMPLE #{test_idx}"))
    else:
        print(pal.bold("  CUSTOM 'WHAT-IF' TEST INPUT"))
    if takeaway:
        print(pal.yellow(f"  Key Takeaway: {takeaway}"))
    print(pal.cyan("=" * WIDTH))

    # Context block for held-out UCDP events
    if meta is not None or actual_label is not None:
        print(pal.bold("Event Context (Unseen Held-Out Test Set):"))
        if test_idx is not None:
            event_id = meta.get("id", "N/A") if meta else "N/A"
            year = meta.get("year", "N/A") if meta else "N/A"
            loc = meta.get("location", "Unknown") if meta else "Unknown"
            print(f"  Test Index : #{test_idx}   |   UCDP ID: {event_id}   |   Year: {year}")
            print(f"  Location   : {loc}")
        if meta and meta.get("headline"):
            print(f"  Headline   : \"{truncate(meta['headline'], 74)}\"")
        if actual_label is not None:
            raw_actor = meta.get("side_b_raw", "") if meta else ""
            raw_suffix = f"  (raw side_b: {raw_actor})" if raw_actor else ""
            print(f"  Ground Truth: {pal.bold(format_class(actual_label))}{pal.dim(raw_suffix)}")
        print(hr("-"))

    # 8-feature input display (compact 2-column table)
    tov_name = VIOLENCE_TYPES.get(features[6], "Other")
    print(pal.bold("Input Feature Vector (8 Numerical Features):"))
    print(
        f"  [0] longitude        = {features[0]:<12.4f}    "
        f"[4] deaths_civilians = {features[4]}"
    )
    print(
        f"  [1] latitude         = {features[1]:<12.4f}    "
        f"[5] best (total)     = {features[5]}"
    )
    print(
        f"  [2] deaths_a (gov)   = {features[2]:<12d}    "
        f"[6] type_of_violence = {features[6]} ({tov_name})"
    )
    print(
        f"  [3] deaths_b (rebel) = {features[3]:<12d}    "
        f"[7] duration_days    = {features[7]}"
    )
    print(hr("-"))

    # Side-by-side summary table
    print(pal.bold("Side-by-Side Predictions (All 3 Pre-Trained Models):"))
    print(f"{'Model':<26} | {'Prediction':<24} | {'Result':<12} | {'Key Signal'}")
    print(hr("-"))

    knn = result["knn"]
    perc = result["perceptron"]
    nb = result["nb"]

    knn_votes = "/".join(str(c) for c in knn["raw_counts"])
    knn_signal = f"k=3 votes [C0/C1/C2]={knn_votes}, nearest d={knn['neighbors'][0]['distance']:.3f}"
    if knn["tie_broken"]:
        knn_signal += " (tie-break)"

    perc_signal = (
        f"score={perc['scores'][perc['winner']]:+.2f} "
        f"(margin {perc['margin']:+.2f} vs C{perc['runner_up']})"
    )

    nb_signal = (
        f"P={nb['probabilities'][nb['winner']]:.1f}%, "
        f"dLogP={nb['margin']:+.2f} vs C{nb['runner_up']}"
    )

    rows = [
        ("K-Nearest Neighbors (k=3)", knn["prediction"], knn_signal),
        ("Linear Perceptron (OvR)", perc["prediction"], perc_signal),
        ("Gaussian Naive Bayes", nb["prediction"], nb_signal),
    ]
    for model_name, pred, signal in rows:
        badge = _status_badge(pred, actual_label, pal)
        print(f"{model_name:<26} | {format_class(pred):<24} | {badge} | {signal}")

    if brief:
        print(hr("="))
        return

    # ------------------------------------------------------------------
    # 1. KNN Glass-Box Explanation
    # ------------------------------------------------------------------
    print()
    print(pal.bold("1) HOW K-NEAREST NEIGHBORS (k=3, Euclidean p=2) DECIDED:"))
    print(
        f"   {'#':<2} | {'Dist':>7} | {'Class':<11} | {'Raw side_b':<18} | "
        f"{'Year':<4} | {'Location':<23} | {'Best':>4} | {'Dur':>3}"
    )
    print("   " + "-" * (WIDTH - 3))
    for nb_item in knn["neighbors"]:
        f_vec = nb_item["features"]
        cls_txt = f"C{nb_item['label']} {CLASS_SHORT[nb_item['label']]}"
        print(
            f"   {nb_item['rank']:<2} | {nb_item['distance']:>7.3f} | {cls_txt:<11} | "
            f"{truncate(nb_item['side_b_raw'], 18):<18} | {nb_item['year']:<4} | "
            f"{truncate(nb_item['location'], 23):<23} | {int(f_vec[5]):>4} | {int(f_vec[7]):>3}"
        )
    vote_summary = (
        f"Vote Tally -> CPP: {knn['raw_counts'][0]}, "
        f"Mindanao: {knn['raw_counts'][1]}, Others: {knn['raw_counts'][2]}"
    )
    if knn["tie_broken"]:
        vote_summary += (
            f"  --> TIE! Popped farthest neighbor(s) -> Winner: {format_class(knn['prediction'], short=True)}"
        )
    else:
        vote_summary += f"  --> Winner: {format_class(knn['prediction'], short=True)}"
    print("   " + pal.dim(vote_summary))

    # ------------------------------------------------------------------
    # 2. Linear Perceptron Glass-Box Explanation
    # ------------------------------------------------------------------
    print()
    print(pal.bold("2) HOW ONE-VS-REST LINEAR PERCEPTRON DECIDED (score = w · x + b):"))
    score_parts = []
    for c in CLASS_IDS:
        tag = f"C{c} {CLASS_SHORT[c]}: {perc['scores'][c]:+.3f}"
        if c == perc["winner"]:
            tag = pal.bold(f"[{tag} <- MAX]")
        score_parts.append(tag)
    print("   Scores: " + "   |   ".join(score_parts))
    print(
        f"   Top features pushing C{perc['winner']} ({CLASS_SHORT[perc['winner']]}) "
        f"over C{perc['runner_up']} ({CLASS_SHORT[perc['runner_up']]}):"
    )
    for d in perc["top_drivers"][:4]:
        direction = f"favors C{perc['winner']}" if d["net_push"] >= 0 else f"favors C{perc['runner_up']}"
        val_fmt = f"{d['value']:.2f}" if isinstance(d["value"], float) else str(d["value"])
        print(
            f"     - {d['key']:<18} (x={val_fmt:>6}) : "
            f"C{perc['winner']} term {d['winner_term']:+8.2f} vs "
            f"C{perc['runner_up']} term {d['runner_up_term']:+8.2f}  "
            f"(net {d['net_push']:+8.2f}, {direction})"
        )

    # ------------------------------------------------------------------
    # 3. Gaussian Naive Bayes Glass-Box Explanation
    # ------------------------------------------------------------------
    print()
    print(pal.bold("3) HOW GAUSSIAN NAIVE BAYES DECIDED (log P(c) + sum log P(x_j | c)):"))
    post_parts = []
    for c in CLASS_IDS:
        tag = f"C{c} {CLASS_SHORT[c]}: {nb['log_posteriors'][c]:+.2f} ({nb['probabilities'][c]:.1f}%)"
        if c == nb["winner"]:
            tag = pal.bold(f"[{tag} <- MAX]")
        post_parts.append(tag)
    print("   Log-Posteriors: " + " | ".join(post_parts))

    w_c, r_c = nb["winner"], nb["runner_up"]
    print(
        f"   {'Feature / Term':<18} | {'Value':>8} | {'LL C0':>9} | {'LL C1':>9} | "
        f"{'LL C2':>9} | {f'dLL (C{w_c}-C{r_c})':>12}"
    )
    print("   " + "-" * (WIDTH - 3))
    prior_delta = nb["log_priors"][w_c] - nb["log_priors"][r_c]
    print(
        f"   {'log P(class)':<18} | {'-':>8} | "
        f"{nb['log_priors'][0]:>+9.2f} | {nb['log_priors'][1]:>+9.2f} | "
        f"{nb['log_priors'][2]:>+9.2f} | {prior_delta:>+12.2f}"
    )
    for row in sorted(nb["feature_breakdown"], key=lambda r: abs(r["delta_vs_runner_up"]), reverse=True):
        val_str = f"{row['value']:.2f}" if isinstance(row["value"], float) else str(row["value"])
        ll = row["ll_by_class"]
        line = (
            f"   {row['key']:<18} | {val_str:>8} | "
            f"{ll[0]:>+9.2f} | {ll[1]:>+9.2f} | {ll[2]:>+9.2f} | "
            f"{row['delta_vs_runner_up']:>+12.2f}"
        )
        if row["key"] == nb["dominant_feature"]["key"]:
            line = pal.yellow(line + "  <-- largest swing")
        print(line)

    if nb["variance_spike_note"]:
        print(
            pal.yellow(
                "   [!] Notice: type_of_violence has near-zero variance (1e-9 smoothing) in CPP,\n"
                "       producing an artificial +9.4 log-likelihood spike whenever type_of_violence = 1!"
            )
        )
    print(hr("="))


def prompt_int(label, min_val, max_val, allow_empty=False):
    while True:
        raw = input(label).strip()
        if allow_empty and raw == "":
            return None
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter an integer between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid integer. Try again.")


def run_presets_menu(bundle, presets, pal, state):
    print()
    print(pal.bold("CURATED 'STORY' TEST CASES (FROM HELD-OUT UNSEEN TEST SET):"))
    print(hr("-"))
    for idx, p in enumerate(presets, start=1):
        t_idx = p["test_idx"]
        actual = bundle["test_data"][t_idx][-1]
        meta = bundle["test_meta"][t_idx]
        print(
            f"  [{idx}] {pal.bold(p['title'])}\n"
            f"      Test #{t_idx} | Actual: {format_class(actual, short=True)} ({meta['side_b_raw']}) | "
            f"{meta['year']} | {truncate(meta['location'], 38)}"
        )
    print("  [a] Run ALL curated cases in sequence")
    print("  [0] Back to main menu")
    choice = input("\nSelect preset [1-" + str(len(presets)) + ", a, 0]: ").strip().lower()
    if choice == "0" or choice == "":
        return
    if choice == "a":
        for idx, p in enumerate(presets, start=1):
            t_idx = p["test_idx"]
            row = bundle["test_data"][t_idx]
            res = predict_all(bundle, row[:-1])
            state["last_features"] = list(row[:-1])
            state["last_label"] = p["title"]
            render_prediction_report(
                res,
                pal,
                actual_label=row[-1],
                meta=bundle["test_meta"][t_idx],
                test_idx=t_idx,
                title=f"PRESET {idx}: {p['title']}",
                takeaway=p["takeaway"],
            )
            if idx < len(presets):
                cont = input("Press Enter for next preset (or 'q' to stop): ").strip().lower()
                if cont == "q":
                    break
        return
    try:
        idx = int(choice)
        if 1 <= idx <= len(presets):
            p = presets[idx - 1]
            t_idx = p["test_idx"]
            row = bundle["test_data"][t_idx]
            res = predict_all(bundle, row[:-1])
            state["last_features"] = list(row[:-1])
            state["last_label"] = f"Preset {idx} (Test #{t_idx})"
            render_prediction_report(
                res,
                pal,
                actual_label=row[-1],
                meta=bundle["test_meta"][t_idx],
                test_idx=t_idx,
                title=f"PRESET {idx}: {p['title']}",
                takeaway=p["takeaway"],
            )
        else:
            print("  Invalid preset number.")
    except ValueError:
        print("  Invalid selection.")


def run_test_explorer(bundle, pal, state):
    test_data = bundle["test_data"]
    test_meta = bundle["test_meta"]
    preds = bundle["test_predictions"]
    max_idx = len(test_data) - 1

    print()
    print(pal.bold(f"HELD-OUT TEST SET EXPLORER (Indices 0..{max_idx})"))
    print("  Enter a number (0-" + str(max_idx) + ") directly, OR a quick filter:")
    print("    m  = random Mindanao event where NB fails & KNN succeeds")
    print("    p  = random CPP event where Perceptron fails & KNN succeeds")
    print("    d  = random event where the 3 models DISAGREE")
    print("    c0 / c1 / c2 = random held-out event of Class 0 / 1 / 2")
    print("    0..881 = inspect that exact test index   |   Enter = back")
    choice = input("\nSelection: ").strip().lower()
    if choice == "":
        return

    picked = None
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx <= max_idx:
            picked = idx
        else:
            print(f"  Index out of range (0..{max_idx}).")
            return
    elif choice == "m":
        pool = [i for i, r in enumerate(test_data) if r[-1] == 1 and preds["nb"][i] != 1 and preds["knn"][i] == 1]
        picked = random.choice(pool) if pool else None
    elif choice == "p":
        pool = [i for i, r in enumerate(test_data) if r[-1] == 0 and preds["perceptron"][i] != 0 and preds["knn"][i] == 0]
        picked = random.choice(pool) if pool else None
    elif choice == "d":
        pool = [
            i
            for i in range(len(test_data))
            if len({preds["nb"][i], preds["perceptron"][i], preds["knn"][i]}) > 1
        ]
        picked = random.choice(pool) if pool else None
    elif choice in ("c0", "c1", "c2"):
        target_cls = int(choice[1])
        pool = [i for i, r in enumerate(test_data) if r[-1] == target_cls]
        picked = random.choice(pool) if pool else None
    else:
        print("  Unrecognized option.")
        return

    if picked is None:
        print("  No matching test samples found.")
        return

    row = test_data[picked]
    res = predict_all(bundle, row[:-1])
    state["last_features"] = list(row[:-1])
    state["last_label"] = f"Test #{picked}"
    render_prediction_report(
        res,
        pal,
        actual_label=row[-1],
        meta=test_meta[picked],
        test_idx=picked,
    )


def run_random_sample(bundle, pal, state):
    while True:
        idx = random.randrange(len(bundle["test_data"]))
        row = bundle["test_data"][idx]
        res = predict_all(bundle, row[:-1])
        state["last_features"] = list(row[:-1])
        state["last_label"] = f"Random Test #{idx}"
        render_prediction_report(
            res,
            pal,
            actual_label=row[-1],
            meta=bundle["test_meta"][idx],
            test_idx=idx,
            title=f"RANDOM UNSEEN TEST SAMPLE #{idx}",
        )
        again = input("Press [r] for another random test sample, or Enter for menu: ").strip().lower()
        if again != "r":
            break


def run_custom_input(bundle, presets, pal, state):
    print()
    print(pal.bold("CUSTOM 'WHAT-IF' TEST INPUT"))
    print("  [1] Tweak a starting sample (press Enter on any feature to keep its value)")
    print("  [2] Enter all 8 features from scratch")
    print("  [0] Back")
    mode = input("\nChoose mode [1/2/0]: ").strip()
    if mode not in ("1", "2"):
        return

    base_features = None
    base_label = None
    prev_result = None

    if mode == "1":
        if state.get("last_features") is not None:
            base_features = list(state["last_features"])
            base_label = state.get("last_label", "last input")
        elif presets:
            t_idx = presets[1]["test_idx"] if len(presets) > 1 else presets[0]["test_idx"]
            base_features = list(bundle["test_data"][t_idx][:-1])
            base_label = f"Test #{t_idx}"
        else:
            base_features = [124.25, 7.20, 2, 3, 0, 5, 1, 0]
            base_label = "Default Mindanao skirmish"
        prev_result = predict_all(bundle, base_features)
        print(f"\nStarting from: {pal.bold(base_label)}")
        print(pal.dim("Press Enter to keep the current value, or type a new number.\n"))
    else:
        print("\nEnter each of the 8 numerical features:\n")

    new_features = []
    for idx, spec in enumerate(FEATURE_SPECS):
        dtype = spec["dtype"]
        default_val = base_features[idx] if base_features is not None else None
        while True:
            if default_val is not None:
                prompt = f"  [{idx}] {spec['label']} ({spec['key']}) [current={default_val}]: "
            else:
                prompt = f"  [{idx}] {spec['label']} ({spec['key']}) ({spec['hint']}): "
            raw = input(prompt).strip()
            if raw == "" and default_val is not None:
                new_features.append(dtype(default_val))
                break
            try:
                val = dtype(raw)
                if spec["key"] == "type_of_violence" and val not in (1, 2, 3):
                    print("      Tip: UCDP type_of_violence is normally 1, 2, or 3 (accepting anyway).")
                new_features.append(val)
                break
            except ValueError:
                print(f"      Please enter a valid {dtype.__name__}. ({spec['hint']})")

    res = predict_all(bundle, new_features)
    state["last_features"] = list(new_features)
    state["last_label"] = "Custom What-If Input"

    # If tweaking, print a crisp Before -> After comparison right above the full report
    if prev_result is not None:
        changes = []
        for idx, spec in enumerate(FEATURE_SPECS):
            if new_features[idx] != base_features[idx]:
                changes.append(f"{spec['key']}: {base_features[idx]} -> {new_features[idx]}")
        print()
        print(pal.magenta("WHAT-IF DIFF (" + (", ".join(changes) if changes else "no feature changes") + "):"))
        for name, key in (("KNN (k=3)", "knn"), ("Perceptron", "perceptron"), ("Naive Bayes", "nb")):
            old_p = prev_result["predictions"][key]
            new_p = res["predictions"][key]
            arrow = (
                pal.yellow(f"C{old_p} ({CLASS_SHORT[old_p]})  ==>  C{new_p} ({CLASS_SHORT[new_p]})  [FLIPPED!]")
                if old_p != new_p
                else pal.dim(f"C{old_p} ({CLASS_SHORT[old_p]})  (unchanged)")
            )
            print(f"  {name:<14}: {arrow}")

    render_prediction_report(
        res,
        pal,
        title="CUSTOM 'WHAT-IF' PREDICTION",
    )


def run_interactive(bundle, load_ms, pal):
    presets = build_curated_presets(bundle)
    state = {"last_features": None, "last_label": None}
    print_banner(bundle, load_ms, pal)

    while True:
        print()
        print(pal.bold("DEMO MENU"))
        print("  [1] Curated 'Story' Test Cases    (4-5 held-out events proving report findings)")
        print("  [2] Held-Out Test Set Explorer    (pick index 0..881 or filter disagreements)")
        print("  [3] Random Unseen Test Sample     (unbiased live draw from test set)")
        print("  [4] Custom 'What-If' Input        (enter 8 features or tweak last sample)")
        print("  [5] Pre-Trained Model Scorecard   (accuracy, balanced acc, macro F1)")
        print("  [0] Exit")

        try:
            choice = input("\nChoose an option [1-5, 0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting demo.")
            break

        if choice == "1":
            run_presets_menu(bundle, presets, pal, state)
        elif choice == "2":
            run_test_explorer(bundle, pal, state)
        elif choice == "3":
            run_random_sample(bundle, pal, state)
        elif choice == "4":
            run_custom_input(bundle, presets, pal, state)
        elif choice == "5":
            print_scorecard(bundle, pal)
        elif choice == "0" or choice.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("  Please choose 1, 2, 3, 4, 5, or 0.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive CLI demo for the 3 pre-trained Philippine conflict classification models."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained_models.pkl")
    parser.add_argument("--preset", type=int, help="Run a curated preset (1-5) non-interactively")
    parser.add_argument("--test-index", type=int, help="Run a specific held-out test index (0-881)")
    parser.add_argument("--random", action="store_true", help="Run a random held-out test sample")
    parser.add_argument(
        "--custom",
        nargs=8,
        metavar=("LON", "LAT", "DA", "DB", "DC", "BEST", "TOV", "DUR"),
        help="Run a custom 8-feature vector non-interactively",
    )
    parser.add_argument("--summary", action="store_true", help="Print pre-trained model scorecard and exit")
    parser.add_argument("--brief", action="store_true", help="Show summary table only (skip per-model math)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args()


def main():
    args = parse_args()
    use_color = (not args.no_color) and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    pal = Palette(enabled=use_color)

    try:
        bundle, load_ms = load_bundle(args.model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Non-interactive one-shot modes (handy for quick checks or scripted demos)
    if args.summary:
        print_banner(bundle, load_ms, pal)
        print_scorecard(bundle, pal)
        return

    if args.preset is not None:
        presets = build_curated_presets(bundle)
        if not (1 <= args.preset <= len(presets)):
            print(f"Error: --preset must be between 1 and {len(presets)}.", file=sys.stderr)
            sys.exit(1)
        p = presets[args.preset - 1]
        t_idx = p["test_idx"]
        row = bundle["test_data"][t_idx]
        print_banner(bundle, load_ms, pal)
        render_prediction_report(
            predict_all(bundle, row[:-1]),
            pal,
            actual_label=row[-1],
            meta=bundle["test_meta"][t_idx],
            test_idx=t_idx,
            title=f"PRESET {args.preset}: {p['title']}",
            takeaway=p["takeaway"],
            brief=args.brief,
        )
        return

    if args.test_index is not None:
        max_idx = len(bundle["test_data"]) - 1
        if not (0 <= args.test_index <= max_idx):
            print(f"Error: --test-index must be between 0 and {max_idx}.", file=sys.stderr)
            sys.exit(1)
        row = bundle["test_data"][args.test_index]
        print_banner(bundle, load_ms, pal)
        render_prediction_report(
            predict_all(bundle, row[:-1]),
            pal,
            actual_label=row[-1],
            meta=bundle["test_meta"][args.test_index],
            test_idx=args.test_index,
            brief=args.brief,
        )
        return

    if args.random:
        idx = random.randrange(len(bundle["test_data"]))
        row = bundle["test_data"][idx]
        print_banner(bundle, load_ms, pal)
        render_prediction_report(
            predict_all(bundle, row[:-1]),
            pal,
            actual_label=row[-1],
            meta=bundle["test_meta"][idx],
            test_idx=idx,
            title=f"RANDOM UNSEEN TEST SAMPLE #{idx}",
            brief=args.brief,
        )
        return

    if args.custom is not None:
        try:
            feats = [
                float(args.custom[0]),
                float(args.custom[1]),
                int(args.custom[2]),
                int(args.custom[3]),
                int(args.custom[4]),
                int(args.custom[5]),
                int(args.custom[6]),
                int(args.custom[7]),
            ]
        except ValueError as exc:
            print(f"Error parsing --custom values: {exc}", file=sys.stderr)
            sys.exit(1)
        print_banner(bundle, load_ms, pal)
        render_prediction_report(
            predict_all(bundle, feats),
            pal,
            title="CUSTOM 'WHAT-IF' PREDICTION",
            brief=args.brief,
        )
        return

    # Default: interactive menu
    run_interactive(bundle, load_ms, pal)


if __name__ == "__main__":
    main()
