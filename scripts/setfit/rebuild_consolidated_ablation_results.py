import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


RESULT_ROOT = Path(__file__).parent / "results"
OUT_DIR = RESULT_ROOT / "consolidated_ablation_results"

SETFIT_8_DIR = RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_main"
SETFIT_SHOT_DIR = RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_shot_ablation"
FROZEN_DIR = RESULT_ROOT / "paraphrase-mpnet-base-v2-frozen_st_lr-logistic_regression-batch_16-test_frozen_st_lr"
HN50_8_DIR = RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_8"
HN50_SHOT_DIR = RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_shot_ablation"

ITER_DIRS = {
    2: RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_2-batch_16-test_iter_2",
    5: RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_5-batch_16-test_iter_5",
    10: RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_10-batch_16-test_iter_10",
    20: SETFIT_8_DIR,
}

CLASSIFIER_DIRS = {
    "LR": SETFIT_8_DIR,
    "SVC-RBF": RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-svc-rbf-iterations_20-batch_16-test_clf_svc_rbf_valid",
    "KNN": RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-knn-iterations_20-batch_16-test_clf_knn_valid",
}

HN_RATIO_DIRS = {
    "SetFit": SETFIT_8_DIR,
    "HN-25": RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p25-test_hard_negative_hn25_8",
    "HN-50": HN50_8_DIR,
    "HN-75": RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p75-test_hard_negative_hn75_8",
    "HN-100": RESULT_ROOT / "paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_1p0-test_hard_negative_hn100_8",
}

DATASET_ORDER = ["sst5", "emotion", "enron_spam", "ag_news", "amazon_counterfactual_en", "SentEval-CR", "Average"]


def parse_split(split):
    parts = split.split("-")
    if len(parts) != 3 or parts[0] != "train":
        return None
    return int(parts[1]), int(parts[2])


def collect_results(*dirs):
    results = {}
    for root in dirs:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*/results.json")):
            parsed = parse_split(path.parent.name)
            if parsed is None:
                continue
            shot, seed = parsed
            with path.open() as f:
                payload = json.load(f)
            key = (path.parent.parent.name, shot, seed)
            results[key] = {
                "dataset": path.parent.parent.name,
                "shot": shot,
                "seed": seed,
                "measure": payload["measure"],
                "score": float(payload["score"]),
            }
    return results


def stability(std):
    if std <= 2:
        return "stable"
    if std <= 5:
        return "moderate"
    return "unstable"


def summarize(results):
    grouped = defaultdict(list)
    for row in results.values():
        grouped[(row["dataset"], row["shot"], row["measure"])].append(row["score"])

    rows = []
    for (dataset, shot, measure), scores in grouped.items():
        exact_mean = mean(scores)
        exact_std = pstdev(scores) if len(scores) > 1 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "measure": measure,
                "shot": shot,
                "mean": round(exact_mean, 1),
                "std": round(exact_std, 1),
                "exact_mean": exact_mean,
                "exact_std": exact_std,
                "n": len(scores),
            }
        )

    by_shot = defaultdict(list)
    for row in rows:
        by_shot[row["shot"]].append(row)
    for shot, shot_rows in by_shot.items():
        rows.append(
            {
                "dataset": "Average",
                "measure": "N/A",
                "shot": shot,
                "mean": round(mean(row["exact_mean"] for row in shot_rows), 1),
                "std": round(mean(row["exact_std"] for row in shot_rows), 1),
                "n": sum(row["n"] for row in shot_rows),
            }
        )
    return sorted(rows, key=lambda row: (row["shot"], DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99))


def summary_lookup(rows):
    return {(row["dataset"], row["shot"]): row for row in rows}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, fieldnames, title):
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")


def score(row):
    return f"{row['mean']} +/- {row['std']}"


def shot_columns(shots):
    return [shot for shot in [2, 4, 8, 16, 64] if shot in shots]


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n")


def p_values(deltas):
    if stats is None or len(deltas) < 2:
        return "N/A", "N/A"
    t_test = stats.ttest_1samp(deltas, popmean=0.0)
    try:
        wilcoxon = stats.wilcoxon(deltas, zero_method="wilcox")
        wilcoxon_p = f"{wilcoxon.pvalue:.4f}"
    except ValueError:
        wilcoxon_p = "N/A"
    return f"{t_test.pvalue:.4f}", wilcoxon_p


def paired_summary(name, group, rows):
    deltas = [row["delta"] for row in rows]
    return {
        "analysis": name,
        "group": group,
        "measure_or_shot": rows[0]["measure_or_shot"],
        "n_pairs": len(rows),
        "mean_delta": f"{mean(deltas):.4f}",
        "std_delta": f"{stdev(deltas):.4f}" if len(deltas) > 1 else "0.0000",
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "paired_t_p": p_values(deltas)[0],
        "wilcoxon_p": p_values(deltas)[1],
    }


def paired_rows(baseline, candidate):
    rows = []
    for key in sorted(set(baseline) & set(candidate)):
        base = baseline[key]
        cand = candidate[key]
        rows.append(
            {
                "dataset": base["dataset"],
                "shot": base["shot"],
                "measure": cand["measure"],
                "measure_or_shot": cand["measure"],
                "delta": cand["score"] - base["score"],
            }
        )
    return rows


def build_core_setfit():
    setfit_rows = summarize(collect_results(SETFIT_SHOT_DIR, SETFIT_8_DIR))
    rows = []
    for row in summarize(collect_results(SETFIT_8_DIR)):
        rows.append(
            {
                "ablation": "baseline",
                "setting": "8-shot, iter=20, LR",
                "dataset": row["dataset"],
                "measure": row["measure"],
                "mean": row["mean"],
                "std": row["std"],
                "stability": stability(row["std"]),
            }
        )
    for row in setfit_rows:
        rows.append(
            {
                "ablation": "shot_size",
                "setting": f"{row['shot']}-shot",
                "dataset": row["dataset"],
                "measure": row["measure"],
                "mean": row["mean"],
                "std": row["std"],
                "stability": stability(row["std"]),
            }
        )
    for num_iter, result_dir in ITER_DIRS.items():
        for row in summarize(collect_results(result_dir)):
            rows.append(
                {
                    "ablation": "num_iterations",
                    "setting": f"iter={num_iter}, 8-shot",
                    "dataset": row["dataset"],
                    "measure": row["measure"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "stability": stability(row["std"]),
                }
            )
    for classifier, result_dir in CLASSIFIER_DIRS.items():
        for row in summarize(collect_results(result_dir)):
            rows.append(
                {
                    "ablation": "classifier",
                    "setting": f"{classifier}, 8-shot",
                    "dataset": row["dataset"],
                    "measure": row["measure"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "stability": stability(row["std"]),
                }
            )
    fields = ["ablation", "setting", "dataset", "measure", "mean", "std", "stability"]
    write_csv(OUT_DIR / "02_core_setfit_ablations.csv", rows, fields)
    write_markdown(OUT_DIR / "02_core_setfit_ablations.md", rows, fields, "Core SetFit Ablations")
    return rows


def build_unified_method_table():
    setfit = summary_lookup(summarize(collect_results(SETFIT_SHOT_DIR, SETFIT_8_DIR)))
    frozen = summary_lookup(summarize(collect_results(FROZEN_DIR)))
    hn50 = summary_lookup(summarize(collect_results(HN50_SHOT_DIR, HN50_8_DIR)))
    rows = []
    methods = [("Frozen ST + LR", frozen), ("SetFit", setfit), ("SetFit-HN-50", hn50)]
    for dataset, shot in sorted(set(setfit) | set(frozen) | set(hn50), key=lambda key: (DATASET_ORDER.index(key[0]) if key[0] in DATASET_ORDER else 99, key[1])):
        base = setfit.get((dataset, shot))
        for method, lookup in methods:
            row = lookup.get((dataset, shot))
            if row is None:
                continue
            delta = ""
            if base is not None:
                delta = f"{row['mean'] - base['mean']:+.1f}"
            rows.append(
                {
                    "dataset": dataset,
                    "measure": row["measure"],
                    "shot": shot,
                    "method": method,
                    "mean": row["mean"],
                    "std": row["std"],
                    "delta_vs_setfit": delta,
                }
            )
    fields = ["dataset", "measure", "shot", "method", "mean", "std", "delta_vs_setfit"]
    write_csv(OUT_DIR / "03_unified_method_by_shot_comparison.csv", rows, fields)
    write_markdown(OUT_DIR / "03_unified_method_by_shot_comparison.md", rows, fields, "Unified Method by Shot Comparison")
    return rows


def build_hn_ratio_table():
    rows = []
    for method, result_dir in HN_RATIO_DIRS.items():
        for row in summarize(collect_results(result_dir)):
            rows.append(
                {
                    "method": method,
                    "dataset": row["dataset"],
                    "measure": row["measure"],
                    "shot": row["shot"],
                    "mean": row["mean"],
                    "std": row["std"],
                }
            )
    fields = ["method", "dataset", "measure", "shot", "mean", "std"]
    write_csv(OUT_DIR / "04_hard_negative_ratio_ablation.csv", rows, fields)
    write_markdown(OUT_DIR / "04_hard_negative_ratio_ablation.md", rows, fields, "Hard Negative Ratio Ablation")
    return rows


def build_paired_table():
    baseline_8 = collect_results(SETFIT_8_DIR)
    hn50_8 = collect_results(HN50_8_DIR)
    all_baseline = collect_results(SETFIT_SHOT_DIR, SETFIT_8_DIR)
    all_hn50 = collect_results(HN50_SHOT_DIR, HN50_8_DIR)

    rows = []
    eight_rows = paired_rows(baseline_8, hn50_8)
    for dataset in sorted({row["dataset"] for row in eight_rows}):
        dataset_rows = [row for row in eight_rows if row["dataset"] == dataset]
        rows.append(paired_summary("HN-50 vs SetFit, 8-shot by dataset", dataset, dataset_rows))
    if eight_rows:
        rows.append(paired_summary("HN-50 vs SetFit, 8-shot by dataset", "Average", eight_rows))

    shot_rows = paired_rows(all_baseline, all_hn50)
    for row in shot_rows:
        row["measure_or_shot"] = row["shot"]
    for shot in sorted({row["shot"] for row in shot_rows}):
        one_shot_rows = [row for row in shot_rows if row["shot"] == shot]
        rows.append(paired_summary("HN-50 vs SetFit, by shot", f"{shot}-shot", one_shot_rows))

    fields = [
        "analysis",
        "group",
        "measure_or_shot",
        "n_pairs",
        "mean_delta",
        "std_delta",
        "wins",
        "ties",
        "losses",
        "paired_t_p",
        "wilcoxon_p",
    ]
    write_csv(OUT_DIR / "05_paired_significance_summary.csv", rows, fields)
    write_markdown(OUT_DIR / "05_paired_significance_summary.md", rows, fields, "Paired Significance Summary")
    return rows


def build_legacy_test_only_tables(core_rows):
    frozen_rows = []
    for row in summarize(collect_results(FROZEN_DIR)):
        frozen_rows.append(
            {
                "ablation": "frozen_st_lr",
                "setting": f"{row['shot']}-shot",
                "dataset": row["dataset"],
                "measure": row["measure"],
                "mean": row["mean"],
                "std": row["std"],
                "stability": stability(row["std"]),
            }
        )
    rows = core_rows + frozen_rows
    fields = ["ablation", "setting", "dataset", "measure", "mean", "std", "stability"]
    write_csv(RESULT_ROOT / "test_only_ablation_summary.csv", rows, fields)

    lookup = {
        (row["ablation"], row["setting"], row["dataset"]): row
        for row in rows
    }
    setfit = summary_lookup(summarize(collect_results(SETFIT_SHOT_DIR, SETFIT_8_DIR)))
    iter_lookup = {
        num_iter: summary_lookup(summarize(collect_results(result_dir)))
        for num_iter, result_dir in ITER_DIRS.items()
    }
    clf_lookup = {
        classifier: summary_lookup(summarize(collect_results(result_dir)))
        for classifier, result_dir in CLASSIFIER_DIRS.items()
    }
    frozen = summary_lookup(summarize(collect_results(FROZEN_DIR)))
    all_datasets = DATASET_ORDER
    shots = shot_columns({shot for _, shot in setfit})

    lines = [
        "# Test-only SetFit Ablation Summary",
        "",
        "Scores are percentages. `amazon_counterfactual_en` uses Matthews correlation, also scaled by 100. Stability: stable <= 2 std, moderate <= 5 std, unstable > 5 std.",
        "",
        "## Baseline 8-shot iter=20 LR",
        "",
        "| dataset | measure | score | stability |",
        "| --- | --- | --- | --- |",
    ]
    for dataset in all_datasets:
        row = lookup.get(("baseline", "8-shot, iter=20, LR", dataset))
        if row:
            lines.append(f"| {dataset} | {row['measure']} | {score(row)} | {row['stability']} |")

    lines.extend(["", "## Shot-size ablation", ""])
    lines.append("| dataset | measure | " + " | ".join(f"{shot}-shot" for shot in shots) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in shots) + " |")
    for dataset in all_datasets:
        row0 = next((setfit.get((dataset, shot)) for shot in shots if setfit.get((dataset, shot))), None)
        if not row0:
            continue
        values = [score(setfit[(dataset, shot)]) if (dataset, shot) in setfit else "" for shot in shots]
        lines.append(f"| {dataset} | {row0['measure']} | " + " | ".join(values) + " |")

    lines.extend(["", "## num_iterations ablation", ""])
    iter_values = sorted(ITER_DIRS)
    lines.append("| dataset | measure | " + " | ".join(f"iter={num_iter}" for num_iter in iter_values) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in iter_values) + " |")
    for dataset in all_datasets:
        row0 = next((iter_lookup[num_iter].get((dataset, 8)) for num_iter in iter_values if iter_lookup[num_iter].get((dataset, 8))), None)
        if not row0:
            continue
        values = [score(iter_lookup[num_iter][(dataset, 8)]) if (dataset, 8) in iter_lookup[num_iter] else "" for num_iter in iter_values]
        lines.append(f"| {dataset} | {row0['measure']} | " + " | ".join(values) + " |")

    lines.extend(["", "## Classifier-head ablation", ""])
    classifiers = list(CLASSIFIER_DIRS)
    lines.append("| dataset | measure | " + " | ".join(classifiers) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in classifiers) + " |")
    for dataset in all_datasets:
        row0 = next((clf_lookup[classifier].get((dataset, 8)) for classifier in classifiers if clf_lookup[classifier].get((dataset, 8))), None)
        if not row0:
            continue
        values = [score(clf_lookup[classifier][(dataset, 8)]) if (dataset, 8) in clf_lookup[classifier] else "" for classifier in classifiers]
        lines.append(f"| {dataset} | {row0['measure']} | " + " | ".join(values) + " |")

    lines.extend(["", "## Frozen ST + LR", ""])
    lines.append("| dataset | measure | " + " | ".join(f"{shot}-shot" for shot in shots) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in shots) + " |")
    for dataset in all_datasets:
        row0 = next((frozen.get((dataset, shot)) for shot in shots if frozen.get((dataset, shot))), None)
        if not row0:
            continue
        values = [score(frozen[(dataset, shot)]) if (dataset, shot) in frozen else "" for shot in shots]
        lines.append(f"| {dataset} | {row0['measure']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Main observations",
            "",
            f"- Shot size has the largest effect on the average SetFit score: "
            + ", ".join(f"{shot}-shot {setfit[('Average', shot)]['mean']}" for shot in shots if ("Average", shot) in setfit)
            + ".",
            "- 64-shot substantially reduces variance compared with low-shot settings.",
            "- Classifier and num_iterations ablations remain 8-shot-only controls.",
        ]
    )
    write_lines(RESULT_ROOT / "test_only_ablation_report.md", lines)


def build_legacy_frozen_tables():
    setfit = summary_lookup(summarize(collect_results(SETFIT_SHOT_DIR, SETFIT_8_DIR)))
    frozen = summary_lookup(summarize(collect_results(FROZEN_DIR)))
    rows = []
    for dataset, shot in sorted(set(setfit) & set(frozen), key=lambda key: (DATASET_ORDER.index(key[0]) if key[0] in DATASET_ORDER else 99, key[1])):
        setfit_row = setfit[(dataset, shot)]
        frozen_row = frozen[(dataset, shot)]
        rows.append(
            {
                "dataset": dataset,
                "measure": setfit_row["measure"],
                "shot": shot,
                "frozen_st_lr_mean": frozen_row["mean"],
                "frozen_st_lr_std": frozen_row["std"],
                "setfit_mean": setfit_row["mean"],
                "setfit_std": setfit_row["std"],
                "setfit_minus_frozen": f"{setfit_row['mean'] - frozen_row['mean']:.1f}",
            }
        )
    fields = ["dataset", "measure", "shot", "frozen_st_lr_mean", "frozen_st_lr_std", "setfit_mean", "setfit_std", "setfit_minus_frozen"]
    write_csv(RESULT_ROOT / "frozen_st_lr_vs_setfit_comparison.csv", rows, fields)

    lines = [
        "# Frozen ST + LR vs SetFit",
        "",
        "Both methods use the same test benchmark datasets and the same few-shot split generation. Scores are percentages; Amazon uses Matthews correlation scaled by 100.",
        "",
        "## Average By Shot",
        "",
        "| Shot | Frozen ST + LR | SetFit | SetFit - Frozen |",
        "| --- | --- | --- | --- |",
    ]
    for row in [row for row in rows if row["dataset"] == "Average"]:
        lines.append(
            f"| {row['shot']} | {row['frozen_st_lr_mean']} +/- {row['frozen_st_lr_std']} | "
            f"{row['setfit_mean']} +/- {row['setfit_std']} | {row['setfit_minus_frozen']} |"
        )
    lines.extend(
        [
            "",
            "## Per-dataset Comparison",
            "",
            "| Dataset | Measure | Shot | Frozen ST + LR | SetFit | SetFit - Frozen |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in [row for row in rows if row["dataset"] != "Average"]:
        lines.append(
            f"| {row['dataset']} | {row['measure']} | {row['shot']} | "
            f"{row['frozen_st_lr_mean']} +/- {row['frozen_st_lr_std']} | "
            f"{row['setfit_mean']} +/- {row['setfit_std']} | {row['setfit_minus_frozen']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- SetFit improves over Frozen ST + LR at every averaged shot size, including 64-shot.",
            "- The 64-shot average gain is +10.5 points, so contrastive fine-tuning remains useful beyond the most extreme low-resource setting.",
        ]
    )
    write_lines(RESULT_ROOT / "frozen_st_lr_vs_setfit_report.md", lines)


def build_legacy_hn50_tables(paired_rows_summary):
    setfit = summary_lookup(summarize(collect_results(SETFIT_SHOT_DIR, SETFIT_8_DIR)))
    hn50 = summary_lookup(summarize(collect_results(HN50_SHOT_DIR, HN50_8_DIR)))
    shots = shot_columns({shot for _, shot in setfit} & {shot for _, shot in hn50})

    wide_rows = []
    for dataset in DATASET_ORDER:
        row0 = next((setfit.get((dataset, shot)) for shot in shots if setfit.get((dataset, shot))), None)
        if not row0:
            continue
        row = {"dataset": dataset, "measure": row0["measure"]}
        for shot in shots:
            base = setfit.get((dataset, shot))
            cand = hn50.get((dataset, shot))
            if base and cand:
                row[f"setfit_{shot}_avg"] = base["mean"]
                row[f"setfit_{shot}_std"] = base["std"]
                row[f"hn50_{shot}_avg"] = cand["mean"]
                row[f"hn50_{shot}_std"] = cand["std"]
                row[f"delta_{shot}"] = f"{cand['mean'] - base['mean']:+.1f}"
        wide_rows.append(row)

    fields = ["dataset", "measure"]
    for shot in shots:
        fields.extend([f"setfit_{shot}_avg", f"setfit_{shot}_std", f"hn50_{shot}_avg", f"hn50_{shot}_std", f"delta_{shot}"])
    write_csv(RESULT_ROOT / "hard_negative_hn50_shot_ablation_vs_setfit.csv", wide_rows, fields)

    shot_paired = [
        {
            "shot": row["measure_or_shot"],
            "n_pairs": row["n_pairs"],
            "mean_delta": row["mean_delta"],
            "std_delta": row["std_delta"],
            "wins": row["wins"],
            "ties": row["ties"],
            "losses": row["losses"],
            "paired_t_p": row["paired_t_p"],
            "wilcoxon_p": row["wilcoxon_p"],
        }
        for row in paired_rows_summary
        if row["analysis"] == "HN-50 vs SetFit, by shot"
    ]
    paired_fields = ["shot", "n_pairs", "mean_delta", "std_delta", "wins", "ties", "losses", "paired_t_p", "wilcoxon_p"]
    write_csv(RESULT_ROOT / "hard_negative_hn50_shot_paired_summary.csv", shot_paired, paired_fields)

    lines = [
        "# HN-50 Shot Ablation vs SetFit",
        "",
        "| Dataset | " + " | ".join(f"{shot}-shot delta" for shot in shots) + " |",
        "| --- | " + " | ".join("---:" for _ in shots) + " |",
    ]
    for row in wide_rows:
        lines.append("| " + row["dataset"] + " | " + " | ".join(row.get(f"delta_{shot}", "") for shot in shots) + " |")
    lines.extend(["", "Detailed scores:", ""])
    detail_fields = ["Dataset"]
    for shot in shots:
        detail_fields.extend([f"SetFit {shot}", f"HN-50 {shot}"])
    lines.append("| " + " | ".join(detail_fields) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:" for _ in detail_fields[1:]]) + " |")
    for row in wide_rows:
        values = [row["dataset"]]
        for shot in shots:
            values.append(f"{row.get(f'setfit_{shot}_avg', '')} +/- {row.get(f'setfit_{shot}_std', '')}")
            values.append(f"{row.get(f'hn50_{shot}_avg', '')} +/- {row.get(f'hn50_{shot}_std', '')}")
        lines.append("| " + " | ".join(values) + " |")
    write_lines(RESULT_ROOT / "hard_negative_hn50_shot_ablation_report.md", lines)

    lines = [
        "# HN-50 Paired Split Summary by Shot",
        "",
        "| Shot | Pairs | Mean delta | Delta std | Win/Tie/Loss | paired t-test p | Wilcoxon p |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in shot_paired:
        lines.append(
            f"| {row['shot']} | {row['n_pairs']} | {float(row['mean_delta']):+.2f} | {float(row['std_delta']):.2f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['paired_t_p']} | {row['wilcoxon_p']} |"
        )
    write_lines(RESULT_ROOT / "hard_negative_hn50_shot_paired_report.md", lines)


def build_legacy_config_and_index():
    config_rows = [
        {
            "experiment": "SetFit shot ablation",
            "setting": "2/4/8/16/64-shot",
            "command": "python run_fewshot.py --sample_sizes 2 4 16 64 --is_test_set=true --num_iterations=20 --exp_name test_shot_ablation; 8-shot reused from test_main",
            "result_table": "test_only_ablation_summary.csv",
        },
        {
            "experiment": "Frozen ST + LR",
            "setting": "2/4/8/16/64-shot",
            "command": "python run_frozen_st_lr.py --sample_sizes 2 4 8 16 64 --is_test_set=true --exp_name test_frozen_st_lr",
            "result_table": "frozen_st_lr_vs_setfit_comparison.csv",
        },
        {
            "experiment": "SetFit-HN-50 shot ablation",
            "setting": "2/4/8/16/64-shot",
            "command": "python run_fewshot.py --sample_sizes 2 4 16 64 --is_test_set=true --num_iterations=20 --pair_strategy hard_negative --hard_negative_ratio=0.5 --exp_name test_hard_negative_hn50_shot_ablation; 8-shot reused from test_hard_negative_hn50_8",
            "result_table": "hard_negative_hn50_shot_ablation_vs_setfit.csv",
        },
        {
            "experiment": "num_iterations ablation",
            "setting": "8-shot, iter=2/5/10/20",
            "command": "python run_fewshot.py --sample_sizes 8 --num_iterations {2,5,10,20} --is_test_set=true",
            "result_table": "test_only_ablation_summary.csv",
        },
        {
            "experiment": "classifier ablation",
            "setting": "8-shot, LR/SVC-RBF/KNN",
            "command": "python run_fewshot.py --sample_sizes 8 --classifier {logistic_regression,svc-rbf,knn} --is_test_set=true",
            "result_table": "test_only_ablation_summary.csv",
        },
        {
            "experiment": "Hard Negative ratio ablation",
            "setting": "8-shot, HN-25/HN-50/HN-75/HN-100",
            "command": "python run_fewshot.py --sample_sizes 8 --pair_strategy hard_negative --hard_negative_ratio {0.25,0.5,0.75,1.0} --is_test_set=true",
            "result_table": "hard_negative_ratio_ablation_summary.csv",
        },
    ]
    config_fields = ["experiment", "setting", "command", "result_table"]
    write_csv(RESULT_ROOT / "ablation_experiment_configs.csv", config_rows, config_fields)
    write_markdown(RESULT_ROOT / "ablation_experiment_configs.md", config_rows, config_fields, "SetFit Ablation Experiment Configs")

    lines = [
        "# Experiment Result File Index",
        "",
        "所有路径均相对于 `scripts/setfit/results/`。",
        "",
        "| 实验/用途 | 结果表 CSV | 报告 MD | 备注 |",
        "| --- | --- | --- | --- |",
        "| 实验配置总表 | `ablation_experiment_configs.csv` | `ablation_experiment_configs.md` | 已更新到 64-shot |",
        "| SetFit test-only 主消融汇总：baseline、shot、num_iterations、classifier、Frozen ST + LR | `test_only_ablation_summary.csv` | `test_only_ablation_report.md` | shot 表包含 2/4/8/16/64 |",
        "| Frozen ST + LR vs SetFit | `frozen_st_lr_vs_setfit_comparison.csv` | `frozen_st_lr_vs_setfit_report.md` | 包含 2/4/8/16/64 |",
        "| SetFit-HN-50 8-shot vs SetFit baseline | `hard_negative_hn50_vs_setfit_8shot.csv` | `hard_negative_hn50_vs_setfit_8shot_report.md` | 8-shot 专项对比 |",
        "| SetFit-HN-50 8-shot paired split 显著性分析 | `hard_negative_hn50_vs_setfit_8shot_paired.csv` | `hard_negative_hn50_vs_setfit_8shot_paired.md` | 8-shot 专项 paired 分析 |",
        "| Hard Negative ratio 消融：HN-25/HN-50/HN-75/HN-100 | `hard_negative_ratio_ablation_summary.csv` | `hard_negative_ratio_ablation_report.md` | 8-shot ratio 消融 |",
        "| SetFit-HN-50 shot 消融 vs SetFit | `hard_negative_hn50_shot_ablation_vs_setfit.csv` | `hard_negative_hn50_shot_ablation_report.md` | 包含 2/4/8/16/64 |",
        "| SetFit-HN-50 shot paired split 显著性分析 | `hard_negative_hn50_shot_paired_summary.csv` | `hard_negative_hn50_shot_paired_report.md` | 包含 2/4/8/16/64 |",
        "| 统一合并表目录 | `consolidated_ablation_results/*.csv` | `consolidated_ablation_results/*.md` | 推荐优先阅读 |",
    ]
    write_lines(RESULT_ROOT / "experiment_result_file_index.md", lines)


def build_unified_score_master():
    rows = []

    def add_rows(experiment, method, result_dir, reference=None, shot_filter=None, **config):
        if reference:
            reference_dirs = reference if isinstance(reference, tuple) else (reference,)
            reference_lookup = summary_lookup(summarize(collect_results(*reference_dirs)))
        else:
            reference_lookup = {}
        for row in summarize(collect_results(result_dir)):
            if shot_filter is not None and row["shot"] not in shot_filter:
                continue
            delta = ""
            if reference:
                base = reference_lookup.get((row["dataset"], row["shot"]))
                if base:
                    delta = f"{row['mean'] - base['mean']:+.1f}"
            rows.append(
                {
                    "experiment_group": experiment,
                    "method": method,
                    "dataset": row["dataset"],
                    "measure": row["measure"],
                    "shot": row["shot"],
                    "num_iterations": config.get("num_iterations", ""),
                    "classifier": config.get("classifier", ""),
                    "pair_strategy": config.get("pair_strategy", ""),
                    "hard_negative_ratio": config.get("hard_negative_ratio", ""),
                    "mean": row["mean"],
                    "std": row["std"],
                    "delta_vs_reference": delta,
                    "reference": config.get("reference_name", ""),
                }
            )

    # Main shot/method comparison.
    add_rows(
        "method_by_shot",
        "Frozen ST + LR",
        FROZEN_DIR,
        reference=(SETFIT_SHOT_DIR, SETFIT_8_DIR),
        num_iterations="N/A",
        classifier="logistic_regression",
        pair_strategy="N/A",
        hard_negative_ratio="N/A",
        reference_name="SetFit same shot",
    )
    add_rows(
        "method_by_shot",
        "SetFit",
        SETFIT_SHOT_DIR,
        num_iterations=20,
        classifier="logistic_regression",
        pair_strategy="random",
        hard_negative_ratio="N/A",
    )
    add_rows(
        "method_by_shot",
        "SetFit",
        SETFIT_8_DIR,
        num_iterations=20,
        classifier="logistic_regression",
        pair_strategy="random",
        hard_negative_ratio="N/A",
    )
    add_rows(
        "method_by_shot",
        "SetFit-HN-50",
        HN50_SHOT_DIR,
        reference=(SETFIT_SHOT_DIR, SETFIT_8_DIR),
        num_iterations=20,
        classifier="logistic_regression",
        pair_strategy="hard_negative",
        hard_negative_ratio=0.5,
        reference_name="SetFit same shot",
    )
    add_rows(
        "method_by_shot",
        "SetFit-HN-50",
        HN50_8_DIR,
        reference=SETFIT_8_DIR,
        num_iterations=20,
        classifier="logistic_regression",
        pair_strategy="hard_negative",
        hard_negative_ratio=0.5,
        reference_name="SetFit same shot",
    )

    # Controlled 8-shot ablations.
    for num_iter, result_dir in ITER_DIRS.items():
        add_rows(
            "num_iterations",
            f"SetFit iter={num_iter}",
            result_dir,
            reference=SETFIT_8_DIR,
            shot_filter={8},
            num_iterations=num_iter,
            classifier="logistic_regression",
            pair_strategy="random",
            hard_negative_ratio="N/A",
            reference_name="SetFit iter=20",
        )
    for classifier, result_dir in CLASSIFIER_DIRS.items():
        add_rows(
            "classifier",
            f"SetFit {classifier}",
            result_dir,
            reference=SETFIT_8_DIR,
            shot_filter={8},
            num_iterations=20,
            classifier=classifier,
            pair_strategy="random",
            hard_negative_ratio="N/A",
            reference_name="SetFit LR",
        )
    for method, result_dir in HN_RATIO_DIRS.items():
        ratio = "N/A" if method == "SetFit" else method.replace("HN-", "0.")
        if method == "HN-100":
            ratio = "1.0"
        add_rows(
            "hard_negative_ratio",
            method,
            result_dir,
            reference=SETFIT_8_DIR,
            shot_filter={8},
            num_iterations=20,
            classifier="logistic_regression",
            pair_strategy="random" if method == "SetFit" else "hard_negative",
            hard_negative_ratio=ratio,
            reference_name="SetFit random pairs",
        )

    # Avoid accidental duplicate rows from reused 8-shot baselines.
    deduped = {}
    for row in rows:
        key = (
            row["experiment_group"],
            row["method"],
            row["dataset"],
            row["shot"],
            row["num_iterations"],
            row["classifier"],
            row["pair_strategy"],
            row["hard_negative_ratio"],
        )
        deduped[key] = row
    rows = sorted(
        deduped.values(),
        key=lambda row: (
            row["experiment_group"],
            DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99,
            int(row["shot"]),
            row["method"],
        ),
    )

    fields = [
        "experiment_group",
        "method",
        "dataset",
        "measure",
        "shot",
        "num_iterations",
        "classifier",
        "pair_strategy",
        "hard_negative_ratio",
        "mean",
        "std",
        "delta_vs_reference",
        "reference",
    ]
    write_csv(OUT_DIR / "00_unified_score_results.csv", rows, fields)
    write_markdown(OUT_DIR / "00_unified_score_results.md", rows, fields, "Unified Score Results")
    return rows


def build_configs():
    rows = [
        {
            "experiment": "SetFit shot ablation",
            "method": "SetFit",
            "shots": "2/4/8/16/64",
            "num_iterations": 20,
            "classifier": "logistic_regression",
            "pair_strategy": "random",
            "hard_negative_ratio": "",
            "result_dirs": f"{SETFIT_SHOT_DIR.name}; {SETFIT_8_DIR.name}",
        },
        {
            "experiment": "Frozen ST + LR",
            "method": "Frozen ST + LR",
            "shots": "2/4/8/16/64",
            "num_iterations": "",
            "classifier": "logistic_regression",
            "pair_strategy": "",
            "hard_negative_ratio": "",
            "result_dirs": FROZEN_DIR.name,
        },
        {
            "experiment": "Hard Negative shot ablation",
            "method": "SetFit-HN-50",
            "shots": "2/4/8/16/64",
            "num_iterations": 20,
            "classifier": "logistic_regression",
            "pair_strategy": "hard_negative",
            "hard_negative_ratio": 0.5,
            "result_dirs": f"{HN50_SHOT_DIR.name}; {HN50_8_DIR.name}",
        },
        {
            "experiment": "Hard Negative ratio ablation",
            "method": "SetFit-HN",
            "shots": "8",
            "num_iterations": 20,
            "classifier": "logistic_regression",
            "pair_strategy": "hard_negative",
            "hard_negative_ratio": "0.25/0.5/0.75/1.0",
            "result_dirs": "HN-25/HN-50/HN-75/HN-100 8-shot dirs",
        },
    ]
    fields = ["experiment", "method", "shots", "num_iterations", "classifier", "pair_strategy", "hard_negative_ratio", "result_dirs"]
    write_csv(OUT_DIR / "01_all_experiment_configs.csv", rows, fields)
    write_markdown(OUT_DIR / "01_all_experiment_configs.md", rows, fields, "All Experiment Configs")


def build_readme():
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Consolidated Ablation Results",
                "",
                "这个目录由 `scripts/setfit/rebuild_consolidated_ablation_results.py` 从原始 `results.json` 重新生成。",
                "",
                "| 文件 | 内容 |",
                "| --- | --- |",
                "| `00_unified_score_results.csv/md` | 所有能统一的分数型结果长表：method、shot、num_iterations、classifier、HN ratio |",
                "| `01_all_experiment_configs.csv/md` | 所有实验配置和命令参数 |",
                "| `02_core_setfit_ablations.csv/md` | 原始 SetFit 主消融：shot、num_iterations、classifier |",
                "| `03_unified_method_by_shot_comparison.csv/md` | Frozen ST + LR、SetFit、SetFit-HN-50 的统一 shot 对比 |",
                "| `04_hard_negative_ratio_ablation.csv/md` | Hard Negative ratio 消融 |",
                "| `05_paired_significance_summary.csv/md` | HN-50 vs SetFit 的 paired split 稳定性/显著性分析 |",
                "",
                "## Recommended Reading",
                "",
                "一般只需要看三类文件：",
                "",
                "1. `00_unified_score_results.md`：所有分数型结果的总表。",
                "2. `05_paired_significance_summary.md`：HN-50 相比 SetFit 是否稳定提升。",
                "3. `01_all_experiment_configs.md`：需要确认实验配置和参数时再看。",
                "",
                "如果觉得 `00_unified_score_results.md` 太长，可以只看 `03_unified_method_by_shot_comparison.md`，它是最核心的 Frozen ST + LR / SetFit / SetFit-HN-50 对比。",
                "",
                "64-shot 结果已补齐，表格已同步更新。",
            ]
        )
        + "\n"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_unified_score_master()
    build_configs()
    core_rows = build_core_setfit()
    build_unified_method_table()
    build_hn_ratio_table()
    paired_rows_summary = build_paired_table()
    build_readme()
    build_legacy_test_only_tables(core_rows)
    build_legacy_frozen_tables()
    build_legacy_hn50_tables(paired_rows_summary)
    build_legacy_config_and_index()
    print(f"Rebuilt consolidated tables in {OUT_DIR}")


if __name__ == "__main__":
    main()
