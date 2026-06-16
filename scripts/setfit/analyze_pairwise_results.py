import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy import stats
except ImportError:  # pragma: no cover - scipy is optional for this utility
    stats = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--baseline_name", default="baseline")
    parser.add_argument("--candidate_name", default="candidate")
    return parser.parse_args()


def collect_results(results_dir: Path):
    results = {}
    for path in sorted(results_dir.glob("*/*/results.json")):
        dataset = path.parent.parent.name
        split = path.parent.name
        with path.open() as f:
            payload = json.load(f)
        results[(dataset, split)] = {
            "dataset": dataset,
            "split": split,
            "score": float(payload["score"]),
            "measure": payload["measure"],
        }
    return results


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


def summarize_group(dataset, rows):
    deltas = [row["delta"] for row in rows]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    t_p, wilcoxon_p = p_values(deltas)
    return {
        "dataset": dataset,
        "measure": rows[0]["measure"],
        "n_pairs": len(rows),
        "baseline_avg": mean(row["baseline_score"] for row in rows),
        "candidate_avg": mean(row["candidate_score"] for row in rows),
        "mean_delta": mean(deltas),
        "std_delta": stdev(deltas) if len(deltas) > 1 else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "paired_t_p": t_p,
        "wilcoxon_p": wilcoxon_p,
    }


def write_csv(path, summaries):
    fieldnames = [
        "dataset",
        "measure",
        "n_pairs",
        "baseline_avg",
        "candidate_avg",
        "mean_delta",
        "std_delta",
        "wins",
        "ties",
        "losses",
        "paired_t_p",
        "wilcoxon_p",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            formatted = row.copy()
            for key in ["baseline_avg", "candidate_avg", "mean_delta", "std_delta"]:
                formatted[key] = f"{formatted[key]:.4f}"
            writer.writerow(formatted)


def write_markdown(path, summaries, baseline_name, candidate_name):
    lines = [
        f"# Paired Split Analysis: {candidate_name} vs {baseline_name}",
        "",
        "| Dataset | Measure | Pairs | Baseline avg | Candidate avg | Mean delta | Delta std | Win/Tie/Loss | paired t-test p | Wilcoxon p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {measure} | {n_pairs} | {baseline_avg:.2f} | {candidate_avg:.2f} | "
            "{mean_delta:+.2f} | {std_delta:.2f} | {wins}/{ties}/{losses} | {paired_t_p} | {wilcoxon_p} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Scores are paired by identical dataset and split folder, such as `train-8-0`.",
            "- Positive mean delta means the candidate method outperformed the baseline on average.",
            "- p-values are descriptive because each dataset has only 10 paired splits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_prefix = Path(args.output_prefix)

    baseline = collect_results(baseline_dir)
    candidate = collect_results(candidate_dir)
    shared_keys = sorted(set(baseline) & set(candidate))
    if not shared_keys:
        raise ValueError("No shared dataset/split results found.")

    paired_rows = []
    for key in shared_keys:
        base = baseline[key]
        cand = candidate[key]
        paired_rows.append(
            {
                "dataset": base["dataset"],
                "split": base["split"],
                "measure": cand["measure"],
                "baseline_score": base["score"],
                "candidate_score": cand["score"],
                "delta": cand["score"] - base["score"],
            }
        )

    datasets = sorted({row["dataset"] for row in paired_rows})
    summaries = [summarize_group(dataset, [row for row in paired_rows if row["dataset"] == dataset]) for dataset in datasets]
    summaries.append(summarize_group("Average", paired_rows))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_prefix.with_suffix(".csv"), summaries)
    write_markdown(output_prefix.with_suffix(".md"), summaries, args.baseline_name, args.candidate_name)
    print(output_prefix.with_suffix(".csv"))
    print(output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
