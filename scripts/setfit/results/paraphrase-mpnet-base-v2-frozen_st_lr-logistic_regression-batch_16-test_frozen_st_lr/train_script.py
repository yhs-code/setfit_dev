import argparse
import json
import os
import pathlib
import sys
import warnings
from collections import Counter
from shutil import copyfile
from warnings import simplefilter

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef

from setfit.data import SAMPLE_SIZES
from setfit.utils import DEV_DATASET_TO_METRIC, TEST_DATASET_TO_METRIC, load_data_splits


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from create_summary_table import create_summary_table  # noqa: E402


simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--datasets", nargs="+", default=["sst2"])
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--exp_name", default="frozen_st_lr")
    parser.add_argument("--is_dev_set", type=bool, default=False)
    parser.add_argument("--is_test_set", type=bool, default=False)
    parser.add_argument("--override_results", default=False, action="store_true")
    parser.add_argument("--add_data_augmentation", default=False)
    return parser.parse_args()


def create_results_path(dataset: str, split_name: str, output_path: str) -> str:
    results_path = os.path.join(output_path, dataset, split_name, "results.json")
    print(f"\n\n======== {os.path.dirname(results_path)} =======")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return results_path


def compute_metric(metric: str, y_true, y_pred) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "matthews_correlation":
        return matthews_corrcoef(y_true, y_pred)
    raise ValueError(f"Unsupported metric for frozen ST + LR baseline: {metric}")


def main():
    args = parse_args()

    parent_directory = pathlib.Path(__file__).parent.absolute()
    output_path = (
        parent_directory
        / "results"
        / f"{args.model.replace('/', '-')}-frozen_st_lr-logistic_regression-batch_{args.batch_size}-{args.exp_name}".rstrip(
            "-"
        )
    )
    os.makedirs(output_path, exist_ok=True)

    train_script_path = os.path.join(output_path, "train_script.py")
    copyfile(__file__, train_script_path)
    with open(train_script_path, "a") as f_out:
        f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    if args.is_dev_set:
        dataset_to_metric = DEV_DATASET_TO_METRIC
    elif args.is_test_set:
        dataset_to_metric = TEST_DATASET_TO_METRIC
    else:
        dataset_to_metric = {dataset: "accuracy" for dataset in args.datasets}

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    for dataset, metric in dataset_to_metric.items():
        few_shot_train_splits, test_data = load_data_splits(dataset, args.sample_sizes, args.add_data_augmentation)
        print(f"Evaluating {dataset} using {metric!r}.")

        counter = Counter(test_data["label"])
        label_samples = sorted(counter.items(), key=lambda label_samples: label_samples[1])
        smallest_n_samples = label_samples[0][1]
        largest_n_samples = label_samples[-1][1]
        if largest_n_samples > smallest_n_samples * 1.5:
            warnings.warn(
                "The test set has a class imbalance "
                f"({', '.join(f'label {label} w. {n_samples} samples' for label, n_samples in label_samples)})."
            )

        test_embeddings = model.encode(
            list(test_data["text"]),
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        y_test = list(test_data["label"])

        for split_name, train_data in few_shot_train_splits.items():
            results_path = create_results_path(dataset, split_name, output_path)
            if os.path.exists(results_path) and not args.override_results:
                print(f"Skipping finished experiment: {results_path}")
                continue

            train_embeddings = model.encode(
                list(train_data["text"]),
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            y_train = list(train_data["label"])

            clf = LogisticRegression()
            clf.fit(train_embeddings, y_train)
            predictions = clf.predict(test_embeddings)
            score = compute_metric(metric, y_test, predictions)
            metrics = {metric: score}
            print(f"Metrics: {metrics}")

            with open(results_path, "w") as f_out:
                json.dump({"score": score * 100, "measure": metric}, f_out, sort_keys=True)

    create_summary_table(str(output_path))


if __name__ == "__main__":
    main()


# Script was called via:
#python scripts/setfit/run_frozen_st_lr.py --sample_sizes 64 --is_test_set=true --exp_name test_frozen_st_lr