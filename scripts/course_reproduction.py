#!/usr/bin/env python
"""Course reproduction experiments for SetFit few-shot text classification.

This script is intentionally independent from the original paper scripts. It
uses the public SetFit API from the current checkout and writes all outputs to
results/course_reproduction/ by default.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "course_reproduction"
DEFAULT_BACKBONE = "sentence-transformers/paraphrase-mpnet-base-v2"
DEFAULT_FALLBACK_BACKBONE = "sentence-transformers/paraphrase-MiniLM-L3-v2"


@dataclass(frozen=True)
class DatasetCandidate:
    path: str
    config: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    candidates: Tuple[DatasetCandidate, ...]
    substitute_candidates: Tuple[DatasetCandidate, ...] = ()
    substitute_reason: str = ""


@dataclass
class LoadedDataset:
    key: str
    display_name: str
    source: str
    train: "Dataset"
    test: "Dataset"
    label_names: List[str]
    notes: List[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "sst5": DatasetSpec(
        key="sst5",
        display_name="SST-5",
        candidates=(
            DatasetCandidate("SetFit/sst5"),
            DatasetCandidate("sst5"),
            DatasetCandidate("SetFit/sst_5"),
        ),
    ),
    "cr": DatasetSpec(
        key="cr",
        display_name="CR",
        candidates=(
            DatasetCandidate("SetFit/SentEval-CR"),
            DatasetCandidate("SetFit/cr"),
            DatasetCandidate("cr"),
        ),
        substitute_candidates=(
            DatasetCandidate("SetFit/sst2"),
            DatasetCandidate("sst2"),
            DatasetCandidate("imdb"),
        ),
        substitute_reason="CR/Customer Reviews could not be loaded directly; SST-2 or IMDB is used as the documented fallback.",
    ),
    "ag_news": DatasetSpec(
        key="ag_news",
        display_name="AG News",
        candidates=(
            DatasetCandidate("SetFit/ag_news"),
            DatasetCandidate("ag_news"),
        ),
    ),
    "sst2": DatasetSpec(
        key="sst2",
        display_name="SST-2",
        candidates=(
            DatasetCandidate("SetFit/sst2"),
            DatasetCandidate("sst2"),
        ),
    ),
    "imdb": DatasetSpec(
        key="imdb",
        display_name="IMDB",
        candidates=(DatasetCandidate("imdb"),),
    ),
}

METHOD_NAMES = {
    "tfidf_lr": "TF-IDF+LR",
    "frozen_st_lr": "Frozen ST+LR",
    "setfit": "SETFIT",
    "bert_ft": "BERT Fine-tuning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Level 1 SetFit course reproduction experiments.")
    parser.add_argument("--datasets", nargs="+", default=["sst5", "ag_news"], help="Dataset keys: sst5 cr ag_news")
    parser.add_argument("--shots", nargs="+", type=int, default=[8, 16, 32], help="Samples per class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", nargs="+", default=[DEFAULT_BACKBONE], help="Sentence Transformer backbone(s).")
    parser.add_argument(
        "--fallback_backbone",
        default=DEFAULT_FALLBACK_BACKBONE,
        help="Backbone tried after a Sentence Transformer/SetFit model load failure. Use '' to disable.",
    )
    parser.add_argument("--max_test_per_class", type=int, default=500, help="Use <=0 to evaluate on the full test split.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--methods", nargs="+", default=["tfidf_lr", "frozen_st_lr", "setfit"], choices=sorted(METHOD_NAMES))
    parser.add_argument("--include_bert", action="store_true", help="Also run standard Transformer fine-tuning.")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--bert_epochs", type=float, default=3.0)
    parser.add_argument("--bert_batch_size", type=int, default=8)
    parser.add_argument("--setfit_epochs", type=int, default=1)
    parser.add_argument("--setfit_iterations", type=int, default=20)
    parser.add_argument("--setfit_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse rows already present in main_results.csv.")
    return parser.parse_args()


def require_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install this checkout first, for example: "
            "pip install -e '.[dev]'"
        ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def candidate_label(candidate: DatasetCandidate) -> str:
    return candidate.path if candidate.config is None else f"{candidate.path}/{candidate.config}"


def load_candidate(candidate: DatasetCandidate):
    datasets = require_module("datasets")
    if candidate.config is None:
        return datasets.load_dataset(candidate.path)
    return datasets.load_dataset(candidate.path, candidate.config)


def choose_split(dataset_dict, names: Sequence[str], role: str):
    for name in names:
        if name in dataset_dict:
            return dataset_dict[name]
    available = ", ".join(dataset_dict.keys())
    raise ValueError(f"No {role} split found. Available splits: {available}")


def normalize_columns(dataset, notes: List[str]):
    text_candidates = ("text", "sentence", "review", "content")
    label_candidates = ("label", "labels", "target")

    text_column = next((column for column in text_candidates if column in dataset.column_names), None)
    label_column = next((column for column in label_candidates if column in dataset.column_names), None)
    if text_column is None or label_column is None:
        raise ValueError(f"Could not infer text/label columns from {dataset.column_names}")

    rename_map = {}
    if text_column != "text":
        rename_map[text_column] = "text"
    if label_column != "label":
        rename_map[label_column] = "label"
    if rename_map:
        notes.append(f"Renamed columns {rename_map} to the SetFit text/label schema.")
        dataset = dataset.rename_columns(rename_map)
    return dataset


def label_names_from_dataset(dataset) -> List[str]:
    feature = dataset.features.get("label")
    names = getattr(feature, "names", None)
    if names:
        return [str(name) for name in names]
    labels = sorted(set(dataset["label"]))
    return [str(label) for label in labels]


def try_load_dataset(spec: DatasetSpec, candidate: DatasetCandidate, notes: List[str]) -> LoadedDataset:
    dataset_dict = load_candidate(candidate)
    train = choose_split(dataset_dict, ("train",), "train")
    test = choose_split(dataset_dict, ("test", "validation", "dev"), "test/evaluation")
    if "test" not in dataset_dict:
        notes.append("Dataset has no test split; using validation/dev split for evaluation.")
    train = normalize_columns(train, notes)
    test = normalize_columns(test, notes)
    return LoadedDataset(
        key=spec.key,
        display_name=spec.display_name,
        source=candidate_label(candidate),
        train=train,
        test=test,
        label_names=label_names_from_dataset(train),
        notes=notes,
    )


def load_dataset_for_key(dataset_key: str) -> LoadedDataset:
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Known keys: {', '.join(sorted(DATASET_SPECS))}")
    spec = DATASET_SPECS[dataset_key]
    errors = []

    for candidate in spec.candidates:
        notes: List[str] = []
        try:
            return try_load_dataset(spec, candidate, notes)
        except Exception as exc:
            errors.append(f"{candidate_label(candidate)}: {type(exc).__name__}: {exc}")

    for candidate in spec.substitute_candidates:
        notes = [spec.substitute_reason, "Original load failures: " + " | ".join(errors)]
        try:
            loaded = try_load_dataset(spec, candidate, notes)
            loaded.display_name = f"{spec.display_name} fallback ({loaded.display_name})"
            return loaded
        except Exception as exc:
            errors.append(f"{candidate_label(candidate)}: {type(exc).__name__}: {exc}")

    raise RuntimeError("All dataset candidates failed: " + " | ".join(errors))


def sample_per_class(dataset, samples_per_class: int, seed: int):
    shuffled = dataset.shuffle(seed=seed)
    selected_indices: List[int] = []
    counts: Dict[object, int] = {}
    for idx, label in enumerate(shuffled["label"]):
        if counts.get(label, 0) >= samples_per_class:
            continue
        selected_indices.append(idx)
        counts[label] = counts.get(label, 0) + 1
    return shuffled.select(selected_indices)


def maybe_limit_test_set(dataset, max_per_class: int, seed: int):
    if max_per_class <= 0:
        return dataset, "Full original evaluation split was used."
    limited = sample_per_class(dataset, max_per_class, seed)
    return limited, f"Evaluation split limited to at most {max_per_class} samples per class."


def get_xy(dataset) -> Tuple[List[str], List[object]]:
    return list(dataset["text"]), list(dataset["label"])


def compute_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> Tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def train_eval_tfidf(train_dataset, test_dataset, seed: int) -> Tuple[float, float]:
    x_train, y_train = get_xy(train_dataset)
    x_test, y_test = get_xy(test_dataset)
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, random_state=seed, solver="liblinear"),
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return compute_metrics(y_test, preds)


def load_sentence_transformer(backbone: str):
    sentence_transformers = require_module("sentence_transformers")
    return sentence_transformers.SentenceTransformer(backbone)


def train_eval_frozen_st(train_dataset, test_dataset, backbone: str, seed: int, max_seq_length: int) -> Tuple[float, float]:
    model = load_sentence_transformer(backbone)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_seq_length
    x_train, y_train = get_xy(train_dataset)
    x_test, y_test = get_xy(test_dataset)
    train_embeddings = model.encode(x_train, batch_size=32, show_progress_bar=False, normalize_embeddings=False)
    test_embeddings = model.encode(x_test, batch_size=32, show_progress_bar=False, normalize_embeddings=False)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_embeddings, y_train)
    preds = clf.predict(test_embeddings)
    return compute_metrics(y_test, preds)


def train_eval_setfit(
    train_dataset,
    test_dataset,
    backbone: str,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    from sentence_transformers import losses
    from setfit import SetFitModel, Trainer, TrainingArguments

    output_dir = Path(args.output_dir) / "checkpoints" / "setfit" / safe_name(backbone)
    model = SetFitModel.from_pretrained(
        backbone,
        head_params={"max_iter": 1000, "random_state": seed},
    )
    model.model_body.max_seq_length = args.max_seq_length
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        num_epochs=args.setfit_epochs,
        num_iterations=args.setfit_iterations,
        body_learning_rate=args.setfit_lr,
        loss=losses.CosineSimilarityLoss,
        max_length=args.max_seq_length,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=seed,
        show_progress_bar=not args.no_progress_bar,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, metric="accuracy")
    trainer.train()
    x_test, y_test = get_xy(test_dataset)
    preds = model.predict(x_test, use_labels=False)
    preds = np.asarray(preds).tolist()
    return compute_metrics(y_test, preds)


def make_transformers_training_args(**kwargs):
    from transformers import TrainingArguments

    try:
        return TrainingArguments(eval_strategy="no", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="no", **kwargs)


def train_eval_bert(train_dataset, test_dataset, dataset: LoadedDataset, seed: int, args: argparse.Namespace) -> Tuple[float, float]:
    require_module("transformers")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    label_values = sorted(set(train_dataset["label"]))
    label2id = {label: idx for idx, label in enumerate(label_values)}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    train_encoded = train_dataset.map(encode_labels)
    test_encoded = test_dataset.map(encode_labels)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length)

    remove_columns = [column for column in train_encoded.column_names if column != "label"]
    tokenized_train = train_encoded.map(tokenize, batched=True, remove_columns=remove_columns)
    tokenized_test = test_encoded.map(tokenize, batched=True, remove_columns=remove_columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model,
        num_labels=len(label_values),
        id2label={idx: dataset.label_names[int(label)] if isinstance(label, int) and int(label) < len(dataset.label_names) else str(label) for label, idx in label2id.items()},
        label2id={str(label): idx for label, idx in label2id.items()},
    )
    training_args = make_transformers_training_args(
        output_dir=str(Path(args.output_dir) / "checkpoints" / "bert" / safe_name(args.bert_model)),
        overwrite_output_dir=True,
        num_train_epochs=args.bert_epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=args.bert_batch_size,
        per_device_eval_batch_size=args.bert_batch_size,
        save_strategy="no",
        report_to="none",
        seed=seed,
        disable_tqdm=args.no_progress_bar,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, tokenizer=tokenizer)
    trainer.train()
    pred_output = trainer.predict(tokenized_test)
    preds = np.argmax(pred_output.predictions, axis=-1)
    y_true = list(tokenized_test["label"])
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return compute_metrics(y_true, preds)


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def existing_keys(output_dir: Path) -> set:
    path = output_dir / "main_results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["Dataset"], df["Shot"], df["Method"], df["Backbone"]))


def append_result(output_dir: Path, row: Dict[str, object]) -> None:
    path = output_dir / "main_results.csv"
    fieldnames = ["Dataset", "Shot", "Method", "Backbone", "Accuracy", "Macro_F1"]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fieldnames})


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("_No successful experiment rows yet._\n", encoding="utf-8")
    else:
        path.write_text(dataframe_to_markdown(df) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = list(df.columns)
        rows = [[str(value) for value in row] for row in df.to_numpy()]
        widths = [
            max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
            for idx, header in enumerate(headers)
        ]
        header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
        return "\n".join([header_line, separator, *body])


def load_main_results(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "main_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Dataset", "Shot", "Method", "Backbone", "Accuracy", "Macro_F1"])
    return pd.read_csv(path)


def build_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, shot, backbone), group in df.groupby(["Dataset", "Shot", "Backbone"], dropna=False):
        frozen = group[group["Method"] == METHOD_NAMES["frozen_st_lr"]]
        setfit = group[group["Method"] == METHOD_NAMES["setfit"]]
        if frozen.empty or setfit.empty or backbone == "n/a":
            continue
        frozen_acc = float(frozen.iloc[-1]["Accuracy"])
        setfit_acc = float(setfit.iloc[-1]["Accuracy"])
        rows.append(
            {
                "Dataset": dataset,
                "Shot": shot,
                "Backbone": backbone,
                "Frozen_ST_LR_Accuracy": frozen_acc,
                "SETFIT_Accuracy": setfit_acc,
                "Improvement": setfit_acc - frozen_acc,
            }
        )
    return pd.DataFrame(rows)


def build_shot_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, backbone), group in df.groupby(["Dataset", "Method", "Backbone"], dropna=False):
        by_shot = {int(row["Shot"]): float(row["Accuracy"]) for _, row in group.iterrows()}
        rows.append(
            {
                "Dataset": dataset,
                "Method": method,
                "Backbone": backbone,
                "Accuracy_8shot": by_shot.get(8, np.nan),
                "Accuracy_16shot": by_shot.get(16, np.nan),
                "Accuracy_32shot": by_shot.get(32, np.nan),
            }
        )
    return pd.DataFrame(rows)


def write_derived_outputs(output_dir: Path, failures: List[Dict[str, str]], dataset_notes: List[str], args: argparse.Namespace) -> None:
    df = load_main_results(output_dir)
    if not df.empty:
        df = df.drop_duplicates(subset=["Dataset", "Shot", "Method", "Backbone"], keep="last")
        df = df.sort_values(["Dataset", "Shot", "Method", "Backbone"])
        df.to_csv(output_dir / "main_results.csv", index=False)

    write_markdown_table(df, output_dir / "main_results.md")
    ablation = build_ablation(df)
    ablation.to_csv(output_dir / "ablation_contrastive.csv", index=False)
    shot_analysis = build_shot_analysis(df)
    shot_analysis.to_csv(output_dir / "shot_analysis.csv", index=False)
    write_summary(output_dir, df, ablation, shot_analysis, failures, dataset_notes, args)


def format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}"


def write_summary(
    output_dir: Path,
    df: pd.DataFrame,
    ablation: pd.DataFrame,
    shot_analysis: pd.DataFrame,
    failures: List[Dict[str, str]],
    dataset_notes: List[str],
    args: argparse.Namespace,
) -> None:
    env = collect_environment()
    lines = [
        "# Course Reproduction Summary",
        "",
        "## Experiment environment",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Dataset setup",
            "",
            "- Requested datasets: " + ", ".join(args.datasets),
            "- Few-shot training data is sampled with the same seed and reused by every method for a given dataset and shot.",
            f"- Seed: {args.seed}",
            f"- Test sampling: {'full test split' if args.max_test_per_class <= 0 else f'at most {args.max_test_per_class} examples per class'}",
        ]
    )
    for note in dataset_notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Methods",
            "",
            "- TF-IDF+LR: TF-IDF unigram/bigram features followed by Logistic Regression.",
            "- Frozen ST+LR: pretrained Sentence Transformer embeddings without contrastive fine-tuning, followed by Logistic Regression.",
            "- SETFIT: official SetFit API with contrastive Sentence Transformer fine-tuning followed by Logistic Regression.",
            "- BERT Fine-tuning: optional standard Transformer sequence-classification fine-tuning when `--include_bert` is used.",
            "",
            "## SetFit training parameters",
            "",
            f"- Backbone(s): {', '.join(args.backbone)}",
            f"- Loss: CosineSimilarityLoss",
            f"- Learning rate: {args.setfit_lr}",
            f"- Batch size: {args.batch_size}",
            f"- Max sequence length: {args.max_seq_length}",
            f"- Epochs: {args.setfit_epochs}",
            f"- Pair generation R / num_iterations: {args.setfit_iterations}",
            f"- Classification head: Logistic Regression",
            "",
            "## Main results",
            "",
        ]
    )
    if df.empty:
        lines.append("_No successful experiment rows yet._")
    else:
        display_df = df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].map(format_percent)
        display_df["Macro_F1"] = display_df["Macro_F1"].map(format_percent)
        lines.append(dataframe_to_markdown(display_df))

    lines.extend(["", "## Contrastive fine-tuning ablation", ""])
    if ablation.empty:
        lines.append("_No paired Frozen ST+LR and SETFIT rows are available yet._")
    else:
        display_ablation = ablation.copy()
        for column in ["Frozen_ST_LR_Accuracy", "SETFIT_Accuracy", "Improvement"]:
            display_ablation[column] = display_ablation[column].map(format_percent)
        lines.append(dataframe_to_markdown(display_ablation))

    lines.extend(["", "## Shot analysis", ""])
    if shot_analysis.empty:
        lines.append("_No shot analysis rows are available yet._")
    else:
        display_shot = shot_analysis.copy()
        for column in ["Accuracy_8shot", "Accuracy_16shot", "Accuracy_32shot"]:
            display_shot[column] = display_shot[column].map(format_percent)
        lines.append(dataframe_to_markdown(display_shot))

    lines.extend(["", "## Issues and resolutions", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- {failure['scope']}: {failure['error_type']}: {failure['message']}"
            )
    else:
        lines.append("- No failures were recorded in this run.")

    lines.append("")
    (output_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")

    if failures:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_environment() -> Dict[str, str]:
    modules = ["python", "setfit", "datasets", "sentence_transformers", "transformers", "torch", "sklearn", "pandas", "numpy"]
    env: Dict[str, str] = {"python": sys.version.split()[0]}
    for module in modules[1:]:
        try:
            imported = importlib.import_module(module)
            env[module] = str(getattr(imported, "__version__", "installed"))
        except Exception as exc:
            env[module] = f"unavailable ({type(exc).__name__})"
    return env


def print_result(dataset: str, shot: int, method: str, backbone: str, train_size: int, test_size: int, accuracy: float, macro_f1: float) -> None:
    print(
        f"[done] dataset={dataset} shot={shot} method={method} backbone={backbone} "
        f"train={train_size} test={test_size} accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}",
        flush=True,
    )


def method_backbones(method: str, args: argparse.Namespace) -> List[str]:
    if method == "tfidf_lr":
        return ["n/a"]
    if method == "bert_ft":
        return [args.bert_model]
    backbones = list(dict.fromkeys(args.backbone))
    if args.fallback_backbone and args.fallback_backbone not in backbones:
        backbones.append(args.fallback_backbone)
    return backbones


def run_method(
    method: str,
    train_dataset,
    test_dataset,
    loaded_dataset: LoadedDataset,
    backbone: str,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    set_seed(args.seed)
    if method == "tfidf_lr":
        return train_eval_tfidf(train_dataset, test_dataset, args.seed)
    if method == "frozen_st_lr":
        return train_eval_frozen_st(train_dataset, test_dataset, backbone, args.seed, args.max_seq_length)
    if method == "setfit":
        return train_eval_setfit(train_dataset, test_dataset, backbone, args.seed, args)
    if method == "bert_ft":
        return train_eval_bert(train_dataset, test_dataset, loaded_dataset, args.seed, args)
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.include_bert and "bert_ft" not in args.methods:
        args.methods.append("bert_ft")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, str]] = []
    dataset_notes: List[str] = []
    already_done = existing_keys(output_dir) if args.skip_existing else set()

    start_time = time.time()
    print(f"[info] writing results to {output_dir}", flush=True)
    print(f"[info] methods={', '.join(METHOD_NAMES[method] for method in args.methods)}", flush=True)

    for dataset_key in args.datasets:
        try:
            loaded = load_dataset_for_key(dataset_key)
            test_dataset, test_note = maybe_limit_test_set(loaded.test, args.max_test_per_class, args.seed)
            dataset_notes.append(
                f"{loaded.display_name}: loaded from `{loaded.source}`; train={len(loaded.train)}, "
                f"eval={len(test_dataset)}. {test_note}"
            )
            for note in loaded.notes:
                dataset_notes.append(f"{loaded.display_name}: {note}")
            print(
                f"[dataset] {loaded.display_name} source={loaded.source} train={len(loaded.train)} test={len(test_dataset)} labels={loaded.label_names}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "scope": f"dataset={dataset_key}",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[skip] dataset={dataset_key} failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        for shot in args.shots:
            train_dataset = sample_per_class(loaded.train, shot, args.seed)
            train_size = len(train_dataset)
            label_counts = pd.Series(train_dataset["label"]).value_counts().sort_index().to_dict()
            print(f"[shot] dataset={loaded.display_name} shot={shot} train_size={train_size} label_counts={label_counts}", flush=True)

            for method in args.methods:
                method_name = METHOD_NAMES[method]
                explicit_backbones = set(args.backbone) if method in {"frozen_st_lr", "setfit"} else set()
                explicit_backbone_success = False
                for backbone in method_backbones(method, args):
                    is_auto_fallback = (
                        method in {"frozen_st_lr", "setfit"}
                        and backbone == args.fallback_backbone
                        and backbone not in explicit_backbones
                    )
                    if is_auto_fallback and explicit_backbone_success:
                        continue
                    key = (loaded.display_name, shot, method_name, backbone)
                    if key in already_done:
                        print(f"[skip] existing result {key}", flush=True)
                        if not is_auto_fallback:
                            explicit_backbone_success = True
                        continue
                    print(
                        f"[run] dataset={loaded.display_name} shot={shot} method={method_name} backbone={backbone}",
                        flush=True,
                    )
                    try:
                        accuracy, macro_f1 = run_method(method, train_dataset, test_dataset, loaded, backbone, args)
                    except Exception as exc:
                        failures.append(
                            {
                                "scope": f"dataset={loaded.display_name}, shot={shot}, method={method_name}, backbone={backbone}",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        print(f"[fail] {method_name} failed on {loaded.display_name}/{shot}/{backbone}: {exc}", flush=True)
                        if method in {"frozen_st_lr", "setfit"} and backbone != args.fallback_backbone:
                            print("[info] will continue to fallback/next backbone if configured", flush=True)
                        continue

                    row = {
                        "Dataset": loaded.display_name,
                        "Shot": shot,
                        "Method": method_name,
                        "Backbone": backbone,
                        "Accuracy": accuracy,
                        "Macro_F1": macro_f1,
                    }
                    append_result(output_dir, row)
                    print_result(loaded.display_name, shot, method_name, backbone, train_size, len(test_dataset), accuracy, macro_f1)
                    if not is_auto_fallback:
                        explicit_backbone_success = True

    write_derived_outputs(output_dir, failures, dataset_notes, args)
    elapsed = time.time() - start_time
    print(f"[info] finished in {elapsed / 60:.2f} min", flush=True)
    print(f"[info] main results: {output_dir / 'main_results.csv'}", flush=True)
    print(f"[info] summary: {output_dir / 'experiment_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
