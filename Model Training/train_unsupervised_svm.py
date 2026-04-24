from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "DataSet" / "fraud_transformed_reducted_scaled_train.csv"
DEFAULT_LINEAR_MODEL_PATH = PROJECT_ROOT / "Model Training" / "models" / "supervised_sgd_svm_model.pkl"
DEFAULT_KERNEL_MODEL_PATH = PROJECT_ROOT / "Model Training" / "models" / "supervised_rbf_svm_model.pkl"

TARGET_COLUMN = "is_fraud"
FREQUENCY_COLUMNS = ["job", "zip"]
MISSING_CATEGORY = "__missing__"
RANDOM_STATE = 42
RECALL_TARGET = 0.80
FPR_TARGET = 0.02

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supervised linear and kernel SVM fraud classifiers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LINEAR_MODEL_PATH)
    parser.add_argument("--kernel-model-path", type=Path, default=DEFAULT_KERNEL_MODEL_PATH)
    parser.add_argument("--training-sample-size", type=int, default=None)
    parser.add_argument("--kernel-training-sample-size", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["linear", "kernel", "both"],
        default=["both"],
        help="Which model(s) to train. Default trains both linear and RBF kernel SVM.",
    )
    parser.add_argument(
        "--param-preset",
        choices=["fast", "default"],
        default="default",
        help="fast is useful for smoke tests; default is the normal tuning grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run tuning and final fitting logic without saving the model file.",
    )
    return parser.parse_args()


def read_csv_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        chunksize=chunksize,
        dtype={"job": "string", "zip": "string"},
        low_memory=False,
    )


def compute_frequency_maps(path: Path, chunksize: int) -> tuple[dict[str, dict[str, float]], int]:
    counts = {column: pd.Series(dtype="int64") for column in FREQUENCY_COLUMNS}
    row_count = 0

    for chunk in read_csv_chunks(path, chunksize):
        row_count += len(chunk)
        for column in FREQUENCY_COLUMNS:
            values = chunk[column].astype("string").fillna(MISSING_CATEGORY)
            counts[column] = counts[column].add(values.value_counts(), fill_value=0)

    frequency_maps = {
        column: (column_counts / row_count).to_dict()
        for column, column_counts in counts.items()
    }
    return frequency_maps, row_count


def load_training_data(
    path: Path,
    sample_size: int | None,
    total_rows: int,
    chunksize: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= total_rows:
        return pd.read_csv(
            path,
            dtype={"job": "string", "zip": "string"},
            low_memory=False,
        )

    sample_fraction = min(1.0, sample_size / total_rows)
    samples = []

    for chunk_index, chunk in enumerate(read_csv_chunks(path, chunksize)):
        sampled_chunk = chunk.sample(
            frac=sample_fraction,
            random_state=random_state + chunk_index,
        )
        if not sampled_chunk.empty:
            samples.append(sampled_chunk)

    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return sampled


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype("string").str.lower()
    return normalized.isin(["true", "1", "yes"])


def prepare_features(
    dataframe: pd.DataFrame,
    frequency_maps: dict[str, dict[str, float]],
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    frame = dataframe.copy()
    target = None

    if TARGET_COLUMN in frame.columns:
        target = parse_bool_series(frame.pop(TARGET_COLUMN))

    for column in FREQUENCY_COLUMNS:
        values = frame[column].astype("string").fillna(MISSING_CATEGORY)
        frame[f"{column}_frequency"] = values.map(frequency_maps[column]).fillna(0.0)
        frame = frame.drop(columns=column)

    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype("int8")

    frame = frame.apply(pd.to_numeric, errors="raise")

    if feature_columns is not None:
        frame = frame.reindex(columns=feature_columns, fill_value=0)

    return frame.astype("float32"), target


def selected_model_names(raw_models: list[str]) -> set[str]:
    return {"linear", "kernel"} if "both" in raw_models else set(raw_models)


def get_linear_param_grid(preset: str) -> list[dict[str, float | str]]:
    if preset == "fast":
        return [
            {
                "loss": "modified_huber",
                "alpha": 0.001,
                "penalty": "l2",
                "fraud_weight": 200,
                "non_fraud_ratio": 20,
            }
        ]

    return [
        {
            "loss": loss,
            "alpha": alpha,
            "penalty": "l2",
            "fraud_weight": fraud_weight,
            "non_fraud_ratio": non_fraud_ratio,
        }
        for loss, alpha, fraud_weight, non_fraud_ratio in product(
            ["modified_huber", "log_loss"],
            [0.001],
            [50, 100, 200, 300],
            [10, 20, 50],
        )
    ]


def build_model(params: dict[str, float | str], random_state: int) -> SGDClassifier:
    return SGDClassifier(
        loss=str(params["loss"]),
        penalty=str(params["penalty"]),
        alpha=float(params["alpha"]),
        class_weight={0: 1.0, 1: float(params["fraud_weight"])},
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
        n_jobs=1,
    )


def get_kernel_param_grid(preset: str) -> list[dict[str, float | str]]:
    if preset == "fast":
        return [
            {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "fraud_weight": 20,
                "non_fraud_ratio": 20,
            }
        ]

    return [
        {
            "kernel": "rbf",
            "C": c_value,
            "gamma": gamma,
            "fraud_weight": fraud_weight,
            "non_fraud_ratio": non_fraud_ratio,
        }
        for c_value, gamma, fraud_weight, non_fraud_ratio in product(
            [0.5, 1.0, 2.0],
            ["scale"],
            [10, 20],
            [20],
        )
    ]


def build_kernel_model(params: dict[str, float | str], random_state: int) -> SVC:
    return SVC(
        kernel=str(params["kernel"]),
        C=float(params["C"]),
        gamma=params["gamma"],
        class_weight={0: 1.0, 1: float(params["fraud_weight"])},
        cache_size=1000,
        random_state=random_state,
    )


def decision_scores(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(features)

    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def metrics_from_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    y_bool = y_true.astype(bool).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_bool, predictions, labels=[False, True]).ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": recall_score(y_bool, predictions, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": precision_score(y_bool, predictions, zero_division=0),
        "f1": f1_score(y_bool, predictions, zero_division=0),
        "average_precision": average_precision_score(y_bool, scores),
        "roc_auc": roc_auc_score(y_bool, scores),
        "predicted_fraud_rate": float(np.mean(predictions)),
    }


def build_threshold_report(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0.001, 0.999, 300)),
                np.array([0.0]),
            ]
        )
    )
    rows = []

    for threshold in thresholds:
        predictions = scores >= threshold
        metrics = metrics_from_predictions(y_true, predictions, scores)
        rows.append({"threshold": float(threshold), **metrics})

    return pd.DataFrame(rows)


def select_threshold(threshold_report: pd.DataFrame) -> pd.Series:
    candidates = threshold_report[threshold_report["recall"] >= RECALL_TARGET]
    if not candidates.empty:
        return candidates.sort_values(
            ["false_positive_rate", "precision", "f1"],
            ascending=[True, False, False],
        ).iloc[0]

    return threshold_report.sort_values(
        ["f1", "average_precision", "roc_auc"],
        ascending=False,
    ).iloc[0]


def undersample_training_fold(
    features: pd.DataFrame,
    target: pd.Series,
    non_fraud_ratio: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    fraud_index = target[target.astype(bool)].index
    non_fraud_index = target[~target.astype(bool)].index

    max_non_fraud = len(fraud_index) * int(non_fraud_ratio)
    sampled_non_fraud_count = min(len(non_fraud_index), max_non_fraud)
    sampled_non_fraud_index = pd.Index(non_fraud_index).to_series().sample(
        n=sampled_non_fraud_count,
        random_state=random_state,
    ).index

    sampled_index = pd.Index(fraud_index).append(pd.Index(sampled_non_fraud_index))
    sampled_index = sampled_index.to_series().sample(
        frac=1.0,
        random_state=random_state,
    ).index

    return features.loc[sampled_index], target.loc[sampled_index]


def tune_hyperparameters(
    features: pd.DataFrame,
    target: pd.Series,
    param_grid: list[dict[str, float | str]],
    validation_size: float,
    random_state: int,
    model_builder,
    model_label: str,
) -> tuple[dict[str, float | str], float, list[dict[str, float | str]]]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=validation_size,
        random_state=random_state,
        stratify=target,
    )

    results = []
    best_params = param_grid[0]
    best_threshold = 0.0
    best_selection_score = -math.inf

    for index, params in enumerate(param_grid, start=1):
        console.log(f"Testing {model_label} params {index}/{len(param_grid)}: {params}")
        sampled_x_train, sampled_y_train = undersample_training_fold(
            x_train,
            y_train,
            int(params["non_fraud_ratio"]),
            random_state + index,
        )
        model = model_builder(params, random_state)
        model.fit(sampled_x_train, sampled_y_train.astype(int))

        scores = decision_scores(model, x_valid)
        threshold_report = build_threshold_report(y_valid, scores)
        selected = select_threshold(threshold_report)
        result = {
            **params,
            "fit_rows": len(sampled_x_train),
            "fit_fraud_rows": int(sampled_y_train.sum()),
            **selected.to_dict(),
        }
        results.append(result)

        target_hit = result["recall"] >= RECALL_TARGET
        selection_score = (
            10 - float(result["false_positive_rate"]) + float(result["precision"])
            if target_hit
            else float(result["recall"])
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_params = params
            best_threshold = float(result["threshold"])

    return best_params, best_threshold, results


def fit_final_model(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, float | str],
    random_state: int,
    model_builder,
):
    features, target = undersample_training_fold(
        features,
        target,
        int(params["non_fraud_ratio"]),
        random_state,
    )
    model = model_builder(params, random_state)
    model.fit(features, target.astype(int))
    return model


def print_tuning_results(results: list[dict[str, float | str]], title: str) -> None:
    table = Table(title=title, show_lines=True)
    columns = [
        column
        for column in [
            "loss",
            "kernel",
            "C",
            "gamma",
            "alpha",
            "fraud_weight",
            "non_fraud_ratio",
            "fit_rows",
            "threshold",
            "recall",
            "false_positive_rate",
            "precision",
            "f1",
            "average_precision",
            "roc_auc",
            "predicted_fraud_rate",
        ]
        if column in results[0]
    ]
    for column in columns:
        table.add_column(column, justify="right")

    sorted_results = sorted(results, key=lambda item: float(item["f1"]), reverse=True)
    for result in sorted_results:
        row = []
        for column in columns:
            value = result[column]
            if column == "fit_rows":
                row.append(f"{int(value):,}")
            elif isinstance(value, float):
                row.append(f"{value:.6f}" if column == "threshold" else f"{value:.4f}")
            else:
                row.append(f"{value}")
        table.add_row(*row)

    console.print(table)


def train_and_save_model(
    *,
    model_label: str,
    model_type: str,
    features: pd.DataFrame,
    target: pd.Series,
    param_grid: list[dict[str, float | str]],
    validation_size: float,
    random_state: int,
    model_builder,
    model_path: Path,
    dry_run: bool,
) -> None:
    best_params, best_threshold, tuning_results = tune_hyperparameters(
        features,
        target,
        param_grid,
        validation_size,
        random_state,
        model_builder,
        model_label,
    )
    print_tuning_results(tuning_results, f"{model_label} SVM Tuning Results")
    console.print(f"[bold green]{model_label} selected params:[/bold green] {best_params}")
    console.print(
        f"[bold green]{model_label} selected threshold:[/bold green] {best_threshold:.6f}"
    )

    final_model = fit_final_model(features, target, best_params, random_state, model_builder)
    artifact = {
        "model_type": model_type,
        "model": final_model,
        "threshold": best_threshold,
        "feature_columns": list(features.columns),
        "params": best_params,
        "target_column": TARGET_COLUMN,
    }

    if dry_run:
        console.print(
            f"[yellow]{model_label} dry run completed. "
            f"Model was fitted on {len(features):,} rows but not saved.[/yellow]"
        )
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, model_path)
    console.print(f"[bold green]{model_label} model artifact saved:[/bold green] {model_path}")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    console.print(
        Panel.fit(
            "Training models: supervised linear SVM and RBF kernel SVM\n"
            "Target: is_fraud\n"
            "Frequency encoding: job, zip\n"
            "Threshold tuning: validation split",
            title="Supervised SVM Training",
            border_style="cyan",
        )
    )

    frequency_maps, total_rows = compute_frequency_maps(args.input, args.chunksize)
    console.log(f"Rows found: {total_rows:,}")

    dataframe = load_training_data(
        args.input,
        args.training_sample_size,
        total_rows,
        args.chunksize,
        args.random_state,
    )
    features, target = prepare_features(dataframe, frequency_maps)
    if target is None:
        raise ValueError(f"{TARGET_COLUMN} column is required for supervised training.")

    console.log(
        f"Training rows loaded: {len(features):,} | fraud labels: {int(target.sum()):,}"
    )

    models_to_train = selected_model_names(args.models)

    if "linear" in models_to_train:
        train_and_save_model(
            model_label="Linear",
            model_type="supervised_sgd_svm",
            features=features,
            target=target,
            param_grid=get_linear_param_grid(args.param_preset),
            validation_size=args.validation_size,
            random_state=args.random_state,
            model_builder=build_model,
            model_path=args.model_path,
            dry_run=args.dry_run,
        )

    if "kernel" in models_to_train:
        kernel_features = features
        kernel_target = target
        if (
            args.kernel_training_sample_size is not None
            and args.kernel_training_sample_size < len(features)
        ):
            sample_index, _ = train_test_split(
                features.index.to_numpy(),
                train_size=args.kernel_training_sample_size,
                random_state=args.random_state,
                stratify=target,
            )
            kernel_features = features.loc[sample_index].reset_index(drop=True)
            kernel_target = target.loc[sample_index].reset_index(drop=True)
            console.log(f"Kernel training rows sampled: {len(kernel_features):,}")

        train_and_save_model(
            model_label="Kernel RBF",
            model_type="supervised_rbf_svm",
            features=kernel_features,
            target=kernel_target,
            param_grid=get_kernel_param_grid(args.param_preset),
            validation_size=args.validation_size,
            random_state=args.random_state,
            model_builder=build_kernel_model,
            model_path=args.kernel_model_path,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
