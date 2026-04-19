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
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "DataSet" / "fraud_transformed_reducted_scaled_train.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model Training" / "models" / "rbf_sgd_oneclass_svm_model.pkl"

TARGET_COLUMN = "is_fraud"
FREQUENCY_COLUMNS = ["job", "zip"]
MISSING_CATEGORY = "__missing__"
RANDOM_STATE = 42

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an unsupervised RBFSampler -> SGDOneClassSVM fraud anomaly model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tuning-sample-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--param-preset",
        choices=["fast", "default"],
        default="default",
        help="fast is useful for smoke tests; default is the normal tuning grid.",
    )
    parser.add_argument(
        "--final-max-rows",
        type=int,
        default=None,
        help="Optional cap for final training rows. Leave empty to train on the full train CSV.",
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


def sample_training_data(
    path: Path,
    sample_size: int,
    total_rows: int,
    chunksize: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size >= total_rows:
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


def get_param_grid(preset: str) -> list[dict[str, float | int]]:
    if preset == "fast":
        gamma_values = [0.05]
        component_values = [64]
        nu_values = [0.005, 0.01]
    else:
        gamma_values = [0.01, 0.05]
        component_values = [64, 128]
        nu_values = [0.005, 0.01]

    return [
        {"gamma": gamma, "n_components": n_components, "nu": nu}
        for gamma, n_components, nu in product(gamma_values, component_values, nu_values)
    ]


def build_model(params: dict[str, float | int], random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "rbf_sampler",
                RBFSampler(
                    gamma=float(params["gamma"]),
                    n_components=int(params["n_components"]),
                    random_state=random_state,
                ),
            ),
            (
                "sgd_one_class_svm",
                SGDOneClassSVM(
                    nu=float(params["nu"]),
                    max_iter=1000,
                    tol=1e-3,
                    shuffle=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def score_model(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    anomaly_scores = -model.decision_function(x_valid)
    predictions = model.predict(x_valid)
    predicted_fraud = predictions == -1
    y_true = y_valid.astype(bool)

    if y_true.nunique() == 2:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        average_precision = average_precision_score(y_true, anomaly_scores)
    else:
        roc_auc = math.nan
        average_precision = math.nan

    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, predicted_fraud, zero_division=0),
        "recall": recall_score(y_true, predicted_fraud, zero_division=0),
        "f1": f1_score(y_true, predicted_fraud, zero_division=0),
        "predicted_anomaly_rate": float(np.mean(predicted_fraud)),
    }


def tune_hyperparameters(
    features: pd.DataFrame,
    target: pd.Series,
    param_grid: list[dict[str, float | int]],
    validation_size: float,
    random_state: int,
) -> tuple[dict[str, float | int], list[dict[str, float]]]:
    stratify = target if target.nunique() == 2 else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )

    results = []
    best_params = param_grid[0]
    best_score = -math.inf

    for index, params in enumerate(param_grid, start=1):
        console.log(f"Testing params {index}/{len(param_grid)}: {params}")
        model = build_model(params, random_state)
        model.fit(x_train)
        metrics = score_model(model, x_valid, y_valid)
        result = {**params, **metrics}
        results.append(result)

        selection_score = metrics["average_precision"]
        if math.isnan(selection_score):
            selection_score = metrics["f1"]

        if selection_score > best_score:
            best_score = selection_score
            best_params = params

    return best_params, results


def fit_final_model(
    path: Path,
    frequency_maps: dict[str, dict[str, float]],
    feature_columns: list[str],
    params: dict[str, float | int],
    chunksize: int,
    random_state: int,
    max_rows: int | None,
) -> tuple[Pipeline, int]:
    rbf_sampler = RBFSampler(
        gamma=float(params["gamma"]),
        n_components=int(params["n_components"]),
        random_state=random_state,
    )
    template = pd.DataFrame(
        np.zeros((1, len(feature_columns)), dtype=np.float32),
        columns=feature_columns,
    )
    rbf_sampler.fit(template)

    svm = SGDOneClassSVM(
        nu=float(params["nu"]),
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        random_state=random_state,
    )

    trained_rows = 0
    for chunk in read_csv_chunks(path, chunksize):
        if max_rows is not None:
            remaining_rows = max_rows - trained_rows
            if remaining_rows <= 0:
                break
            chunk = chunk.head(remaining_rows)

        features, _ = prepare_features(chunk, frequency_maps, feature_columns=feature_columns)
        transformed = rbf_sampler.transform(features).astype("float32", copy=False)
        svm.partial_fit(transformed)
        trained_rows += len(features)
        console.log(f"Final training rows processed: {trained_rows:,}")

    model = Pipeline(
        steps=[
            ("rbf_sampler", rbf_sampler),
            ("sgd_one_class_svm", svm),
        ]
    )
    return model, trained_rows


def print_tuning_results(results: list[dict[str, float]]) -> None:
    table = Table(title="Hyperparameter Tuning Results", show_lines=True)
    for column in [
        "gamma",
        "n_components",
        "nu",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "predicted_anomaly_rate",
    ]:
        table.add_column(column, justify="right")

    sorted_results = sorted(
        results,
        key=lambda item: (
            -1 if math.isnan(float(item["average_precision"])) else float(item["average_precision"])
        ),
        reverse=True,
    )
    for result in sorted_results:
        table.add_row(
            f"{result['gamma']}",
            f"{int(result['n_components'])}",
            f"{result['nu']}",
            f"{result['average_precision']:.4f}",
            f"{result['roc_auc']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['predicted_anomaly_rate']:.4f}",
        )

    console.print(table)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    console.print(
        Panel.fit(
            "Training model: RBFSampler -> SGDOneClassSVM\n"
            "Frequency encoding: job, zip\n"
            "Target usage: is_fraud is kept only for validation metrics",
            title="Unsupervised SVM Training",
            border_style="cyan",
        )
    )

    frequency_maps, total_rows = compute_frequency_maps(args.input, args.chunksize)
    console.log(f"Rows found: {total_rows:,}")

    tuning_sample = sample_training_data(
        args.input,
        args.tuning_sample_size,
        total_rows,
        args.chunksize,
        args.random_state,
    )
    features, target = prepare_features(tuning_sample, frequency_maps)
    if target is None:
        raise ValueError(f"{TARGET_COLUMN} column is required for validation metrics.")

    feature_columns = list(features.columns)
    fraud_count = int(target.sum())
    console.log(
        f"Tuning sample rows: {len(features):,} | fraud labels for evaluation: {fraud_count:,}"
    )

    param_grid = get_param_grid(args.param_preset)
    best_params, tuning_results = tune_hyperparameters(
        features,
        target,
        param_grid,
        args.validation_size,
        args.random_state,
    )
    print_tuning_results(tuning_results)
    console.print(f"[bold green]Selected params:[/bold green] {best_params}")

    final_model, trained_rows = fit_final_model(
        args.input,
        frequency_maps,
        feature_columns,
        best_params,
        args.chunksize,
        args.random_state,
        args.final_max_rows,
    )

    if args.dry_run:
        console.print(
            f"[yellow]Dry run completed. Model was fitted on {trained_rows:,} rows but not saved.[/yellow]"
        )
        return

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(final_model, args.model_path)
    console.print(f"[bold green]Model saved:[/bold green] {args.model_path}")


if __name__ == "__main__":
    main()
