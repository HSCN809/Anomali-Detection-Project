from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import load
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from train_unsupervised_svm import (
    TARGET_COLUMN,
    compute_frequency_maps,
    prepare_features,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "DataSet" / "synthetic_fraud_transformed_reducted_scaled_train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "DataSet" / "synthetic_fraud_transformed_reducted_scaled_test.csv"
DEFAULT_REDUCED_PATH = PROJECT_ROOT / "DataSet" / "synthetic_fraud_transformed_reducted.csv"
DEFAULT_SAMPLED_SOURCE_PATH = PROJECT_ROOT / "DataSet" / "synthetic_fraud_transformed_reducted_sampled.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model Training" / "models" / "synthetic_supervised_sgd_svm_model.pkl"
DEFAULT_KERNEL_MODEL_PATH = PROJECT_ROOT / "Model Training" / "models" / "synthetic_supervised_rbf_svm_model.pkl"
DEFAULT_ONECLASS_MODEL_PATH = PROJECT_ROOT / "Model Training" / "models" / "synthetic_oneclass_rbf_svm_model.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Model Training" / "evaluation_outputs_synthetic"

TEST_SIZE = 0.20
SOURCE_SAMPLE_SIZE = 50_000
SOURCE_FRAUD_RATE = 0.05
RANDOM_STATE = 42
RECALL_TARGET = 0.80
FPR_TARGET = 0.02

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the unsupervised fraud anomaly model on the scaled test set."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--reduced", type=Path, default=DEFAULT_REDUCED_PATH)
    parser.add_argument("--sampled-source", type=Path, default=DEFAULT_SAMPLED_SOURCE_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--kernel-model", type=Path, default=DEFAULT_KERNEL_MODEL_PATH)
    parser.add_argument("--oneclass-model", type=Path, default=DEFAULT_ONECLASS_MODEL_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["linear", "kernel", "oneclass", "both", "all"],
        default=["oneclass"],
        help="Which model artifact(s) to evaluate. 'both' means linear+kernel; default evaluates the synthetic one-class model.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--inference-sample-size", type=int, default=10_000)
    return parser.parse_args()


def selected_model_paths(args: argparse.Namespace) -> list[Path]:
    if "all" in args.models:
        selected = {"linear", "kernel", "oneclass"}
    elif "both" in args.models:
        selected = {"linear", "kernel"}
    else:
        selected = set(args.models)
    paths = []
    if "linear" in selected:
        paths.append(args.model)
    if "kernel" in selected:
        paths.append(args.kernel_model)
    if "oneclass" in selected:
        paths.append(args.oneclass_model)
    return paths


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype("string").str.lower()
    return normalized.isin(["true", "1", "yes"])


def load_test_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scaled test dataset not found: {path}")

    return pd.read_csv(
        path,
        dtype={"job": "string", "zip": "string"},
        low_memory=False,
    )


def sample_imbalanced_source(dataframe: pd.DataFrame) -> pd.DataFrame:
    target = parse_bool_series(dataframe[TARGET_COLUMN])
    fraud_count = int(round(SOURCE_SAMPLE_SIZE * SOURCE_FRAUD_RATE))
    non_fraud_count = SOURCE_SAMPLE_SIZE - fraud_count

    fraud_rows = dataframe[target]
    non_fraud_rows = dataframe[~target]

    if len(fraud_rows) < fraud_count:
        raise ValueError(
            f"Not enough fraud rows for requested source sample: "
            f"needed={fraud_count:,}, available={len(fraud_rows):,}"
        )
    if len(non_fraud_rows) < non_fraud_count:
        raise ValueError(
            f"Not enough non-fraud rows for requested source sample: "
            f"needed={non_fraud_count:,}, available={len(non_fraud_rows):,}"
        )

    sampled = pd.concat(
        [
            fraud_rows.sample(n=fraud_count, random_state=RANDOM_STATE),
            non_fraud_rows.sample(n=non_fraud_count, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def load_sampled_source(reduced_path: Path, sampled_source_path: Path) -> pd.DataFrame:
    if sampled_source_path.exists():
        return pd.read_csv(
            sampled_source_path,
            usecols=["amt", TARGET_COLUMN],
            low_memory=False,
        )

    if not reduced_path.exists():
        raise FileNotFoundError(f"Reduced dataset not found: {reduced_path}")

    reduced = pd.read_csv(
        reduced_path,
        usecols=["amt", TARGET_COLUMN],
        low_memory=False,
    )
    return sample_imbalanced_source(reduced)


def reconstruct_original_test_amounts(
    reduced_path: Path,
    sampled_source_path: Path,
    expected_target: pd.Series,
) -> pd.Series:
    reduced = load_sampled_source(reduced_path, sampled_source_path)
    reduced[TARGET_COLUMN] = parse_bool_series(reduced[TARGET_COLUMN])

    _, original_test = train_test_split(
        reduced,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=reduced[TARGET_COLUMN],
    )
    original_test = original_test.reset_index(drop=True)

    if len(original_test) != len(expected_target):
        raise ValueError(
            "Original test reconstruction does not match scaled test row count. "
            f"original={len(original_test):,}, scaled={len(expected_target):,}"
        )

    if not original_test[TARGET_COLUMN].reset_index(drop=True).equals(
        expected_target.reset_index(drop=True)
    ):
        raise ValueError(
            "Original amount reconstruction does not align with scaled test labels. "
            "Re-run PreProcessing/data_scaling.py if the sampled/split files were regenerated differently."
        )

    return original_test["amt"].reset_index(drop=True)


def sample_evaluation_set(
    features: pd.DataFrame,
    target: pd.Series,
    original_amounts: pd.Series,
    sample_size: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if sample_size is None or sample_size >= len(features):
        return (
            features.reset_index(drop=True),
            target.reset_index(drop=True),
            original_amounts.reset_index(drop=True),
        )

    all_index = features.index.to_numpy()
    stratify = target if target.nunique() == 2 else None
    sample_index, _ = train_test_split(
        all_index,
        train_size=sample_size,
        random_state=random_state,
        stratify=stratify,
    )

    return (
        features.loc[sample_index].reset_index(drop=True),
        target.loc[sample_index].reset_index(drop=True),
        original_amounts.loc[sample_index].reset_index(drop=True),
    )


def unpack_model_artifact(loaded_artifact) -> tuple[object, dict]:
    if isinstance(loaded_artifact, dict) and "model" in loaded_artifact:
        return loaded_artifact["model"], loaded_artifact

    return loaded_artifact, {
        "model_type": "unsupervised",
        "threshold": None,
        "feature_columns": None,
    }


def get_feature_columns(
    model,
    artifact: dict,
    prepared_features: pd.DataFrame,
) -> list[str]:
    if artifact.get("feature_columns"):
        return list(artifact["feature_columns"])

    if hasattr(model, "named_steps") and "rbf_sampler" in model.named_steps:
        rbf_sampler = model.named_steps["rbf_sampler"]
        if hasattr(rbf_sampler, "feature_names_in_"):
            return list(rbf_sampler.feature_names_in_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return list(prepared_features.columns)


def score_features(model, features: pd.DataFrame, model_type: str) -> np.ndarray:
    if model_type.startswith("supervised"):
        if hasattr(model, "decision_function"):
            return model.decision_function(features)

        probabilities = model.predict_proba(features)
        return probabilities[:, 1]

    return -model.decision_function(features)


def predict_with_timing(
    model,
    features: pd.DataFrame,
    sample_size: int,
    model_type: str,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    sample = features.head(min(sample_size, len(features)))
    start = time.perf_counter()
    score_features(model, sample, model_type)
    elapsed = time.perf_counter() - start
    per_transaction_ms = (elapsed / len(sample)) * 1000

    scores = score_features(model, features, model_type)
    if model_type.startswith("supervised"):
        if threshold is None:
            predictions = model.predict(features).astype(bool)
        else:
            predictions = scores >= threshold
    else:
        predictions = scores >= (0.0 if threshold is None else threshold)

    return scores, predictions, per_transaction_ms


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    scores: np.ndarray,
    original_amounts: pd.Series,
) -> dict[str, float]:
    y_true_bool = y_true.astype(bool).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true_bool, y_pred, labels=[False, True]).ravel()

    total_fraud_amount = float(original_amounts[y_true_bool].sum())
    captured_fraud_amount = float(original_amounts[y_true_bool & y_pred].sum())
    fraud_loss_capture_rate = (
        captured_fraud_amount / total_fraud_amount if total_fraud_amount else 0.0
    )

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "recall": recall_score(y_true_bool, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": precision_score(y_true_bool, y_pred, zero_division=0),
        "f1_score": f1_score(y_true_bool, y_pred, zero_division=0),
        "auprc": average_precision_score(y_true_bool, scores),
        "roc_auc": roc_auc_score(y_true_bool, scores),
        "predicted_fraud_rate": float(np.mean(y_pred)),
        "total_fraud_amount": total_fraud_amount,
        "captured_fraud_amount": captured_fraud_amount,
        "fraud_loss_capture_rate": fraud_loss_capture_rate,
    }


def build_threshold_report(
    y_true: pd.Series,
    scores: np.ndarray,
    original_amounts: pd.Series,
) -> pd.DataFrame:
    y_true_bool = y_true.astype(bool).to_numpy()
    quantile_thresholds = np.quantile(scores, np.linspace(0.001, 0.999, 240))
    thresholds = np.unique(np.concatenate([quantile_thresholds, np.array([0.0])]))
    rows = []

    for threshold in thresholds:
        y_pred = scores >= threshold
        metrics = calculate_metrics(y_true, y_pred, scores, original_amounts)
        rows.append(
            {
                "threshold": float(threshold),
                "recall": metrics["recall"],
                "false_positive_rate": metrics["false_positive_rate"],
                "precision": metrics["precision"],
                "f1_score": metrics["f1_score"],
                "fraud_loss_capture_rate": metrics["fraud_loss_capture_rate"],
                "predicted_fraud_rate": metrics["predicted_fraud_rate"],
                "fraud_predictions": int(y_pred.sum()),
                "true_fraud_captured": int(np.sum(y_true_bool & y_pred)),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_operating_threshold(threshold_report: pd.DataFrame) -> pd.Series | None:
    candidates = threshold_report[
        (threshold_report["recall"] >= RECALL_TARGET)
        & (threshold_report["false_positive_rate"] <= FPR_TARGET)
    ]
    if candidates.empty:
        return None

    return candidates.sort_values(
        ["f1_score", "precision", "fraud_loss_capture_rate"],
        ascending=False,
    ).iloc[0]


def format_metric_value(metric: str, value: float) -> str:
    if metric in {"tn", "fp", "fn", "tp", "fraud_predictions", "true_fraud_captured"}:
        return f"{int(value):,}"
    if "amount" in metric:
        return f"{value:,.2f}"
    if metric == "inference_time_ms_per_transaction":
        return f"{value:.6f} ms"
    return f"{value:.4f}"


def print_summary_table(metrics: dict[str, float], title: str) -> pd.DataFrame:
    ordered_metrics = [
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "recall",
        "false_positive_rate",
        "precision",
        "f1_score",
        "auprc",
        "roc_auc",
        "predicted_fraud_rate",
        "total_fraud_amount",
        "captured_fraud_amount",
        "fraud_loss_capture_rate",
        "inference_time_ms_per_transaction",
    ]
    summary = pd.DataFrame(
        {
            "metric": ordered_metrics,
            "value": [format_metric_value(metric, metrics[metric]) for metric in ordered_metrics],
        }
    )

    table = Table(title=title, header_style="bold magenta")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for row in summary.itertuples(index=False):
        table.add_row(row.metric, row.value)

    console.print(table)
    return summary


def save_table_png(dataframe: pd.DataFrame, title: str, output_path: Path) -> None:
    figure_height = max(3.4, len(dataframe) * 0.34 + 1.5)
    figure, axis = plt.subplots(figsize=(9.5, figure_height), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.axis("off")
    axis.set_title(title, color="#f4f4f5", fontsize=13, fontstyle="italic", pad=14)

    table = axis.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.28)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d4d8")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#17191c")
            cell.set_text_props(color="#ff66ff", weight="bold")
        else:
            cell.set_facecolor("#111315")
            cell.set_text_props(color="#00d7ff" if column_index == 0 else "#f4f4f5")

    figure.savefig(output_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_confusion_matrix_png(metrics: dict[str, float], output_path: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(6.8, 5.8), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    image = axis.imshow(matrix, cmap="magma")

    axis.set_title("Confusion Matrix", color="#f4f4f5", pad=14)
    axis.set_xlabel("Predicted", color="#f4f4f5")
    axis.set_ylabel("Actual", color="#f4f4f5")
    axis.set_xticks([0, 1], labels=["Not Fraud", "Fraud"])
    axis.set_yticks([0, 1], labels=["Not Fraud", "Fraud"])
    axis.tick_params(colors="#f4f4f5")

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=12,
                fontweight="bold",
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors="#f4f4f5")
    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_precision_recall_curve_png(y_true: pd.Series, scores: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true.astype(bool), scores)
    auprc = average_precision_score(y_true.astype(bool), scores)

    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.plot(recall, precision, color="#00d7ff", linewidth=2.2, label=f"AUPRC = {auprc:.4f}")
    axis.set_title("Precision-Recall Curve", color="#f4f4f5", pad=14)
    axis.set_xlabel("Recall", color="#f4f4f5")
    axis.set_ylabel("Precision", color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(alpha=0.22)
    axis.legend(facecolor="#17191c", edgecolor="#d4d4d8", labelcolor="#f4f4f5")
    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_roc_curve_png(y_true: pd.Series, scores: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true.astype(bool), scores)
    roc_auc = roc_auc_score(y_true.astype(bool), scores)

    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.plot(fpr, tpr, color="#ff66ff", linewidth=2.2, label=f"ROC-AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], color="#d4d4d8", linestyle="--", linewidth=1)
    axis.set_title("ROC Curve", color="#f4f4f5", pad=14)
    axis.set_xlabel("False Positive Rate", color="#f4f4f5")
    axis.set_ylabel("Recall", color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(alpha=0.22)
    axis.legend(facecolor="#17191c", edgecolor="#d4d4d8", labelcolor="#f4f4f5")
    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_threshold_metrics_png(threshold_report: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.6), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")

    x = threshold_report["predicted_fraud_rate"] * 100
    axis.plot(x, threshold_report["recall"] * 100, color="#00d7ff", label="Recall")
    axis.plot(
        x,
        threshold_report["false_positive_rate"] * 100,
        color="#ff66ff",
        label="False Positive Rate",
    )
    axis.plot(x, threshold_report["precision"] * 100, color="#facc15", label="Precision")
    axis.axhline(RECALL_TARGET * 100, color="#00d7ff", linestyle="--", linewidth=1)
    axis.axhline(FPR_TARGET * 100, color="#ff66ff", linestyle="--", linewidth=1)

    axis.set_title("Threshold Tradeoff", color="#f4f4f5", pad=14)
    axis.set_xlabel("Flagged Transaction Rate (%)", color="#f4f4f5")
    axis.set_ylabel("Metric (%)", color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(alpha=0.22)
    axis.legend(facecolor="#17191c", edgecolor="#d4d4d8", labelcolor="#f4f4f5")
    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_fraud_loss_capture_curve_png(
    y_true: pd.Series,
    scores: np.ndarray,
    original_amounts: pd.Series,
    output_path: Path,
) -> None:
    y_true_bool = y_true.astype(bool).to_numpy()
    order = np.argsort(scores)[::-1]
    sorted_fraud_amounts = np.where(y_true_bool[order], original_amounts.to_numpy()[order], 0.0)
    total_fraud_amount = sorted_fraud_amounts.sum()

    flagged_rate = (np.arange(1, len(scores) + 1) / len(scores)) * 100
    if total_fraud_amount:
        captured_rate = (np.cumsum(sorted_fraud_amounts) / total_fraud_amount) * 100
    else:
        captured_rate = np.zeros_like(flagged_rate)

    figure, axis = plt.subplots(figsize=(9, 5.4), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.plot(flagged_rate, captured_rate, color="#00d7ff", linewidth=2.2)
    axis.set_title("Fraud Loss Capture Curve", color="#f4f4f5", pad=14)
    axis.set_xlabel("Flagged Transaction Rate (%)", color="#f4f4f5")
    axis.set_ylabel("Captured Fraud Amount (%)", color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)


def export_pngs(
    output_dir: Path,
    summary: pd.DataFrame,
    metrics: dict[str, float],
    y_true: pd.Series,
    scores: np.ndarray,
    threshold_report: pd.DataFrame,
    original_amounts: pd.Series,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    threshold_report.to_csv(output_dir / "threshold_report.csv", index=False)
    save_table_png(summary, "Model Evaluation Summary", output_dir / "evaluation_summary_table.png")
    save_confusion_matrix_png(metrics, output_dir / "confusion_matrix.png")
    save_precision_recall_curve_png(y_true, scores, output_dir / "precision_recall_curve.png")
    save_roc_curve_png(y_true, scores, output_dir / "roc_curve.png")
    save_threshold_metrics_png(threshold_report, output_dir / "threshold_metrics.png")
    save_fraud_loss_capture_curve_png(
        y_true,
        scores,
        original_amounts,
        output_dir / "fraud_loss_capture_curve.png",
    )


def model_output_name(model_path: Path, artifact: dict) -> str:
    model_type = str(artifact.get("model_type", "")).lower()
    if "oneclass" in model_type or "one_class" in model_type:
        return "oneclass_rbf_svm"
    if "rbf" in model_type or "kernel" in model_type:
        return "kernel_rbf_svm"
    if "sgd" in model_type or "linear" in model_type:
        return "linear_sgd_svm"

    return model_path.stem.replace("supervised_", "").replace("_model", "")


def evaluate_model(
    *,
    model_path: Path,
    base_output_dir: Path,
    test_dataframe: pd.DataFrame,
    prepared_without_order: pd.DataFrame,
    y_true: pd.Series,
    original_amounts: pd.Series,
    frequency_maps: dict[str, dict[str, float]],
    eval_sample_size: int | None,
    random_state: int,
    inference_sample_size: int,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loaded_artifact = load(model_path)
    model, artifact = unpack_model_artifact(loaded_artifact)
    model_type = str(artifact.get("model_type", "unsupervised"))
    threshold = artifact.get("threshold")
    output_dir = base_output_dir / model_output_name(model_path, artifact)

    console.print(
        Panel.fit(
            f"[bold]Model:[/bold] {model_path}\n"
            f"[bold]Model type:[/bold] {model_type}\n"
            f"[bold]Output dir:[/bold] {output_dir}",
            title="Model Evaluation",
            border_style="cyan",
        )
    )

    feature_columns = get_feature_columns(model, artifact, prepared_without_order)
    features, y_true_ordered = prepare_features(
        test_dataframe,
        frequency_maps,
        feature_columns=feature_columns,
    )
    eval_features, eval_target, eval_amounts = sample_evaluation_set(
        features,
        y_true_ordered,
        original_amounts,
        eval_sample_size,
        random_state,
    )
    if eval_sample_size is not None:
        console.log(f"Evaluation sample rows: {len(eval_features):,}")

    scores, predictions, inference_time_ms = predict_with_timing(
        model,
        eval_features,
        inference_sample_size,
        model_type,
        threshold,
    )
    metrics = calculate_metrics(eval_target, predictions, scores, eval_amounts)
    metrics["inference_time_ms_per_transaction"] = inference_time_ms

    threshold_report = build_threshold_report(eval_target, scores, eval_amounts)
    operating_threshold = select_operating_threshold(threshold_report)
    if operating_threshold is None:
        console.print(
            "[yellow]No sampled threshold reached both targets: "
            f"Recall >= {RECALL_TARGET:.0%} and FPR <= {FPR_TARGET:.0%}.[/yellow]"
        )
    else:
        console.print(
            "[green]Candidate threshold reached targets:[/green] "
            f"threshold={operating_threshold['threshold']:.6f}, "
            f"recall={operating_threshold['recall']:.4f}, "
            f"fpr={operating_threshold['false_positive_rate']:.4f}, "
            f"precision={operating_threshold['precision']:.4f}"
        )

    summary = print_summary_table(
        metrics,
        f"{model_output_name(model_path, artifact)} Evaluation Summary",
    )
    export_pngs(
        output_dir,
        summary,
        metrics,
        eval_target,
        scores,
        threshold_report,
        eval_amounts,
    )
    console.print(f"[green]PNG outputs exported:[/green] {output_dir}")


def main() -> None:
    args = parse_args()
    if not args.train.exists():
        raise FileNotFoundError(f"Scaled train dataset not found: {args.train}")
    model_paths = selected_model_paths(args)
    missing_model_paths = [path for path in model_paths if not path.exists()]
    if missing_model_paths:
        missing = ", ".join(str(path) for path in missing_model_paths)
        raise FileNotFoundError(f"Model file(s) not found: {missing}")

    console.print(
        Panel.fit(
            f"[bold]Models:[/bold] {', '.join(str(path) for path in model_paths)}\n"
            f"[bold]Test data:[/bold] {args.test}\n"
            f"[bold]Sampled source:[/bold] {args.sampled_source}\n"
            f"[bold]Output dir:[/bold] {args.output_dir}",
            title="Model Evaluation",
            border_style="cyan",
        )
    )

    frequency_maps, train_rows = compute_frequency_maps(args.train, args.chunksize)
    console.log(f"Frequency maps rebuilt from train rows: {train_rows:,}")

    test_dataframe = load_test_dataset(args.test)
    prepared_without_order, y_true = prepare_features(test_dataframe, frequency_maps)
    if y_true is None:
        raise ValueError(f"{TARGET_COLUMN} column is required for evaluation.")

    original_amounts = reconstruct_original_test_amounts(
        args.reduced,
        args.sampled_source,
        y_true,
    )

    for model_path in model_paths:
        evaluate_model(
            model_path=model_path,
            base_output_dir=args.output_dir,
            test_dataframe=test_dataframe,
            prepared_without_order=prepared_without_order,
            y_true=y_true,
            original_amounts=original_amounts,
            frequency_maps=frequency_maps,
            eval_sample_size=args.eval_sample_size,
            random_state=args.random_state,
            inference_sample_size=args.inference_sample_size,
        )


if __name__ == "__main__":
    main()
