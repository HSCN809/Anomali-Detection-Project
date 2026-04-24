from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_transformed_reducted.csv"
SAMPLED_OUTPUT_PATH = ROOT_DIR / "DataSet" / "fraud_transformed_reducted_sampled_50k.csv"
TRAIN_OUTPUT_PATH = ROOT_DIR / "DataSet" / "fraud_transformed_reducted_scaled_train.csv"
TEST_OUTPUT_PATH = ROOT_DIR / "DataSet" / "fraud_transformed_reducted_scaled_test.csv"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "prep_outputs"

TARGET_COLUMN = "is_fraud"
SAMPLE_SIZE = 50_000
FRAUD_SAMPLE_RATE = 0.05
TEST_SIZE = 0.20
RANDOM_STATE = 42

CYCLIC_COLUMNS = {
    "transaction_hour": 24,
    "transaction_day_of_week": 7,
    "transaction_month": 12,
}
LOG_ROBUST_COLUMNS = [
    "amt",
    "city_pop",
    "customer_merchant_distance_km",
]
STANDARD_SCALE_COLUMNS = ["customer_age"]
MINMAX_SCALE_COLUMNS = ["transaction_day_of_month"]
READ_DTYPES = {
    "zip": "string",
}

console = Console()


def load_dataset() -> pd.DataFrame:
    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Reduced dataset not found: {INPUT_DATASET_PATH}\n"
            "Create it first with: python PreProcessing/data_reduction.py"
        )

    return pd.read_csv(INPUT_DATASET_PATH, dtype=READ_DTYPES, low_memory=False)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype("string").str.lower()
    return normalized.isin(["true", "1", "yes"])


def sample_imbalanced_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    target = parse_bool_series(dataframe[TARGET_COLUMN])
    fraud_count = int(round(SAMPLE_SIZE * FRAUD_SAMPLE_RATE))
    non_fraud_count = SAMPLE_SIZE - fraud_count

    fraud_rows = dataframe[target]
    non_fraud_rows = dataframe[~target]

    if len(fraud_rows) < fraud_count:
        raise ValueError(
            f"Not enough fraud rows for requested sample: "
            f"needed={fraud_count:,}, available={len(fraud_rows):,}"
        )
    if len(non_fraud_rows) < non_fraud_count:
        raise ValueError(
            f"Not enough non-fraud rows for requested sample: "
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


def add_cyclic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()

    for column, period in CYCLIC_COLUMNS.items():
        radians = 2 * np.pi * transformed[column] / period
        transformed[f"{column}_sin"] = np.sin(radians)
        transformed[f"{column}_cos"] = np.cos(radians)

    transformed = transformed.drop(columns=list(CYCLIC_COLUMNS))
    return transformed


def split_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe[TARGET_COLUMN],
    )


def apply_scaling(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled_train = train_dataframe.copy()
    scaled_test = test_dataframe.copy()

    log_train = np.log1p(scaled_train[LOG_ROBUST_COLUMNS])
    log_test = np.log1p(scaled_test[LOG_ROBUST_COLUMNS])
    robust_scaler = RobustScaler()
    scaled_train[LOG_ROBUST_COLUMNS] = robust_scaler.fit_transform(log_train)
    scaled_test[LOG_ROBUST_COLUMNS] = robust_scaler.transform(log_test)

    standard_scaler = StandardScaler()
    scaled_train[STANDARD_SCALE_COLUMNS] = standard_scaler.fit_transform(
        scaled_train[STANDARD_SCALE_COLUMNS]
    )
    scaled_test[STANDARD_SCALE_COLUMNS] = standard_scaler.transform(
        scaled_test[STANDARD_SCALE_COLUMNS]
    )

    minmax_scaler = MinMaxScaler()
    scaled_train[MINMAX_SCALE_COLUMNS] = minmax_scaler.fit_transform(
        scaled_train[MINMAX_SCALE_COLUMNS]
    )
    scaled_test[MINMAX_SCALE_COLUMNS] = minmax_scaler.transform(
        scaled_test[MINMAX_SCALE_COLUMNS]
    )

    return scaled_train, scaled_test


def save_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    temp_output_path = output_path.with_suffix(".tmp.csv")

    try:
        dataframe.to_csv(temp_output_path, index=False)
        temp_output_path.replace(output_path)
    except PermissionError as exc:
        try:
            temp_output_path.unlink(missing_ok=True)
        except PermissionError:
            pass

        raise SystemExit(
            "Could not write scaled dataset. Close any application that may "
            f"be using this file, then run the script again: {output_path}"
        ) from exc


def print_table(dataframe: pd.DataFrame, title: str) -> None:
    table = Table(title=title, header_style="bold magenta")
    for column in dataframe.columns:
        table.add_column(str(column))

    for row in dataframe.itertuples(index=False):
        table.add_row(*(str(value) for value in row))

    console.print(table)


def save_table_png(dataframe: pd.DataFrame, title: str, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    figure_height = max(3.5, len(dataframe) * 0.36 + 1.4)
    figure_width = max(8, len(dataframe.columns) * 2.8)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.axis("off")
    axis.set_title(title, color="#f4f4f5", fontsize=12, fontstyle="italic", pad=14)

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
            continue

        cell.set_facecolor("#111315")
        cell.set_text_props(color="#00d7ff" if column_index == 0 else "#f4f4f5")

    figure.savefig(output_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    console.print(f"[green]PNG exported:[/green] {output_path}")


def save_class_distribution_png(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "scaling_train_test_class_distribution.png"

    plot_data = summary.pivot(index="split", columns="is_fraud", values="percent")
    figure, axis = plt.subplots(figsize=(8.5, 5), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    plot_data.plot(kind="bar", ax=axis, color=["#00d7ff", "#ff66ff"])

    axis.set_title("Train/Test Class Distribution", color="#f4f4f5", pad=14)
    axis.set_xlabel("Split", color="#f4f4f5")
    axis.set_ylabel("Class Percent", color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5", rotation=0)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(title="is_fraud")

    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f%%", color="#f4f4f5", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    console.print(f"[green]PNG exported:[/green] {output_path}")


def build_scaling_summary(
    source_dataframe: pd.DataFrame,
    sampled_dataframe: pd.DataFrame,
    encoded_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    sampled_target = parse_bool_series(sampled_dataframe[TARGET_COLUMN])

    return pd.DataFrame(
        {
            "metric": [
                "input_rows",
                "sampled_rows",
                "sampled_fraud_rows",
                "sampled_non_fraud_rows",
                "sampled_fraud_percent",
                "source_columns",
                "post_cyclic_columns",
                "encoded_columns",
                "sample_output",
                "train_rows",
                "test_rows",
                "train_output",
                "test_output",
            ],
            "value": [
                f"{len(source_dataframe):,}",
                f"{len(sampled_dataframe):,}",
                f"{int(sampled_target.sum()):,}",
                f"{int((~sampled_target).sum()):,}",
                f"{sampled_target.mean() * 100:.2f}%",
                f"{len(source_dataframe.columns):,}",
                f"{len(encoded_dataframe.columns):,}",
                "6 cyclic columns",
                "DataSet/fraud_transformed_reducted_sampled_50k.csv",
                f"{len(train_dataframe):,}",
                f"{len(test_dataframe):,}",
                "DataSet/fraud_transformed_reducted_scaled_train.csv",
                "DataSet/fraud_transformed_reducted_scaled_test.csv",
            ],
        }
    )


def build_operation_report() -> pd.DataFrame:
    rows = []

    for column in CYCLIC_COLUMNS:
        rows.append(
            {
                "column": column,
                "operation": "cyclic encoding",
                "fit_scope": "sampled dataset",
            }
        )

    for column in LOG_ROBUST_COLUMNS:
        rows.append(
            {
                "column": column,
                "operation": "log1p + RobustScaler",
                "fit_scope": "train only",
            }
        )

    for column in STANDARD_SCALE_COLUMNS:
        rows.append(
            {
                "column": column,
                "operation": "StandardScaler",
                "fit_scope": "train only",
            }
        )

    for column in MINMAX_SCALE_COLUMNS:
        rows.append(
            {
                "column": column,
                "operation": "MinMaxScaler",
                "fit_scope": "train only",
            }
        )

    return pd.DataFrame(rows)


def build_class_distribution(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for split_name, dataframe in [("train", train_dataframe), ("test", test_dataframe)]:
        counts = dataframe[TARGET_COLUMN].value_counts().sort_index()
        percents = dataframe[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

        for class_value in counts.index:
            rows.append(
                {
                    "split": split_name,
                    "is_fraud": str(class_value),
                    "count": f"{counts[class_value]:,}",
                    "percent": round(float(percents[class_value]), 4),
                }
            )

    return pd.DataFrame(rows)


def export_scaling_analysis(
    source_dataframe: pd.DataFrame,
    sampled_dataframe: pd.DataFrame,
    encoded_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    summary = build_scaling_summary(
        source_dataframe,
        sampled_dataframe,
        encoded_dataframe,
        train_dataframe,
        test_dataframe,
    )
    operation_report = build_operation_report()
    class_distribution = build_class_distribution(train_dataframe, test_dataframe)

    print_table(summary, "Data Scaling Summary")
    save_table_png(summary, "Data Scaling Summary", "scaling_summary.png")

    print_table(operation_report, "Data Scaling Operation Report")
    save_table_png(
        operation_report,
        "Data Scaling Operation Report",
        "scaling_operation_report.png",
    )

    print_table(class_distribution, "Train/Test Class Distribution")
    save_table_png(
        class_distribution,
        "Train/Test Class Distribution",
        "scaling_train_test_class_distribution_table.png",
    )
    save_class_distribution_png(class_distribution)


def main() -> None:
    dataframe = load_dataset()
    sampled_dataframe = sample_imbalanced_dataset(dataframe)
    encoded_dataframe = add_cyclic_features(sampled_dataframe)
    train_dataframe, test_dataframe = split_dataset(encoded_dataframe)
    scaled_train, scaled_test = apply_scaling(train_dataframe, test_dataframe)

    save_dataset(sampled_dataframe, SAMPLED_OUTPUT_PATH)
    save_dataset(scaled_train, TRAIN_OUTPUT_PATH)
    save_dataset(scaled_test, TEST_OUTPUT_PATH)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {INPUT_DATASET_PATH}\n"
            f"[bold]Sample output:[/bold] {SAMPLED_OUTPUT_PATH}\n"
            f"[bold]Train output:[/bold] {TRAIN_OUTPUT_PATH}\n"
            f"[bold]Test output:[/bold] {TEST_OUTPUT_PATH}\n"
            f"[bold]Sample:[/bold] {SAMPLE_SIZE:,} rows, {FRAUD_SAMPLE_RATE:.0%} fraud\n"
            f"[bold]Split:[/bold] 80/20 stratified",
            title="Data Scaling",
            border_style="cyan",
        )
    )
    export_scaling_analysis(
        dataframe,
        sampled_dataframe,
        encoded_dataframe,
        scaled_train,
        scaled_test,
    )


if __name__ == "__main__":
    main()
