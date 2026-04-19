from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_merged.csv"
TRANSFORMED_DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_transformed.csv"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "prep_outputs"

TIME_COLUMN = "trans_date_trans_time"
TARGET_COLUMN = "is_fraud"
ONE_HOT_COLUMNS = ("category", "gender")
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
EARTH_RADIUS_KM = 6371.0

console = Console()


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {DATASET_PATH}\n"
            "Create it first with: python DataSet/merge_fraud_datasets.py"
        )

    return pd.read_csv(DATASET_PATH, low_memory=False)


def add_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    transaction_time = pd.to_datetime(transformed[TIME_COLUMN], errors="coerce")

    transformed["transaction_hour"] = transaction_time.dt.hour.astype("Int8")
    transformed["transaction_day_of_week"] = transaction_time.dt.dayofweek.astype("Int8")
    transformed["transaction_day_of_month"] = transaction_time.dt.day.astype("Int8")
    transformed["transaction_month"] = transaction_time.dt.month.astype("Int8")
    transformed["is_weekend"] = (
        transformed["transaction_day_of_week"].isin([5, 6]).astype("int8")
    )
    transformed["is_night"] = (
        transformed["transaction_hour"].isin(NIGHT_HOURS).astype("int8")
    )

    return transformed


def add_distance_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()

    customer_latitude = np.radians(transformed["lat"].astype(float))
    customer_longitude = np.radians(transformed["long"].astype(float))
    merchant_latitude = np.radians(transformed["merch_lat"].astype(float))
    merchant_longitude = np.radians(transformed["merch_long"].astype(float))

    latitude_delta = merchant_latitude - customer_latitude
    longitude_delta = merchant_longitude - customer_longitude

    haversine_a = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(customer_latitude)
        * np.cos(merchant_latitude)
        * np.sin(longitude_delta / 2) ** 2
    )
    haversine_c = 2 * np.arcsin(np.sqrt(np.clip(haversine_a, 0, 1)))
    transformed["customer_merchant_distance_km"] = EARTH_RADIUS_KM * haversine_c

    return transformed


def encode_categorical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        dataframe,
        columns=list(ONE_HOT_COLUMNS),
        prefix=list(ONE_HOT_COLUMNS),
        dtype="int8",
    )


def transform_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.drop(columns=["Unnamed: 0"], errors="ignore")
    transformed = add_time_features(transformed)
    transformed = add_distance_feature(transformed)
    transformed = encode_categorical_features(transformed)

    return transformed


def save_transformed_dataset(dataframe: pd.DataFrame) -> None:
    temp_output_path = TRANSFORMED_DATASET_PATH.with_suffix(".tmp.csv")

    try:
        dataframe.to_csv(temp_output_path, index=False)
        temp_output_path.replace(TRANSFORMED_DATASET_PATH)
    except PermissionError as exc:
        try:
            temp_output_path.unlink(missing_ok=True)
        except PermissionError:
            pass

        raise SystemExit(
            "Could not write transformed dataset. Close any application that may "
            f"be using this file, then run the script again: {TRANSFORMED_DATASET_PATH}"
        ) from exc


def print_summary_table(summary: pd.DataFrame, title: str) -> None:
    table = Table(title=title, header_style="bold magenta")
    for column in summary.columns:
        table.add_column(str(column))

    for row in summary.itertuples(index=False):
        table.add_row(*(str(value) for value in row))

    console.print(table)


def save_table_png(summary: pd.DataFrame, title: str, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    figure_height = max(3.5, len(summary) * 0.38 + 1.2)
    figure_width = max(8, len(summary.columns) * 2.4)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.axis("off")
    axis.set_title(title, color="#f4f4f5", fontsize=12, fontstyle="italic", pad=14)

    table = axis.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

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


def save_bar_chart(
    data: pd.Series,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    bars = axis.bar(data.index.astype(str), data.values, color="#00d7ff")

    axis.set_title(title, color="#f4f4f5", fontsize=13, pad=14)
    axis.set_xlabel(xlabel, color="#f4f4f5")
    axis.set_ylabel(ylabel, color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(axis="y", alpha=0.22)

    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            color="#f4f4f5",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    console.print(f"[green]PNG exported:[/green] {output_path}")


def save_distance_histogram(dataframe: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "transformation_distance_distribution.png"

    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.hist(
        dataframe["customer_merchant_distance_km"],
        bins=40,
        color="#00d7ff",
        edgecolor="#111315",
    )
    axis.set_title("Customer-Merchant Distance Distribution", color="#f4f4f5", pad=14)
    axis.set_xlabel("Distance (km)", color="#f4f4f5")
    axis.set_ylabel("Transaction Count", color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(axis="y", alpha=0.22)

    figure.tight_layout()
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    console.print(f"[green]PNG exported:[/green] {output_path}")


def build_distance_bucket_fraud_rate(dataframe: pd.DataFrame) -> pd.Series:
    distance_bins = [0, 10, 25, 50, 100, 250, np.inf]
    distance_labels = ["0-10", "10-25", "25-50", "50-100", "100-250", "250+"]
    distance_bucket = pd.cut(
        dataframe["customer_merchant_distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )

    return dataframe.groupby(distance_bucket, observed=True)[TARGET_COLUMN].mean()


def export_transformation_analysis(dataframe: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "created_time_features",
                "created_distance_features",
                "created_encoded_features",
                "output_file",
            ],
            "value": [
                f"{len(dataframe):,}",
                f"{len(dataframe.columns):,}",
                "6 columns",
                "1 column",
                f"{len([column for column in dataframe.columns if column.startswith(('category_', 'gender_'))])} one-hot columns",
                "DataSet/fraud_transformed.csv",
            ],
        }
    )
    print_summary_table(summary, "Data Transformation Summary")
    save_table_png(summary, "Data Transformation Summary", "transformation_summary.png")

    encoded_columns = pd.DataFrame(
        {
            "encoded_column": [
                column
                for column in dataframe.columns
                if column.startswith(("category_", "gender_"))
            ]
        }
    )
    print_summary_table(encoded_columns, "Encoded Columns")
    save_table_png(
        encoded_columns,
        "Encoded Columns",
        "transformation_encoded_columns.png",
    )

    hourly_fraud_rate = dataframe.groupby("transaction_hour")[TARGET_COLUMN].mean()
    save_bar_chart(
        hourly_fraud_rate,
        title="Fraud Rate by Transaction Hour",
        xlabel="Transaction Hour",
        ylabel="Fraud Rate",
        filename="transformation_fraud_rate_by_hour.png",
    )

    day_fraud_rate = dataframe.groupby("transaction_day_of_week")[TARGET_COLUMN].mean()
    day_fraud_rate.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    save_bar_chart(
        day_fraud_rate,
        title="Fraud Rate by Day of Week",
        xlabel="Day of Week",
        ylabel="Fraud Rate",
        filename="transformation_fraud_rate_by_day_of_week.png",
    )

    distance_bucket_fraud_rate = build_distance_bucket_fraud_rate(dataframe)
    save_bar_chart(
        distance_bucket_fraud_rate,
        title="Fraud Rate by Customer-Merchant Distance",
        xlabel="Distance Bucket (km)",
        ylabel="Fraud Rate",
        filename="transformation_fraud_rate_by_distance_bucket.png",
    )
    save_distance_histogram(dataframe)


def main() -> None:
    dataframe = load_dataset()
    transformed_dataframe = transform_dataset(dataframe)
    save_transformed_dataset(transformed_dataframe)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {DATASET_PATH}\n"
            f"[bold]Output:[/bold] {TRANSFORMED_DATASET_PATH}\n"
            f"[bold]Rows:[/bold] {len(transformed_dataframe):,}\n"
            f"[bold]Columns:[/bold] {len(transformed_dataframe.columns):,}",
            title="Data Transformation",
            border_style="cyan",
        )
    )
    export_transformation_analysis(transformed_dataframe)


if __name__ == "__main__":
    main()
