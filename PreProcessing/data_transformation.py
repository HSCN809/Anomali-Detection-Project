import argparse
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
DATASET_PATH = ROOT_DIR / "DataSet" / "synthetic_fraud_cleaned.csv"
TRANSFORMED_DATASET_PATH = ROOT_DIR / "DataSet" / "synthetic_fraud_transformed.csv"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "synthetic_prep_outputs"

TIME_COLUMN = "unix_time"
DOB_COLUMN = "dob"
TARGET_COLUMN = "is_fraud"
ONE_HOT_COLUMNS = ("category", "gender")
STRING_COLUMNS = ("cc_num", "zip")
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6}
EARTH_RADIUS_KM = 6371.0

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform raw fraud transaction data.")
    parser.add_argument("--input", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=TRANSFORMED_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic cleaned dataset not found: {path}\n"
            "Create it first with: python PreProcessing/data_cleaning.py"
        )

    return pd.read_csv(path, low_memory=False)


def resolve_output_dir(explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir
    return OUTPUT_DIR


def add_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    transaction_time = pd.to_datetime(transformed[TIME_COLUMN], unit="s", errors="coerce")

    transformed["transaction_hour"] = transaction_time.dt.hour.astype("Int8")
    transformed["transaction_day_of_week"] = transaction_time.dt.dayofweek.astype("Int8")
    transformed["transaction_day_of_month"] = transaction_time.dt.day.astype("Int8")
    transformed["transaction_month"] = transaction_time.dt.month.astype("Int8")
    transformed["is_weekend"] = transformed["transaction_day_of_week"].isin([5, 6])
    transformed["is_night"] = transformed["transaction_hour"].isin(NIGHT_HOURS)

    return transformed


def add_age_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    transaction_time = pd.to_datetime(transformed[TIME_COLUMN], unit="s", errors="coerce")
    birth_date = pd.to_datetime(transformed[DOB_COLUMN], errors="coerce")

    age = ((transaction_time - birth_date).dt.days / 365.25).round()
    transformed["customer_age"] = age.fillna(age.median()).astype("int64")

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
        dtype=bool,
    )


def apply_dtype_conversions(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()

    for column in STRING_COLUMNS:
        if column in transformed.columns:
            transformed[column] = transformed[column].astype("string")

    if TARGET_COLUMN in transformed.columns:
        transformed[TARGET_COLUMN] = transformed[TARGET_COLUMN].astype(bool)

    return transformed


def transform_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = apply_dtype_conversions(dataframe)
    transformed = add_time_features(transformed)
    transformed = add_age_feature(transformed)
    transformed = add_distance_feature(transformed)
    return encode_categorical_features(transformed)


def save_transformed_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    temp_output_path = output_path.with_suffix(".tmp.csv")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(temp_output_path, index=False)
        temp_output_path.replace(output_path)
    except PermissionError as exc:
        try:
            temp_output_path.unlink(missing_ok=True)
        except PermissionError:
            pass

        raise SystemExit(
            "Could not write transformed dataset. Close any application that may "
            f"be using this file, then run the script again: {output_path}"
        ) from exc


def print_summary_table(summary: pd.DataFrame, title: str) -> None:
    table = Table(title=title, header_style="bold magenta")
    for column in summary.columns:
        table.add_column(str(column))

    for row in summary.itertuples(index=False):
        table.add_row(*(str(value) for value in row))

    console.print(table)


def save_table_png(summary: pd.DataFrame, title: str, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

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
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

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


def save_distance_histogram(dataframe: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transformation_distance_distribution.png"

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


def export_transformation_analysis(
    dataframe: pd.DataFrame,
    output_path: Path,
    output_dir: Path,
) -> None:
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "created_time_features",
                "created_age_features",
                "created_distance_features",
                "created_encoded_features",
                "output_file",
            ],
            "value": [
                f"{len(dataframe):,}",
                f"{len(dataframe.columns):,}",
                "6 columns",
                "1 column",
                "1 column",
                f"{len([column for column in dataframe.columns if column.startswith(('category_', 'gender_'))])} one-hot columns",
                str(output_path),
            ],
        }
    )
    print_summary_table(summary, "Data Transformation Summary")
    save_table_png(summary, "Data Transformation Summary", "transformation_summary.png", output_dir)

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
        output_dir,
    )

    hourly_fraud_rate = dataframe.groupby("transaction_hour")[TARGET_COLUMN].mean()
    save_bar_chart(
        hourly_fraud_rate,
        title="Fraud Rate by Transaction Hour",
        xlabel="Transaction Hour",
        ylabel="Fraud Rate",
        filename="transformation_fraud_rate_by_hour.png",
        output_dir=output_dir,
    )

    day_fraud_rate = dataframe.groupby("transaction_day_of_week")[TARGET_COLUMN].mean()
    day_fraud_rate = day_fraud_rate.reindex(range(7), fill_value=0)
    day_fraud_rate.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    save_bar_chart(
        day_fraud_rate,
        title="Fraud Rate by Day of Week",
        xlabel="Day of Week",
        ylabel="Fraud Rate",
        filename="transformation_fraud_rate_by_day_of_week.png",
        output_dir=output_dir,
    )

    distance_bucket_fraud_rate = build_distance_bucket_fraud_rate(dataframe)
    save_bar_chart(
        distance_bucket_fraud_rate,
        title="Fraud Rate by Customer-Merchant Distance",
        xlabel="Distance Bucket (km)",
        ylabel="Fraud Rate",
        filename="transformation_fraud_rate_by_distance_bucket.png",
        output_dir=output_dir,
    )
    save_distance_histogram(dataframe, output_dir)


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.input)
    output_dir = resolve_output_dir(args.output_dir)
    transformed_dataframe = transform_dataset(dataframe)
    save_transformed_dataset(transformed_dataframe, args.output)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {args.input}\n"
            f"[bold]Output:[/bold] {args.output}\n"
            f"[bold]Rows:[/bold] {len(transformed_dataframe):,}\n"
            f"[bold]Columns:[/bold] {len(transformed_dataframe.columns):,}",
            title="Data Transformation",
            border_style="cyan",
        )
    )
    export_transformation_analysis(transformed_dataframe, args.output, output_dir)


if __name__ == "__main__":
    main()
