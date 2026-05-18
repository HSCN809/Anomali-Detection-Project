import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "DataSet" / "synthetic_fraud_transformed.csv"
OUTPUT_DIR = ROOT_DIR / "EDA" / "synthetic_eda_outputs"

TARGET_COLUMN = "is_fraud"
CATEGORY_PREFIX = "category_"
SCATTER_SAMPLE_SIZE = 60_000
RANDOM_STATE = 42

console = Console()
sns.set_theme(style="darkgrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EDA charts for transformed fraud data.")
    parser.add_argument("--input", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def get_category_columns(dataset_path: Path) -> list[str]:
    columns = pd.read_csv(dataset_path, nrows=0).columns
    return [column for column in columns if column.startswith(CATEGORY_PREFIX)]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Transformed dataset not found: {dataset_path}\n"
            "Create it first with: python PreProcessing/data_transformation.py"
        )

    category_columns = get_category_columns(dataset_path)
    usecols = [
        "amt",
        TARGET_COLUMN,
        "customer_merchant_distance_km",
        "transaction_hour",
        "transaction_day_of_week",
        "transaction_month",
        "is_night",
        "is_weekend",
        "customer_age",
        "city_pop",
        "gender_F",
        "gender_M",
        *category_columns,
    ]
    return pd.read_csv(dataset_path, usecols=usecols, low_memory=False)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str, output_dir: Path) -> None:
    output_path = output_dir / filename
    ensure_output_dir(output_dir)
    plt.savefig(output_path, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    console.print(f"[green]PNG exported:[/green] {output_path}")


def style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, color="#f4f4f5", fontsize=14, pad=14)
    axis.set_xlabel(xlabel, color="#f4f4f5")
    axis.set_ylabel(ylabel, color="#f4f4f5")
    axis.tick_params(colors="#f4f4f5")
    axis.grid(alpha=0.22)
    axis.set_facecolor("#111315")


def create_figure(width: float = 10, height: float = 5.8):
    figure, axis = plt.subplots(figsize=(width, height), dpi=150)
    figure.patch.set_facecolor("#111315")
    return figure, axis


def save_table_png(dataframe: pd.DataFrame, title: str, filename: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    output_path = output_dir / filename

    figure_height = max(3.5, len(dataframe) * 0.38 + 1.4)
    figure_width = max(8, len(dataframe.columns) * 2.7)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.axis("off")
    axis.set_title(title, color="#f4f4f5", fontsize=13, fontstyle="italic", pad=14)

    table = axis.table(
        cellText=dataframe.astype(str).values,
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


def print_table(dataframe: pd.DataFrame, title: str) -> None:
    table = Table(title=title, header_style="bold magenta")
    for column in dataframe.columns:
        table.add_column(str(column))

    for row in dataframe.itertuples(index=False):
        table.add_row(*(str(value) for value in row))

    console.print(table)


def add_category_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    category_columns = [
        column for column in transformed.columns if column.startswith(CATEGORY_PREFIX)
    ]
    category_values = transformed[category_columns].idxmax(axis=1)
    transformed["category"] = category_values.str.replace(CATEGORY_PREFIX, "", regex=False)
    return transformed


def plot_target_distribution(dataframe: pd.DataFrame, output_dir: Path) -> None:
    summary = dataframe[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

    _, axis = create_figure(8, 5)
    bars = axis.bar(summary.index.astype(str), summary.values, color=["#00d7ff", "#ff66ff"])
    style_axis(axis, "Target Distribution", "is_fraud", "Percent")

    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color="#f4f4f5",
            fontsize=9,
        )

    save_current_figure("eda_target_distribution.png", output_dir)


def plot_amount_distribution(dataframe: pd.DataFrame, output_dir: Path) -> None:
    amount_cap = dataframe["amt"].quantile(0.995)

    _, axis = create_figure()
    axis.hist(dataframe["amt"].clip(upper=amount_cap), bins=80, color="#00d7ff")
    style_axis(
        axis,
        "Transaction Amount Distribution (capped at 99.5th percentile)",
        "Amount",
        "Transaction Count",
    )
    save_current_figure("eda_amt_distribution.png", output_dir)

    _, axis = create_figure()
    axis.hist(np.log1p(dataframe["amt"]), bins=80, color="#00d7ff")
    style_axis(axis, "Log Transaction Amount Distribution", "log1p(amount)", "Count")
    save_current_figure("eda_amt_log_distribution.png", output_dir)


def plot_amount_by_fraud(dataframe: pd.DataFrame, output_dir: Path) -> None:
    plot_data = dataframe[[TARGET_COLUMN, "amt"]].copy()
    plot_data["log_amount"] = np.log1p(plot_data["amt"])

    _, axis = create_figure(8, 5.8)
    sns.boxplot(
        data=plot_data,
        x=TARGET_COLUMN,
        y="log_amount",
        hue=TARGET_COLUMN,
        palette={False: "#00d7ff", True: "#ff66ff"},
        legend=False,
        ax=axis,
    )
    style_axis(axis, "Transaction Amount by Fraud Status", "is_fraud", "log1p(amount)")
    save_current_figure("eda_amt_by_fraud_boxplot.png", output_dir)


def export_amount_summary(dataframe: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        dataframe.groupby(TARGET_COLUMN)["amt"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            q90=lambda values: values.quantile(0.90),
            q95=lambda values: values.quantile(0.95),
            q99=lambda values: values.quantile(0.99),
            max="max",
        )
        .reset_index()
    )
    summary["is_fraud"] = summary["is_fraud"].astype(str)
    numeric_columns = ["mean", "median", "q90", "q95", "q99", "max"]
    summary[numeric_columns] = summary[numeric_columns].round(2)

    print_table(summary, "Amount Summary by Fraud Status")
    save_table_png(summary, "Amount Summary by Fraud Status", "eda_amt_summary_table.png", output_dir)


def build_category_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    category_data = add_category_column(dataframe)
    summary = (
        category_data.groupby("category")[TARGET_COLUMN]
        .agg(transaction_count="count", fraud_count="sum", fraud_rate="mean")
        .sort_values("fraud_rate", ascending=False)
        .reset_index()
    )
    summary["fraud_rate_percent"] = (summary["fraud_rate"] * 100).round(4)
    return summary.drop(columns=["fraud_rate"])


def plot_category_analysis(dataframe: pd.DataFrame, output_dir: Path) -> None:
    summary = build_category_summary(dataframe)

    _, axis = create_figure(12, 6.2)
    axis.bar(summary["category"], summary["fraud_rate_percent"], color="#00d7ff")
    style_axis(axis, "Fraud Rate by Spending Category", "Category", "Fraud Rate (%)")
    axis.tick_params(axis="x", rotation=45)
    save_current_figure("eda_category_fraud_rate.png", output_dir)

    count_summary = summary.sort_values("transaction_count", ascending=False)
    x_positions = np.arange(len(count_summary))
    width = 0.42

    _, axis = create_figure(12, 6.2)
    axis.bar(
        x_positions - width / 2,
        count_summary["transaction_count"],
        width,
        label="transactions",
        color="#00d7ff",
    )
    axis.bar(
        x_positions + width / 2,
        count_summary["fraud_count"],
        width,
        label="frauds",
        color="#ff66ff",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(count_summary["category"], rotation=45, ha="right")
    axis.legend()
    style_axis(axis, "Transaction and Fraud Counts by Category", "Category", "Count")
    save_current_figure("eda_category_transaction_counts.png", output_dir)

    table_summary = summary.copy()
    table_summary["transaction_count"] = table_summary["transaction_count"].map("{:,}".format)
    table_summary["fraud_count"] = table_summary["fraud_count"].map("{:,}".format)
    save_table_png(
        table_summary,
        "Category Fraud Summary",
        "eda_category_fraud_summary_table.png",
        output_dir,
    )


def plot_distance_distribution(dataframe: pd.DataFrame, output_dir: Path) -> None:
    distance_cap = dataframe["customer_merchant_distance_km"].quantile(0.995)

    _, axis = create_figure()
    axis.hist(
        dataframe["customer_merchant_distance_km"].clip(upper=distance_cap),
        bins=80,
        color="#00d7ff",
    )
    style_axis(
        axis,
        "Customer-Merchant Distance Distribution (capped at 99.5th percentile)",
        "Distance (km)",
        "Transaction Count",
    )
    save_current_figure("eda_distance_distribution.png", output_dir)


def plot_distance_by_fraud(dataframe: pd.DataFrame, output_dir: Path) -> None:
    plot_data = dataframe[[TARGET_COLUMN, "customer_merchant_distance_km"]].copy()

    _, axis = create_figure(8, 5.8)
    sns.boxplot(
        data=plot_data,
        x=TARGET_COLUMN,
        y="customer_merchant_distance_km",
        hue=TARGET_COLUMN,
        palette={False: "#00d7ff", True: "#ff66ff"},
        legend=False,
        ax=axis,
    )
    style_axis(
        axis,
        "Distance by Fraud Status",
        "is_fraud",
        "Customer-Merchant Distance (km)",
    )
    save_current_figure("eda_distance_by_fraud_boxplot.png", output_dir)


def plot_distance_bucket_fraud_rate(dataframe: pd.DataFrame, output_dir: Path) -> None:
    distance_bins = [0, 10, 25, 50, 100, 250, np.inf]
    distance_labels = ["0-10", "10-25", "25-50", "50-100", "100-250", "250+"]
    bucket = pd.cut(
        dataframe["customer_merchant_distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )
    fraud_rate = dataframe.groupby(bucket, observed=True)[TARGET_COLUMN].mean() * 100

    _, axis = create_figure(9, 5.5)
    bars = axis.bar(fraud_rate.index.astype(str), fraud_rate.values, color="#00d7ff")
    style_axis(
        axis,
        "Fraud Rate by Customer-Merchant Distance",
        "Distance Bucket (km)",
        "Fraud Rate (%)",
    )
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color="#f4f4f5",
            fontsize=8,
        )
    save_current_figure("eda_distance_bucket_fraud_rate.png", output_dir)


def plot_amount_vs_distance_sample(dataframe: pd.DataFrame, output_dir: Path) -> None:
    fraud_rows = dataframe[dataframe[TARGET_COLUMN]]
    non_fraud_rows = dataframe[~dataframe[TARGET_COLUMN]]
    non_fraud_sample = non_fraud_rows.sample(
        n=min(SCATTER_SAMPLE_SIZE, (~dataframe[TARGET_COLUMN]).sum()),
        random_state=RANDOM_STATE,
    )
    sample = pd.concat([fraud_rows, non_fraud_sample], ignore_index=True)

    _, axis = create_figure(10, 6)
    sns.scatterplot(
        data=sample,
        x="customer_merchant_distance_km",
        y=np.log1p(sample["amt"]),
        hue=TARGET_COLUMN,
        palette={False: "#00d7ff", True: "#ff66ff"},
        alpha=0.42,
        s=10,
        ax=axis,
    )
    style_axis(
        axis,
        "Sampled Amount vs Distance by Fraud Status",
        "Customer-Merchant Distance (km)",
        "log1p(amount)",
    )
    save_current_figure("eda_amt_vs_distance_sample.png", output_dir)


def plot_hourly_fraud_rate(dataframe: pd.DataFrame, output_dir: Path) -> None:
    fraud_rate = dataframe.groupby("transaction_hour")[TARGET_COLUMN].mean() * 100

    _, axis = create_figure(11, 5.8)
    bars = axis.bar(fraud_rate.index.astype(str), fraud_rate.values, color="#00d7ff")
    style_axis(axis, "Fraud Rate by Transaction Hour", "Transaction Hour", "Fraud Rate (%)")
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color="#f4f4f5",
            fontsize=7,
        )
    save_current_figure("eda_hourly_fraud_rate.png", output_dir)


def plot_is_night_fraud_rate(dataframe: pd.DataFrame, output_dir: Path) -> None:
    fraud_rate = dataframe.groupby("is_night")[TARGET_COLUMN].mean() * 100

    _, axis = create_figure(7.5, 5)
    bars = axis.bar(fraud_rate.index.astype(str), fraud_rate.values, color="#00d7ff")
    style_axis(axis, "Fraud Rate by Night Flag", "is_night", "Fraud Rate (%)")
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color="#f4f4f5",
            fontsize=9,
        )
    save_current_figure("eda_is_night_fraud_rate.png", output_dir)


def plot_correlation_heatmap(dataframe: pd.DataFrame, output_dir: Path) -> None:
    correlation_columns = [
        "amt",
        "customer_merchant_distance_km",
        "transaction_hour",
        "transaction_day_of_week",
        "transaction_month",
        "is_night",
        "is_weekend",
        "customer_age",
        "city_pop",
        "gender_F",
        "gender_M",
        TARGET_COLUMN,
    ]
    category_columns = [
        column for column in dataframe.columns if column.startswith(CATEGORY_PREFIX)
    ]
    correlation_columns.extend(category_columns)

    correlation_data = dataframe[correlation_columns].astype(float)
    correlation_matrix = correlation_data.corr()

    figure, axis = create_figure(15, 12)
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5.5, "color": "#111315"},
        square=False,
        linewidths=0.25,
        linecolor="#2f3338",
        cbar_kws={"shrink": 0.78},
        ax=axis,
    )
    axis.set_title("Feature Correlation Heatmap", color="#f4f4f5", fontsize=15, pad=16)
    axis.tick_params(colors="#f4f4f5", labelsize=8)
    axis.set_facecolor("#111315")

    colorbar = axis.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color="#f4f4f5")
    plt.setp(colorbar.ax.get_yticklabels(), color="#f4f4f5")
    figure.tight_layout()
    save_current_figure("eda_correlation_heatmap.png", output_dir)


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.input)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {args.input}\n"
            f"[bold]Rows:[/bold] {len(dataframe):,}\n"
            f"[bold]Output:[/bold] {args.output_dir}",
            title="Exploratory Data Analysis",
            border_style="cyan",
        )
    )

    plot_target_distribution(dataframe, args.output_dir)
    plot_amount_distribution(dataframe, args.output_dir)
    plot_amount_by_fraud(dataframe, args.output_dir)
    export_amount_summary(dataframe, args.output_dir)
    plot_category_analysis(dataframe, args.output_dir)
    plot_distance_distribution(dataframe, args.output_dir)
    plot_distance_by_fraud(dataframe, args.output_dir)
    plot_distance_bucket_fraud_rate(dataframe, args.output_dir)
    plot_amount_vs_distance_sample(dataframe, args.output_dir)
    plot_hourly_fraud_rate(dataframe, args.output_dir)
    plot_is_night_fraud_rate(dataframe, args.output_dir)
    plot_correlation_heatmap(dataframe, args.output_dir)


if __name__ == "__main__":
    main()
