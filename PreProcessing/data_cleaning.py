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
DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_merged.csv"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "prep_outputs"
SYNTHETIC_OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "synthetic_prep_outputs"
NUMERIC_IMPUTE_COLUMNS = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
]
CATEGORICAL_IMPUTE_COLUMNS = [
    "merchant",
    "category",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "trans_date_trans_time",
    "cc_num",
    "trans_num",
    "source_dataset",
]
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and optionally clean missing values.")
    parser.add_argument("--input", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    return parser.parse_args()


def load_dataset(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {path}\n"
            "Create it first with: python DataSet/merge_fraud_datasets.py"
        )

    return pd.read_csv(path, low_memory=False, nrows=nrows)


def clean_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()

    for column in NUMERIC_IMPUTE_COLUMNS:
        if column in cleaned.columns:
            values = pd.to_numeric(cleaned[column], errors="coerce")
            median = values.median()
            cleaned[column] = values.fillna(0 if pd.isna(median) else median)

    for column in CATEGORICAL_IMPUTE_COLUMNS:
        if column not in cleaned.columns:
            continue

        values = cleaned[column].replace("", np.nan)
        mode = values.dropna().mode()
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        if column in {"merchant", "job", "zip"}:
            fill_value = "Unknown"
        cleaned[column] = values.fillna(fill_value)

    return cleaned


def save_cleaned_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    temp_output_path = output_path.with_suffix(".tmp.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(temp_output_path, index=False)
    temp_output_path.replace(output_path)


def resolve_output_dir(path: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir
    return SYNTHETIC_OUTPUT_DIR if "synthetic" in path.stem.lower() else OUTPUT_DIR


def build_missing_values_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_values = dataframe.isnull().sum().sort_values(ascending=False)
    total_rows = len(dataframe)

    return pd.DataFrame(
        {
            "Column": missing_values.index,
            "Missing Count": [f"{count:,}" for count in missing_values],
            "Missing Percent": [
                f"{(count / total_rows * 100) if total_rows else 0:.2f}%"
                for count in missing_values
            ],
        }
    )


def print_missing_values_table(summary: pd.DataFrame) -> None:
    table = Table(
        title="Missing Values by Column",
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("Column", style="cyan")
    table.add_column("Missing Count", justify="right")
    table.add_column("Missing Percent", justify="right")

    for row in summary.itertuples(index=False):
        table.add_row(row.Column, row[1], row[2])

    console.print(table)


def export_missing_values_png(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "missing_value_analysis.png"

    figure_height = max(4.5, len(summary) * 0.34 + 1.5)
    figure, axis = plt.subplots(figsize=(8.4, figure_height), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.axis("off")
    axis.set_title(
        "Missing Values by Column",
        color="#f4f4f5",
        fontsize=12,
        fontstyle="italic",
        pad=14,
    )

    image_table = axis.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.42, 0.28, 0.30],
    )
    image_table.auto_set_font_size(False)
    image_table.set_fontsize(9)
    image_table.scale(1, 1.35)

    for (row_index, column_index), cell in image_table.get_celld().items():
        cell.set_edgecolor("#d4d4d8")
        cell.set_linewidth(0.8)

        if row_index == 0:
            cell.set_facecolor("#17191c")
            cell.set_text_props(color="#ff66ff", weight="bold")
            cell.set_linewidth(1.5)
            continue

        cell.set_facecolor("#111315")
        if column_index == 0:
            cell.set_text_props(color="#00d7ff")
        else:
            cell.set_text_props(color="#f4f4f5", ha="right")

    figure.savefig(
        image_path,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    console.print(f"[green]PNG exported:[/green] {image_path}")


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.input, args.nrows)
    output_dir = resolve_output_dir(args.input, args.output_dir)
    missing_values_summary = build_missing_values_summary(dataframe)

    console.print(
        Panel.fit(
            f"[bold]Loaded dataset[/bold]\n"
            f"Path: [cyan]{args.input}[/cyan]\n"
            f"Rows: [green]{len(dataframe):,}[/green]\n"
            f"Columns: [green]{len(dataframe.columns):,}[/green]",
            title="Data Cleaning",
            border_style="cyan",
        )
    )

    print_missing_values_table(missing_values_summary)
    export_missing_values_png(missing_values_summary, output_dir)

    if args.output is not None:
        cleaned_dataframe = clean_missing_values(dataframe)
        save_cleaned_dataset(cleaned_dataframe, args.output)
        cleaned_missing_count = int(cleaned_dataframe.isnull().sum().sum())
        console.print(
            f"[green]Cleaned dataset exported:[/green] {args.output} "
            f"| remaining missing values: {cleaned_missing_count:,}"
        )


if __name__ == "__main__":
    main()
