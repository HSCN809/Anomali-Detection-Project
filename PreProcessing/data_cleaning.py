from pathlib import Path

import matplotlib
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_merged.csv"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "prep_outputs"
MISSING_VALUES_IMAGE_PATH = OUTPUT_DIR / "missing_value_analysis.png"
console = Console()


def load_dataset(nrows: int | None = None) -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {DATASET_PATH}\n"
            "Create it first with: python DataSet/merge_fraud_datasets.py"
        )

    return pd.read_csv(DATASET_PATH, low_memory=False, nrows=nrows)


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


def export_missing_values_png(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        MISSING_VALUES_IMAGE_PATH,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    console.print(f"[green]PNG exported:[/green] {MISSING_VALUES_IMAGE_PATH}")


def main() -> None:
    dataframe = load_dataset()
    missing_values_summary = build_missing_values_summary(dataframe)

    console.print(
        Panel.fit(
            f"[bold]Loaded dataset[/bold]\n"
            f"Path: [cyan]{DATASET_PATH}[/cyan]\n"
            f"Rows: [green]{len(dataframe):,}[/green]\n"
            f"Columns: [green]{len(dataframe.columns):,}[/green]",
            title="Data Cleaning",
            border_style="cyan",
        )
    )

    print_missing_values_table(missing_values_summary)
    export_missing_values_png(missing_values_summary)


if __name__ == "__main__":
    main()
