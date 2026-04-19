from pathlib import Path

import matplotlib
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_transformed.csv"
OUTPUT_DATASET_PATH = ROOT_DIR / "DataSet" / "fraud_transformed_reducted.csv"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "prep_outputs"

COLUMNS_TO_DROP = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "unix_time",
    "cc_num",
    "merchant",
    "first",
    "last",
    "street",
    "city",
    "state",
    "lat",
    "long",
    "dob",
    "trans_num",
    "merch_lat",
    "merch_long",
    "source_dataset",
]

READ_DTYPES = {
    "cc_num": "string",
    "zip": "string",
}

console = Console()


def load_dataset() -> pd.DataFrame:
    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Transformed dataset not found: {INPUT_DATASET_PATH}\n"
            "Create it first with: python PreProcessing/data_transformation.py"
        )

    return pd.read_csv(INPUT_DATASET_PATH, dtype=READ_DTYPES, low_memory=False)


def reduce_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    existing_columns_to_drop = [
        column for column in COLUMNS_TO_DROP if column in dataframe.columns
    ]
    return dataframe.drop(columns=existing_columns_to_drop)


def save_reduced_dataset(dataframe: pd.DataFrame) -> None:
    temp_output_path = OUTPUT_DATASET_PATH.with_suffix(".tmp.csv")

    try:
        dataframe.to_csv(temp_output_path, index=False)
        temp_output_path.replace(OUTPUT_DATASET_PATH)
    except PermissionError as exc:
        try:
            temp_output_path.unlink(missing_ok=True)
        except PermissionError:
            pass

        raise SystemExit(
            "Could not write reduced dataset. Close any application that may "
            f"be using this file, then run the script again: {OUTPUT_DATASET_PATH}"
        ) from exc


def build_reduction_summary(
    original_dataframe: pd.DataFrame,
    reduced_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dropped_columns = [
        column for column in COLUMNS_TO_DROP if column in original_dataframe.columns
    ]
    missing_drop_columns = [
        column for column in COLUMNS_TO_DROP if column not in original_dataframe.columns
    ]

    return pd.DataFrame(
        {
            "metric": [
                "input_rows",
                "input_columns",
                "output_rows",
                "output_columns",
                "dropped_columns",
                "missing_requested_columns",
                "output_file",
            ],
            "value": [
                f"{len(original_dataframe):,}",
                f"{len(original_dataframe.columns):,}",
                f"{len(reduced_dataframe):,}",
                f"{len(reduced_dataframe.columns):,}",
                f"{len(dropped_columns):,}",
                f"{len(missing_drop_columns):,}",
                "DataSet/fraud_transformed_reducted.csv",
            ],
        }
    )


def build_column_report(
    original_dataframe: pd.DataFrame,
    reduced_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for column in COLUMNS_TO_DROP:
        rows.append(
            {
                "column": column,
                "status": "dropped" if column in original_dataframe.columns else "not_found",
            }
        )

    for column in reduced_dataframe.columns:
        rows.append({"column": column, "status": "kept"})

    return pd.DataFrame(rows)


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

    figure_height = max(3.5, len(dataframe) * 0.35 + 1.4)
    figure_width = max(8, len(dataframe.columns) * 2.6)
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


def export_reduction_analysis(
    original_dataframe: pd.DataFrame,
    reduced_dataframe: pd.DataFrame,
) -> None:
    summary = build_reduction_summary(original_dataframe, reduced_dataframe)
    column_report = build_column_report(original_dataframe, reduced_dataframe)

    print_table(summary, "Data Reduction Summary")
    save_table_png(summary, "Data Reduction Summary", "reduction_summary.png")

    print_table(column_report, "Data Reduction Column Report")
    save_table_png(
        column_report,
        "Data Reduction Column Report",
        "reduction_column_report.png",
    )


def main() -> None:
    original_dataframe = load_dataset()
    reduced_dataframe = reduce_dataset(original_dataframe)
    save_reduced_dataset(reduced_dataframe)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {INPUT_DATASET_PATH}\n"
            f"[bold]Output:[/bold] {OUTPUT_DATASET_PATH}\n"
            f"[bold]Rows:[/bold] {len(reduced_dataframe):,}\n"
            f"[bold]Columns:[/bold] {len(reduced_dataframe.columns):,}",
            title="Data Reduction",
            border_style="cyan",
        )
    )
    export_reduction_analysis(original_dataframe, reduced_dataframe)


if __name__ == "__main__":
    main()
