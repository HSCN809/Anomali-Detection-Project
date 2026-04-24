import argparse
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
SYNTHETIC_OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "synthetic_prep_outputs"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop high-cardinality/raw columns.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Transformed dataset not found: {path}\n"
            "Create it first with: python PreProcessing/data_transformation.py"
        )

    return pd.read_csv(path, dtype=READ_DTYPES, low_memory=False)


def resolve_output_dir(path: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir
    return SYNTHETIC_OUTPUT_DIR if "synthetic" in path.stem.lower() else OUTPUT_DIR


def reduce_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    existing_columns_to_drop = [
        column for column in COLUMNS_TO_DROP if column in dataframe.columns
    ]
    return dataframe.drop(columns=existing_columns_to_drop)


def save_reduced_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
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
            "Could not write reduced dataset. Close any application that may "
            f"be using this file, then run the script again: {output_path}"
        ) from exc


def build_reduction_summary(
    original_dataframe: pd.DataFrame,
    reduced_dataframe: pd.DataFrame,
    output_path: Path,
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
                str(output_path),
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


def save_table_png(dataframe: pd.DataFrame, title: str, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

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
    output_path: Path,
    output_dir: Path,
) -> None:
    summary = build_reduction_summary(original_dataframe, reduced_dataframe, output_path)
    column_report = build_column_report(original_dataframe, reduced_dataframe)

    print_table(summary, "Data Reduction Summary")
    save_table_png(summary, "Data Reduction Summary", "reduction_summary.png", output_dir)

    print_table(column_report, "Data Reduction Column Report")
    save_table_png(
        column_report,
        "Data Reduction Column Report",
        "reduction_column_report.png",
        output_dir,
    )


def main() -> None:
    args = parse_args()
    original_dataframe = load_dataset(args.input)
    output_dir = resolve_output_dir(args.input, args.output_dir)
    reduced_dataframe = reduce_dataset(original_dataframe)
    save_reduced_dataset(reduced_dataframe, args.output)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {args.input}\n"
            f"[bold]Output:[/bold] {args.output}\n"
            f"[bold]Rows:[/bold] {len(reduced_dataframe):,}\n"
            f"[bold]Columns:[/bold] {len(reduced_dataframe.columns):,}",
            title="Data Reduction",
            border_style="cyan",
        )
    )
    export_reduction_analysis(original_dataframe, reduced_dataframe, args.output, output_dir)


if __name__ == "__main__":
    main()
