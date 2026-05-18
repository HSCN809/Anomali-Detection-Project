from pathlib import Path
import re

try:
    import matplotlib
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"{exc.name} is required for this script. Install dependencies with: "
        "pip install -r requirements.txt"
    ) from exc

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ModuleNotFoundError:
    Console = None
    Panel = None
    Table = None


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "DataSet"
OUTPUT_DIR = ROOT_DIR / "PreProcessing" / "synthetic_prep_outputs"
DATASET_FILES = ("synthetic_fraud_merged.csv",)
TARGET_COLUMN = "is_fraud"
console = Console() if Console is not None else None
exported_image_paths: set[Path] = set()


def print_section(title: str) -> None:
    if console is not None:
        console.print(Panel.fit(title, style="bold cyan"))
        return

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_value(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, float):
        return f"{value:,.2f}"

    return str(value)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "table"


def normalize_table_data(data, *, index: bool = True) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        normalized = data.reset_index()
        normalized.columns = ["column", "value"]
        return normalized

    if isinstance(data, pd.DataFrame):
        normalized = data.copy()
        if index:
            index_name = normalized.index.name or "index"
            normalized.insert(0, index_name, normalized.index)
        normalized = normalized.reset_index(drop=True)
        return normalized

    return pd.DataFrame(data)


def export_table_png(data, *, title: str, file_stem: str, index: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{slugify(file_stem)}.png"
    if output_path in exported_image_paths:
        return

    exported_image_paths.add(output_path)
    table_data = normalize_table_data(data, index=index)
    table_data = table_data.apply(lambda column: column.map(format_value))

    row_count = len(table_data)
    column_count = len(table_data.columns)
    max_text_length = max(
        [len(str(column)) for column in table_data.columns]
        + [len(value) for value in table_data.to_numpy().flatten()],
        default=8,
    )

    figure_width = min(28, max(8, column_count * 1.45, max_text_length * 0.18))
    figure_height = min(36, max(4.5, row_count * 0.36 + 1.6))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=150)
    figure.patch.set_facecolor("#111315")
    axis.set_facecolor("#111315")
    axis.axis("off")
    axis.set_title(
        title,
        color="#f4f4f5",
        fontsize=12,
        fontstyle="italic",
        pad=14,
    )

    table = axis.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.28)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d4d8")
        cell.set_linewidth(0.55)

        is_header = row_index == 0
        cell.set_facecolor("#17191c" if is_header else "#111315")
        if is_header:
            cell.set_text_props(color="#ff66ff", weight="bold")
            cell.set_linewidth(1.2)
        else:
            text_color = "#00d7ff" if column_index == 0 else "#f4f4f5"
            cell.set_text_props(color=text_color)

    figure.savefig(output_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    if console is not None:
        console.print(f"[green]PNG exported:[/green] {output_path}")
    else:
        print(f"PNG exported: {output_path}")


def print_plain_table(data, *, index: bool = True) -> None:
    print(data.to_string(index=index) if isinstance(data, pd.DataFrame) else data)


def print_rich_table(data: pd.DataFrame, *, index: bool = True) -> None:
    table = Table(show_header=True, header_style="bold magenta", show_lines=False)

    if index:
        table.add_column("index", style="dim")

    for column in data.columns:
        table.add_column(str(column))

    for row_index, row in data.iterrows():
        values = [format_value(row_index)] if index else []
        values.extend(format_value(value) for value in row)
        table.add_row(*values)

    console.print(table)


def print_table(data, *, index: bool = True) -> None:
    if isinstance(data, pd.Series):
        data = data.reset_index()
        data.columns = ["column", "value"]
        index = False

    if console is None:
        print_plain_table(data, index=index)
        return

    if not isinstance(data, pd.DataFrame):
        console.print(data)
        return

    print_rich_table(data, index=index)


def print_and_export_table(
    data,
    *,
    title: str,
    file_stem: str,
    index: bool = True,
) -> None:
    print_table(data, index=index)
    export_table_png(data, title=title, file_stem=file_stem, index=index)


def analyze_dataframe(name: str, df: pd.DataFrame) -> None:
    dataset_name = Path(name).stem
    dataset_slug = slugify(dataset_name)

    print_section(f"{name} - General Information")
    general_info = pd.DataFrame(
        {
            "metric": ["rows", "columns", "memory_usage_mb"],
            "value": [
                f"{df.shape[0]:,}",
                f"{df.shape[1]:,}",
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
            ],
        }
    )
    print_and_export_table(
        general_info,
        title=f"{dataset_name} - General Information",
        file_stem=f"{dataset_slug}_general_information",
        index=False,
    )

    print_section(f"{name} - Columns")
    columns = pd.DataFrame({"column": df.columns.to_list()})
    print_and_export_table(
        columns,
        title=f"{dataset_name} - Columns",
        file_stem=f"{dataset_slug}_columns",
        index=False,
    )

    print("\nData types:")
    print_and_export_table(
        df.dtypes.value_counts(),
        title=f"{dataset_name} - Data Types",
        file_stem=f"{dataset_slug}_data_types",
    )

    print_section(f"{name} - head()")
    print_and_export_table(
        df.head(),
        title=f"{dataset_name} - Head",
        file_stem=f"{dataset_slug}_head",
        index=False,
    )

    print_section(f"{name} - describe()")
    print_and_export_table(
        df.describe(include="all").transpose(),
        title=f"{dataset_name} - Describe",
        file_stem=f"{dataset_slug}_describe",
    )

    print_section(f"{name} - Missing Values")
    missing_values = (
        pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_percent": df.isna().mean() * 100,
            }
        )
        .sort_values("missing_count", ascending=False)
    )
    print_and_export_table(
        missing_values,
        title=f"{dataset_name} - Missing Values",
        file_stem=f"{dataset_slug}_missing_values",
    )

    print_section(f"{name} - Duplicate Rows")
    duplicate_rows = pd.DataFrame(
        {"metric": ["duplicate_row_count"], "value": [f"{df.duplicated().sum():,}"]}
    )
    print_and_export_table(
        duplicate_rows,
        title=f"{dataset_name} - Duplicate Rows",
        file_stem=f"{dataset_slug}_duplicate_rows",
        index=False,
    )

    if TARGET_COLUMN in df.columns:
        print_section(f"{name} - Target Distribution ({TARGET_COLUMN})")
        target_summary = pd.DataFrame(
            {
                "count": df[TARGET_COLUMN].value_counts(dropna=False),
                "percent": df[TARGET_COLUMN].value_counts(dropna=False, normalize=True)
                * 100,
            }
        )
        print_and_export_table(
            target_summary,
            title=f"{dataset_name} - Target Distribution",
            file_stem=f"{dataset_slug}_target_distribution_{TARGET_COLUMN}",
        )

    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        print_section(f"{name} - Numeric Columns Summary")
        print_and_export_table(
            df[numeric_columns].describe().transpose(),
            title=f"{dataset_name} - Numeric Columns Summary",
            file_stem=f"{dataset_slug}_numeric_columns_summary",
        )

    categorical_columns = df.select_dtypes(include=["object", "category", "str"]).columns
    if len(categorical_columns) > 0:
        print_section(f"{name} - Categorical Columns Cardinality")
        cardinality = df[categorical_columns].nunique().sort_values(ascending=False)
        print_and_export_table(
            cardinality,
            title=f"{dataset_name} - Categorical Columns Cardinality",
            file_stem=f"{dataset_slug}_categorical_columns_cardinality",
        )

        print_section(f"{name} - Top Values for Categorical Columns")
        for column in categorical_columns:
            print(f"\n{column}:")
            print_and_export_table(
                df[column].value_counts(dropna=False).head(10),
                title=f"{dataset_name} - Top Values - {column}",
                file_stem=f"{dataset_slug}_top_values_{slugify(column)}",
            )

    if TARGET_COLUMN in df.columns and TARGET_COLUMN in numeric_columns:
        print_section(f"{name} - Numeric Correlation With {TARGET_COLUMN}")
        correlations = (
            df[numeric_columns]
            .corr(numeric_only=True)[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
        print_and_export_table(
            correlations,
            title=f"{dataset_name} - Numeric Correlation With {TARGET_COLUMN}",
            file_stem=f"{dataset_slug}_numeric_correlation_with_{TARGET_COLUMN}",
        )


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    for filename in DATASET_FILES:
        file_path = DATASET_DIR / filename

        if not file_path.exists():
            print_section(f"Missing File: {filename}")
            print(f"Expected path: {file_path}")
            continue

        print_section(f"Loading {filename}")
        dataframe = pd.read_csv(file_path, low_memory=False)
        analyze_dataframe(filename, dataframe)


if __name__ == "__main__":
    main()
