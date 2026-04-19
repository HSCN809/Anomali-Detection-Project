from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pandas is required for this script. Install dependencies with: "
        "pip install -r requirements.txt"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "DataSet"
DATASET_FILES = ("fraudTrain.csv", "fraudTest.csv")
TARGET_COLUMN = "is_fraud"


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_dataframe(name: str, df: pd.DataFrame) -> None:
    print_section(f"{name} - General Information")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumns:")
    print(df.columns.to_list())

    print("\nData types:")
    print(df.dtypes.value_counts())

    print_section(f"{name} - head()")
    print(df.head())

    print_section(f"{name} - describe()")
    print(df.describe(include="all").transpose())

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
    print(missing_values)

    print_section(f"{name} - Duplicate Rows")
    print(f"Duplicate row count: {df.duplicated().sum():,}")

    if TARGET_COLUMN in df.columns:
        print_section(f"{name} - Target Distribution ({TARGET_COLUMN})")
        target_summary = pd.DataFrame(
            {
                "count": df[TARGET_COLUMN].value_counts(dropna=False),
                "percent": df[TARGET_COLUMN].value_counts(dropna=False, normalize=True)
                * 100,
            }
        )
        print(target_summary)

    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        print_section(f"{name} - Numeric Columns Summary")
        print(df[numeric_columns].describe().transpose())

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) > 0:
        print_section(f"{name} - Categorical Columns Cardinality")
        cardinality = df[categorical_columns].nunique().sort_values(ascending=False)
        print(cardinality)

        print_section(f"{name} - Top Values for Categorical Columns")
        for column in categorical_columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).head(10))

    if TARGET_COLUMN in df.columns and TARGET_COLUMN in numeric_columns:
        print_section(f"{name} - Numeric Correlation With {TARGET_COLUMN}")
        correlations = (
            df[numeric_columns]
            .corr(numeric_only=True)[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
        print(correlations)


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
