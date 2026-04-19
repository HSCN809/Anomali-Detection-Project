from pathlib import Path

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
TRAIN_FILE = DATASET_DIR / "fraudTrain.csv"
TEST_FILE = DATASET_DIR / "fraudTest.csv"
OUTPUT_FILE = DATASET_DIR / "fraud_merged.csv"


def csv_compression(file_path: Path) -> str:
    with file_path.open("rb") as file:
        return "zip" if file.read(4) == b"PK\x03\x04" else "infer"


def read_fraud_csv(file_path: Path, source_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path, compression=csv_compression(file_path), low_memory=False)
    dataframe["source_dataset"] = source_name
    return dataframe


def main() -> None:
    train_dataframe = read_fraud_csv(TRAIN_FILE, "train")
    test_dataframe = read_fraud_csv(TEST_FILE, "test")
    merged_dataframe = pd.concat([train_dataframe, test_dataframe], ignore_index=True)

    merged_dataframe.to_csv(OUTPUT_FILE, index=False)

    print(f"Merged rows: {len(merged_dataframe):,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
