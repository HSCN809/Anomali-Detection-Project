import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "DataSet" / "synthetic_fraud_transformed_reducted_scaled_train.csv"

df = pd.read_csv(DATASET_PATH, dtype={"cc_num": "str", "zip": "str"})

print(df.head(10))
print(df.info())
print(df["is_night"].head())
