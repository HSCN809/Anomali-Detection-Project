import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "DataSet" / "synthetic_fraud_transformed_reducted_scaled_train.csv"

dataframe = pd.read_csv(DATASET_PATH, dtype={"cc_num": "str", "zip": "str"})

print(dataframe.head(10))
print(dataframe.info())
print(dataframe["is_night"].head())
