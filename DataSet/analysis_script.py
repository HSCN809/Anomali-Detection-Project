import pandas as pd

df = pd.read_csv("fraud_transformed.csv", dtype={"cc_num": "str", "zip": "str"})

print(df.head(10))
print(df.info())
print(df["is_night"].head())