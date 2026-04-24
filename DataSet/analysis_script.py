import pandas as pd

df = pd.read_csv("fraud_transformed_reducted_scaled_train.csv", dtype={"cc_num": "str", "zip": "str"})

print(df.head(10))
print(df.info())
print(df["is_night"].head())