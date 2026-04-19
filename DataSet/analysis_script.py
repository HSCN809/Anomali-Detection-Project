import pandas as pd

df = pd.read_csv('fraud_transformed.csv')

print(df.head(10))
print(df.info())