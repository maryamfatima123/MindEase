import pandas as pd

df = pd.read_csv("clean_reddit_dataset.csv")

# Show first few rows
print(df.head())

# Show column names
print("Columns:", df.columns.tolist())

# Show label distribution
print(df["label"].value_counts())
