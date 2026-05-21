import pandas as pd

dataset_path = "data/movie_reviews.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully")
print("---------------------------")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment label counts:")
print(df["sentiment"].value_counts())