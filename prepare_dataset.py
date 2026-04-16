import pandas as pd

# Load dataset
df = pd.read_csv("combineddata.csv")

print("✅ Original dataset loaded")
print("Columns:", df.columns)

# Rename for consistency
df = df.rename(columns={"statement": "text", "status": "label"})

# Normalize labels to lowercase
df["label"] = df["label"].str.lower()

print("\n🔎 Unique labels before mapping:", df["label"].unique())
print(df["label"].value_counts())

# Map labels (merge suicidal → depression, normal → none, ignore extras)
label_map = {
    "depression": "Depression",
    "suicidal": "Depression",
    "anxiety": "Anxiety",
    "normal": "None"
}

df["label"] = df["label"].map(label_map).astype(str)

# Drop rows with "nan" (not mapped)
df = df[df["label"] != "nan"]

print("\n✅ After mapping:")
print(df["label"].value_counts())

# Save cleaned dataset
# Save cleaned dataset (ensure "None" is preserved as text, not NaN)
df.to_csv("clean_reddit_dataset_v2.csv", index=False, na_rep="NA")

print("\n🎉 Saved new dataset as clean_reddit_dataset_v2.csv")
