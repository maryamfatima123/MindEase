# clean_data.py

import pandas as pd

# Load with tab separator, fix header split
df = pd.read_csv("data/Combined Data.csv", sep="\t", engine="python", on_bad_lines="skip", names=["index", "text", "label"], header=None)

# Drop empty rows
df = df.dropna(subset=["text", "label"])

# Normalize label names
label_map = {
    "Anxiety": 2,
    "Anxious": 2,
    "Depression": 1,
    "Depressed": 1,
    "None": 0,
    "Suicidal": 3
}

df['label'] = df['label'].map(label_map)
df = df[df['label'].notna()]

# Save final output
df[['text', 'label']].to_csv("clean_reddit_dataset.csv", index=False)

# Summary
print("📊 Label distribution:")
print(df['label'].value_counts())
print("✅ Cleaned dataset saved: clean_reddit_dataset.csv")
