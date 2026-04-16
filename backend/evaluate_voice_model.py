import os
import json
from voice_model import analyze_voice
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# path to your test set (CSV with "filepath" and "label" columns)
test_df = pd.read_csv("voice_testset.csv")

true_labels, pred_labels = [], []

for i, row in test_df.iterrows():
    result = analyze_voice(row["filepath"])
    true_labels.append(row["label"])
    pred_labels.append(result["label"])

print("\n🎧 Voice Model Evaluation:")
print(classification_report(true_labels, pred_labels))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))
