import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Load Model and Tokenizer ---
MODEL_PATH = "./mental_health_bert_final"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Load Dataset ---
df = pd.read_csv("clean_reddit_dataset_v2.csv", keep_default_na=False, na_values=[])
label2id = {"Depression": 0, "Anxiety": 1, "None": 2}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# --- Validation Split (same as training setup) ---
_, val_texts, _, val_labels = train_test_split(
    df["text"], df["label_id"], test_size=0.1, stratify=df["label_id"], random_state=42
)

# --- Optional: limit for quick testing ---
# val_texts = val_texts.sample(300, random_state=42)
# val_labels = val_labels.loc[val_texts.index]

# --- Predict in Batches ---
batch_size = 16
preds, true_labels = [], []

for i in tqdm(range(0, len(val_texts), batch_size), desc="Evaluating"):
    batch_texts = val_texts[i:i + batch_size].tolist()
    batch_labels = val_labels[i:i + batch_size].tolist()

    encodings = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    preds.extend(batch_preds)
    true_labels.extend(batch_labels)

# --- Evaluation ---
print("\n📊 FINAL MODEL EVALUATION REPORT:\n")
print(classification_report(true_labels, preds, target_names=list(id2label.values())))

print("\nConfusion Matrix:\n")
print(confusion_matrix(true_labels, preds))
