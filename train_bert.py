import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

# =====================
# Config
# =====================
EPOCHS = 3  # 👈 change to 3 for full training
SUBSET_SIZE = None # 👈 None = full dataset, or set number (e.g. 300) for quick test

# =====================
# Load Dataset
# =====================
# Load dataset (preserve "None" labels)
df = pd.read_csv("clean_reddit_dataset_v2.csv", keep_default_na=False, na_values=[])
print("📊 Label distribution before split:\n", df["label"].value_counts())

# Encode labels
label2id = {"Depression": 0, "Anxiety": 1, "None": 2}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# =====================
# Balanced Subset Sampling
# =====================
if SUBSET_SIZE is not None:
    n_classes = len(label2id)
    per_class = SUBSET_SIZE // n_classes

    df_balanced = []
    for label_id in df["label_id"].unique():
        df_class = df[df["label_id"] == label_id]
        df_balanced.append(
            df_class.sample(
                n=per_class,
                random_state=42,
                replace=len(df_class) < per_class  # oversample if needed
            )
        )
    df = pd.concat(df_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"⚡ Using balanced subset of {len(df)} samples for quick training")

# =====================
# Train/Validation Split
# =====================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label_id"], test_size=0.1, stratify=df["label_id"], random_state=42
)

# =====================
# Dataset Class
# =====================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = MentalHealthDataset(train_texts, train_labels)
val_dataset = MentalHealthDataset(val_texts, val_labels)

# =====================
# Handle Class Imbalance
# =====================
class_counts = np.bincount(train_labels)
print("📊 Training class distribution:", dict(zip(id2label.values(), class_counts)))

class_weights = np.zeros(len(label2id))
for i in range(len(label2id)):
    class_weights[i] = 1.0 / (class_counts[i] if class_counts[i] > 0 else 1e-6)

sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =====================
# Model Setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = AdamW(model.parameters(), lr=2e-5)

# =====================
# Training Loop
# =====================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"🧠 Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} finished. Avg loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\n📊 Validation Report:\n", classification_report(true_labels, preds, target_names=list(id2label.values())))

    # Save checkpoint
    model.save_pretrained(f"./mental_health_bert_epoch{epoch+1}")
    tokenizer.save_pretrained(f"./mental_health_bert_epoch{epoch+1}")

# =====================
# Save Final Model
# =====================
model.save_pretrained("./mental_health_bert_final")
tokenizer.save_pretrained("./mental_health_bert_final")
print("🎉 Training complete. Model saved to ./mental_health_bert_final")
