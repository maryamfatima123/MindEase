# model_loader.py — load fine-tuned BERT model from project root (not backend folder)

import os
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
import torch

# Load environment variables (.env)
load_dotenv()

def load_model_and_tokenizer():
    """
    Loads the fine-tuned BERT model and tokenizer from the project root directory.
    The model folder must contain: config.json, pytorch_model.bin, tokenizer files, etc.
    """
    # --- resolve project root ---
    project_root = Path(__file__).resolve().parent.parent  # backend folder's parent
    default_model_path = project_root / "mental_health_bert_final"

    # --- read from env (optional override) ---
    model_path_env = os.getenv("MODEL_PATH")
    model_dir = Path(model_path_env).resolve() if model_path_env else default_model_path

    print(f"🧠 Loading fine-tuned BERT model from: {model_dir}")

    if not model_dir.exists():
        raise FileNotFoundError(f"❌ Model directory not found: {model_dir}")

    # --- Load model + tokenizer ---
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    # --- enforce correct label mapping ---
    label2id = {"Depression": 0, "Anxiety": 1, "None": 2}
    id2label = {v: k for k, v in label2id.items()}
    model.config.label2id = label2id
    model.config.id2label = id2label
    model.config.num_labels = len(label2id)

    # --- use correct device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("✅ Model and tokenizer loaded successfully!")
    print(f"🔧 id2label: {model.config.id2label}")
    print(f"💻 Using device: {device}")
    return tokenizer, model
