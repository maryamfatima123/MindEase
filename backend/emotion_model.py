# emotion_model.py
"""
Text emotion classifier wrapper.
Uses the shared loader in model_loader.py so the same trained model is used everywhere.
"""

from model_loader import load_model_and_tokenizer
import torch
import numpy as np

# Load once (lazy) when first called
_tokenizer = None
_model = None

MIN_CONFIDENCE = 0.40  # tune this as desired (0.40-0.50 recommended)

def _ensure_loaded():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer, _model = load_model_and_tokenizer()

def predict_emotion_from_text(text: str):
    """
    Return a dict: {label, confidence, probs, pred_id}
    - label: 'Depression'/'Anxiety'/'None'
    - confidence: max probability
    - probs: list of probabilities in order [Depression, Anxiety, None]
    """
    _ensure_loaded()
    if text is None:
        return {"label": "None", "confidence": 0.0, "probs": [0.0, 0.0, 0.0], "pred_id": None}

    text = text.strip().lower()
    if text == "":
        return {"label": "None", "confidence": 0.0, "probs": [0.0, 0.0, 0.0], "pred_id": None}

    # 🩸 Detect suicidal / self-harm cues first
    suicidal_keywords = [
        "kill myself", "end my life", "suicide", "die", "better off dead",
        "can't go on", "no reason to live", "want to die", "take my life"
    ]
    if any(phrase in text for phrase in suicidal_keywords):
        return {"label": "Depression", "confidence": 1.0, "probs": [1.0, 0.0, 0.0], "pred_id": 0}

    # 🧠 Model prediction
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        confidence = float(probs.max())
        id2label = _model.config.id2label
        label = id2label.get(pred_id, "None")

    # 🔹 Always return the model label (no “uncertain” state)
    return {"label": label, "confidence": confidence, "probs": probs.tolist(), "pred_id": pred_id}
