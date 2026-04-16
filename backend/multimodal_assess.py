# multimodal_assess.py
import os
import tempfile
from typing import Optional
from multimodal_screening import analyze_face, analyze_voice, fuse_predictions
from emotion_model import predict_emotion_from_text  # your existing text emotion analyzer

def analyze_multimodal(
    text_input: Optional[str] = None,
    voice_file: Optional[bytes] = None,
    image_file: Optional[bytes] = None
):
    """
    Runs text, voice, and face emotion analysis and fuses the predictions.
    Returns:
        {
            "final_label": "Depression"/"Anxiety"/"Normal"/"Uncertain",
            "final_confidence": 0.82,
            "reason": "...",
            "details": [...],
            "recommendation": "Suggested screening: PHQ-9"
        }
    """
    preds = []

    # 🧠 Text analysis
    if text_input:
        try:
            label = predict_emotion_from_text(text_input)
            preds.append({"label": label, "confidence": 0.85})
        except Exception as e:
            print("Text analysis failed:", e)

    # 🎤 Voice analysis
    if voice_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(voice_file)
                tmp_path = tmp.name
            preds.append(analyze_voice(tmp_path))
            os.remove(tmp_path)
        except Exception as e:
            print("Voice analysis failed:", e)

    # 📸 Face analysis
    if image_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_file)
                tmp_path = tmp.name
            preds.append(analyze_face(tmp_path))
            os.remove(tmp_path)
        except Exception as e:
            print("Face analysis failed:", e)

    # 🧩 Fuse all predictions
    fusion = fuse_predictions(preds)

    # 🩺 Suggest screening based on fused result
    label = fusion.get("final_label", "Uncertain")
    if label == "Depression":
        recommendation = "Based on your inputs, PHQ-9 screening is recommended."
    elif label == "Anxiety":
        recommendation = "Based on your inputs, GAD-7 screening is recommended."
    elif label == "Normal":
        recommendation = "Your current emotional state appears stable. Keep monitoring regularly."
    else:
        recommendation = "Unable to confidently determine mood. Consider a check-in with both PHQ-9 and GAD-7."

    fusion["recommendation"] = recommendation
    return fusion
