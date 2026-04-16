import os
import numpy as np
from typing import Dict, Optional

from voice_model import analyze_voice
from emotion_model import predict_emotion_from_text as analyze_text  # BERT-based text model

# Face analysis (DeepFace)
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None


# --- Confidence thresholds ---
MULTI_CONF_THRESH = 0.25
SINGLE_CONF_THRESH = 0.15


# --- Emotion mapping helper ---
def map_emotion_to_label(raw_emotion: str) -> str:
    e = (raw_emotion or "").lower()
    if any(x in e for x in ["sad", "cry", "bored", "depress", "tired", "down", "upset"]):
        return "Depression"
    if any(x in e for x in ["fear", "angry", "panic", "stress", "tense", "anxious", "worry"]):
        return "Anxiety"
    return "None"


# --- Face Analysis ---
from PIL import Image
import os

from PIL import Image
import os

def analyze_face(image_path: str) -> Dict[str, float]:
    """
    Analyze emotion from a face image using DeepFace, with macOS screenshot & PNG support.
    Includes detailed debug logging.
    """
    if not image_path or not os.path.exists(image_path):
        print("⚠️ No image file found at given path.")
        return {"label": "No input received", "confidence": 0.0}

    if DeepFace is None:
        print("⚠️ DeepFace not installed.")
        return {"label": "No input received", "confidence": 0.0, "note": "DeepFace not installed"}

    try:
        # 🧩 Step 1 — Convert image to JPG (DeepFace is bad with PNG/HEIC)
        converted_path = image_path
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # remove alpha channel
                converted_path = os.path.splitext(image_path)[0] + "_converted.jpg"
                img.save(converted_path, "JPEG")
                print(f"🖼️ Converted image saved as: {converted_path}")
        except Exception as conv_err:
            print(f"⚠️ Image conversion failed: {conv_err}. Using original path.")

        # 🧩 Step 2 — Analyze emotion
        print("🔍 Running DeepFace analysis...")
        res = DeepFace.analyze(
            img_path=converted_path,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False
        )

        print(f"🧩 Raw DeepFace output: {res}")

        # DeepFace sometimes returns a list
        if isinstance(res, list) and res:
            res = res[0]

        dominant_emotion = (res.get("dominant_emotion") or "").lower()
        emotion_scores = res.get("emotion", {}) or {}
        total = sum(emotion_scores.values()) or 1.0
        confidence = float(emotion_scores.get(dominant_emotion, 0.0)) / total

        mapped_label = map_emotion_to_label(dominant_emotion)
        print(f"🧍‍♀️ Face emotion: {dominant_emotion} → {mapped_label} ({confidence:.3f})")

        # Clean up temp conversion
        if converted_path != image_path and os.path.exists(converted_path):
            os.remove(converted_path)

        return {"label": mapped_label, "confidence": round(confidence, 3)}

    except Exception as e:
        print(f"⚠️ Face analysis failed: {e}")
        return {"label": "No input received", "confidence": 0.0, "note": str(e)}


# --- Fusion Logic ---
def fuse_predictions(predictions: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Fuse predictions from available modalities (text, voice, face).
    Returns final combined prediction with confidence and reason.
    """
    valid_preds = {
        k: v for k, v in predictions.items()
        if v and v.get("label") not in ["No input received", "None", "Normal"]
    }

    if not valid_preds:
        return {
            "final_label": "No input received",
            "final_confidence": 0.0,
            "reason": "no_valid_modalities"
        }

    # Group by label
    label_confidences = {}
    for source, pred in valid_preds.items():
        label = pred["label"]
        conf = pred["confidence"]
        label_confidences.setdefault(label, []).append((source, conf))

    best_label, best_conf, best_sources = None, 0.0, []

    for label, confs in label_confidences.items():
        avg_conf = np.mean([c for _, c in confs])
        if avg_conf > best_conf:
            best_label = label
            best_conf = avg_conf
            best_sources = [s for s, _ in confs]

    # If all confidences too low
    if best_conf < MULTI_CONF_THRESH:
        return {
            "final_label": "No significant signs detected",
            "final_confidence": round(best_conf, 3),
            "reason": f"low_confidence ({best_conf:.2f})"
        }

    return {
        "final_label": best_label,
        "final_confidence": round(best_conf, 3),
        "reason": f"{', '.join(best_sources)}_dominant"
    }


# --- Multimodal Analysis (Text + Voice + Face) ---
def analyze_multimodal(
    text: Optional[str] = None,
    voice_path: Optional[str] = None,
    face_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run multimodal analysis (text, voice, face) and return structured results.
    """
    # ---- TEXT ----
    if text and text.strip():
        try:
            text_result = analyze_text(text)
            if isinstance(text_result, str):
                text_result = {"label": text_result, "confidence": 0.7}
        except Exception as e:
            text_result = {"label": "Error", "confidence": 0.0, "note": f"Text error: {e}"}
    else:
        text_result = {"label": "No input received", "confidence": 0.0}

    # ---- VOICE ----
    if voice_path and os.path.exists(voice_path):
        try:
            voice_result = analyze_voice(voice_path)
        except Exception as e:
            voice_result = {"label": "Error", "confidence": 0.0, "note": f"Voice error: {e}"}
    else:
        voice_result = {"label": "No input received", "confidence": 0.0}

    # ---- FACE ----
    if face_path and os.path.exists(face_path):
        try:
            face_result = analyze_face(face_path)
        except Exception as e:
            face_result = {"label": "Error", "confidence": 0.0, "note": f"Face error: {e}"}
    else:
        face_result = {"label": "No input received", "confidence": 0.0}

    # ---- Combine all ----
    all_results = {
        "text": text_result,
        "voice": voice_result,
        "face": face_result,
    }

    combined = fuse_predictions(all_results) or {
        "final_label": "Unknown",
        "final_confidence": 0.0,
        "reason": "no_fusion"
    }

    return {
        "text_result": text_result,
        "voice_result": voice_result,
        "face_result": face_result,
        "final_result": combined
    }
