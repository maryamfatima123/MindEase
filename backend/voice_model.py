# voice_model.py
import os
import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import os
from pydub import AudioSegment

# 🧩 Use the imageio-provided FFmpeg binary
os.environ["FFMPEG_BINARY"] = "/Users/macair/easenv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-x86_64-v7.1"
AudioSegment.converter = os.environ["FFMPEG_BINARY"]

# ✅ Ensure FFmpeg works even on macOS (via imageio-ffmpeg)
try:
    import imageio_ffmpeg
    from pydub import AudioSegment

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]
    AudioSegment.converter = ffmpeg_path
    print(f"🎬 Using FFmpeg binary at: {ffmpeg_path}")
except Exception as e:
    print(f"⚠️ Could not set FFmpeg path automatically: {e}")

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

print("🎤 Loading fine-tuned voice emotion recognition model...")

# Use feature extractor instead of processor (this model has no tokenizer)
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Map raw model emotions to Depression/Anxiety/None
def map_voice_emotion_to_label(raw_emotion: str) -> str:
    e = (raw_emotion or "").lower()
    if any(x in e for x in ["sad", "neutral", "tired", "calm", "bored", "cry", "disgust"]):
        return "Depression"
    if any(x in e for x in ["angry", "fear", "panic", "stress", "tense", "anx"]):
        return "Anxiety"
    return "None"

def analyze_voice(file_path: str):
    """
    Analyze an audio file and return a dict:
    { 'label': 'Depression' | 'Anxiety' | 'None', 'confidence': float }
    Supports WAV and MP3 files.
    """
    try:
        # Try to load with torchaudio — supports WAV/MP3/FLAC
        speech, sr = torchaudio.load(file_path)
    except Exception as e:
        print("⚠️ torchaudio load failed:", e)
        try:
            # Fallback to pydub for mp3 or m4a files
            audio = AudioSegment.from_file(file_path)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            speech = torch.tensor(samples) / np.iinfo(np.int16).max
            sr = audio.frame_rate
        except Exception as e2:
            print("❌ Voice decoding failed completely:", e2)
            return {"label": "None", "confidence": 0.0, "note": f"decode_error: {e2}"}

    try:
        # Convert to mono if needed
        if speech.ndim > 1:
            speech = speech.mean(dim=0)
        speech = speech / (speech.abs().max() + 1e-9)  # normalize amplitude

        # Extract features
        inputs = extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence, pred_id = torch.max(probs, dim=-1)

        label = model.config.id2label[pred_id.item()]
        mapped_label = map_voice_emotion_to_label(label)
        conf_val = round(confidence.item(), 3)

        print(f"🎧 Raw voice emotion: {label}, mapped: {mapped_label}, confidence: {conf_val}")
        return {"label": mapped_label, "confidence": conf_val}

    except Exception as e:
        print("⚠️ Voice analysis failed:", e)
        return {"label": "None", "confidence": 0.0, "note": f"error: {e}"}
