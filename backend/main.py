# main.py — Refactored unified backend entrypoint with emotion tracking
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Response
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from jose import JWTError
from typing import Optional
import os
import tempfile
import traceback
from dotenv import load_dotenv
from datetime import datetime

# --- Load environment variables ---
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mental_health.db")
MODEL_PATH = os.getenv("MODEL_PATH", "mental_health_bert_final")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

# --- Local imports ---
import auth as auth_utils
from models import Base, User, Score
from multimodal_screening import analyze_multimodal, fuse_predictions, analyze_face
from emotion_model import predict_emotion_from_text, _ensure_loaded
from voice_model import analyze_voice

# --- Database setup ---
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- FastAPI app setup ---
app = FastAPI(title="AI Mental Health Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Dependency: DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Utility: Current user from token ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    username = auth_utils.decode_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Request Models ---
class SignupRequest(BaseModel):
    username: str
    password: str

# --- Auth Routes ---
@app.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = auth_utils.hash_password(req.password)
    new_user = User(username=req.username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    token = auth_utils.create_access_token({"sub": req.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = auth_utils.create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# --- DEV helper — Issue a token automatically for local dev (INSECURE: dev-only) ---
@app.get("/dev/issue_token")
def dev_issue_token(response: Response, db: Session = Depends(get_db)):
    """
    Dev-only: create/return an access token for a dev user so frontend can auto-login.
    Enable by setting DEV_MODE=true in your backend .env.
    """
    if os.getenv("DEV_MODE", "false").lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Not found")

    dev_username = os.getenv("DEV_USERNAME", "M")
    dev_password = os.getenv("DEV_PASSWORD", "m")

    user = db.query(User).filter(User.username == dev_username).first()
    if not user:
        # create dev user (dev-only)
        hashed = auth_utils.hash_password(dev_password)
        user = User(username=dev_username, hashed_password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)

    token = auth_utils.create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


# --- Get Scores (PHQ-9 / GAD-7) ---
@app.get("/scores")
def get_scores(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    scores = db.query(Score).filter(Score.user_id == current_user.id).all()
    # Return both 'value' and 'score_value' keys for frontend compatibility
    out = []
    for s in scores:
        item = {
            "id": s.id,
            "type": s.score_type,
            "value": s.score_value,
            "score_value": s.score_value,
            "timestamp": s.timestamp.isoformat(),
        }
        out.append(item)
    return out

# --- Emotion History Endpoint ---
@app.get("/emotion-history")
def get_emotion_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    scores = (
        db.query(Score)
        .filter(Score.user_id == current_user.id, Score.score_type == "emotion_result")
        .order_by(Score.timestamp.desc())
        .all()
    )
    return [
        {"timestamp": s.timestamp.isoformat(), "value": s.score_value}
        for s in scores
    ]

# --- Text-only Emotion Prediction ---
@app.post("/predict_text")
def predict_text(payload: dict, current_user: User = Depends(get_current_user)):
    text = payload.get("text", "") if isinstance(payload, dict) else ""
    try:
        res = predict_emotion_from_text(text)
        return {"text_emotion": res}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")

# --- 🧠 Multimodal Analysis (Text, Voice, Face) ---
@app.post("/analyze_multimodal")
async def analyze_multimodal_endpoint(
    text: Optional[str] = Form(None),
    voice_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
):
    print("🔍 AUTH HEADER:", current_user.username if current_user else "No User")

    text_res, voice_res, face_res = None, None, None

    try:
        # --- TEXT ---
        if text:
            text_res = predict_emotion_from_text(text)
        else:
            text_res = {"label": "No input received", "confidence": 0.0}

        # --- VOICE ---
        tmp_voice = None
        if voice_file:
            try:
                contents = await voice_file.read()
                tmp_voice = os.path.join(tempfile.gettempdir(), f"upload_voice_{os.getpid()}.wav")
                with open(tmp_voice, "wb") as f:
                    f.write(contents)
                print(f"🎤 Saved voice to: {tmp_voice}")
                voice_res = analyze_voice(tmp_voice)
            except Exception as e:
                print(f"⚠️ Voice analysis failed: {e}")
                voice_res = {"label": "Error", "confidence": 0.0, "note": str(e)}
            finally:
                if tmp_voice and os.path.exists(tmp_voice):
                    os.remove(tmp_voice)
        else:
            voice_res = {"label": "No input received", "confidence": 0.0}

        # --- FACE ---
        tmp_face = None
        if image_file:
            try:
                contents = await image_file.read()
                suffix = ".jpg"
                if image_file.filename.lower().endswith(".png"):
                    suffix = ".png"
                tmp_face = os.path.join(tempfile.gettempdir(), f"upload_face_{os.getpid()}{suffix}")
                with open(tmp_face, "wb") as f:
                    f.write(contents)
                print(f"📷 Saved uploaded face image to: {tmp_face}")
                face_res = analyze_face(tmp_face)
            except Exception as e:
                print(f"⚠️ Face analysis error: {e}")
                face_res = {"label": "Error", "confidence": 0.0, "note": str(e)}
            finally:
                if tmp_face and os.path.exists(tmp_face):
                    os.remove(tmp_face)
        else:
            face_res = {"label": "No input received", "confidence": 0.0}

        # --- COMBINE ---
        predictions = {"text": text_res, "voice": voice_res, "face": face_res}
        fusion = fuse_predictions(predictions)

        # --- SAVE TO DB ---
        try:
            final_label = fusion.get("final_label", "Unknown")
            emotion_map = {"Depression": 2, "Anxiety": 1, "None": 0, "Normal": 0, "Uncertain": 0}
            numeric_val = emotion_map.get(final_label, 0)
            db = SessionLocal()
            new_score = Score(
                user_id=current_user.id,
                score_type="emotion_result",
                score_value=numeric_val,
                timestamp=datetime.utcnow(),
            )
            db.add(new_score)
            db.commit()
            db.close()
        except Exception as save_err:
            print("⚠️ Could not save emotion result:", save_err)

        return {
            "text_emotion": text_res,
            "voice_emotion": voice_res,
            "face_emotion": face_res,
            "fusion": fusion,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")

# --- Model Check Endpoint ---
@app.get("/check_model")
def check_model():
    try:
        _ensure_loaded()
        from emotion_model import _model
        id2label = getattr(_model.config, "id2label", None)
        return {"model_loaded": True, "id2label": id2label}
    except Exception as e:
        return {"model_loaded": False, "error": str(e)}
