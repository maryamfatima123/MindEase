# schemas.py
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class ChatMessage(BaseModel):
    message: str

class ScreeningResponse(BaseModel):
    screening_type: str
    questions: list[str]

class ScoreSubmission(BaseModel):
    screening_type: str
    score: int
