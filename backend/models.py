from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    # Relationship to scores
    scores = relationship("Score", back_populates="user")


class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    score_type = Column(String)  # PHQ-9 or GAD-7
    score_value = Column(Integer)
    timestamp = Column(DateTime)

    user = relationship("User", back_populates="scores")
