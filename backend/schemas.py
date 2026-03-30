from pydantic import BaseModel
from typing import Optional, Literal


class UploadResponse(BaseModel):
    session_id: str
    message: str
    chunks_created: int


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class ShortLongConfig(BaseModel):
    count: int
    marks_each: int


class GenerateRequest(BaseModel):
    session_id: str
    type: Literal["mcq", "descriptive"]
    topic_focus: Optional[str] = ""
    total_marks: int
    duration_minutes: int
    difficulty: Literal["easy", "medium", "hard"]
    # MCQ specific
    num_questions: Optional[int] = None
    marks_per_question: Optional[int] = None
    negative_marking: Optional[bool] = False
    # Descriptive specific
    short_questions: Optional[ShortLongConfig] = None
    long_questions: Optional[ShortLongConfig] = None


class GenerateResponse(BaseModel):
    paper_type: str
    questions: list
    metadata: dict
