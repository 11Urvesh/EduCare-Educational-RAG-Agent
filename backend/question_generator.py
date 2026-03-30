import json
import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_BASE = Path("vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MCQ_PROMPT = """You are EduCare, an expert exam paper setter.
Using ONLY the provided study material context below, generate a multiple choice question paper.

Exam Specifications:
- Topic focus: {topic_focus}
- Total questions: {num_questions}
- Marks per question: {marks_per_question}
- Total marks: {total_marks}
- Duration: {duration_minutes} minutes
- Difficulty level: {difficulty}
- Negative marking: {negative_marking}

Context from study material:
{context}

IMPORTANT: Return ONLY a valid JSON array with no explanation and no markdown fences. Format:
[
  {{
    "q_no": 1,
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "Brief explanation why this is correct"
  }}
]"""

DESCRIPTIVE_PROMPT = """You are EduCare, an expert exam paper setter.
Using ONLY the provided study material context below, generate a descriptive question paper.

Exam Specifications:
- Topic focus: {topic_focus}
- Short questions: {short_q_count} questions × {short_q_marks} marks each
- Long questions: {long_q_count} questions × {long_q_marks} marks each
- Total marks: {total_marks}
- Duration: {duration_minutes} minutes
- Difficulty level: {difficulty}

Context from study material:
{context}

IMPORTANT: Return ONLY a valid JSON array with no explanation and no markdown fences. Format:
[
  {{
    "q_no": 1,
    "type": "short",
    "marks": {short_q_marks},
    "question": "...",
    "answer_hint": "Key points expected in the answer"
  }},
  {{
    "q_no": 6,
    "type": "long",
    "marks": {long_q_marks},
    "question": "...",
    "answer_hint": "Key points expected in the answer"
  }}
]"""


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=8000,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _get_context(session_id: str, topic_focus: str, k: int = 12) -> str:
    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_BASE / session_id),
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        collection_name=session_id,
    )
    query = topic_focus if topic_focus else "main topics and key concepts"
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


def _parse_json(raw: str) -> list:
    # Strip markdown fences if model ignores instructions
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned malformed JSON (possibly truncated). Try fewer questions or a shorter topic. Detail: {e}")


def generate_mcq(session_id: str, config: dict) -> list:
    context = _get_context(session_id, config.get("topic_focus", ""))
    prompt = MCQ_PROMPT.format(
        topic_focus=config.get("topic_focus") or "All topics in the material",
        num_questions=config["num_questions"],
        marks_per_question=config["marks_per_question"],
        total_marks=config["total_marks"],
        duration_minutes=config["duration_minutes"],
        difficulty=config["difficulty"],
        negative_marking="Yes" if config.get("negative_marking") else "No",
        context=context,
    )
    response = _get_llm().invoke(prompt)
    # print("\n===== RAW LLM OUTPUT (MCQ) =====\n", response.content, "\n================================\n")
    return _parse_json(response.content)


def generate_descriptive(session_id: str, config: dict) -> list:
    context = _get_context(session_id, config.get("topic_focus", ""))
    short_q = config.get("short_questions") or {}
    long_q = config.get("long_questions") or {}

    prompt = DESCRIPTIVE_PROMPT.format(
        topic_focus=config.get("topic_focus") or "All topics in the material",
        short_q_count=short_q.get("count", 0),
        short_q_marks=short_q.get("marks_each", 0),
        long_q_count=long_q.get("count", 0),
        long_q_marks=long_q.get("marks_each", 0),
        total_marks=config["total_marks"],
        duration_minutes=config["duration_minutes"],
        difficulty=config["difficulty"],
        context=context,
    )
    response = _get_llm().invoke(prompt)
    # print("\n===== RAW LLM OUTPUT (DESCRIPTIVE) =====\n", response.content, "\n=========================================\n")
    return _parse_json(response.content)
