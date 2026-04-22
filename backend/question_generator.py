import json
import os
from pathlib import Path

from langchain_groq import ChatGroq
from backend.vectorstore import load_store

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _get_llm() -> ChatGroq:
    return ChatGroq( 
        model="llama-3.1-8b-instant", # "llama-3.3-70b-versatile"
        temperature=0.5,
        max_tokens=8000,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _get_context(session_id: str, topic_focus: str, k: int = 12) -> str:
    vectorstore = load_store(session_id)
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
    prompt = _load_prompt("mcq_prompt.txt").format(
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

    short_count = short_q.get("count", 0)
    prompt = _load_prompt("descriptive_prompt.txt").format(
        topic_focus=config.get("topic_focus") or "All topics in the material",
        short_q_count=short_count,
        short_q_marks=short_q.get("marks_each", 0),
        long_q_count=long_q.get("count", 0),
        long_q_marks=long_q.get("marks_each", 0),
        long_q_start=short_count + 1,
        total_marks=config["total_marks"],
        duration_minutes=config["duration_minutes"],
        difficulty=config["difficulty"],
        context=context,
    )
    response = _get_llm().invoke(prompt)
    # print("\n===== RAW LLM OUTPUT (DESCRIPTIVE) =====\n", response.content, "\n=========================================\n")
    return _parse_json(response.content)
