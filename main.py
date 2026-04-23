import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.ingestion import create_vector_store, delete_session_store, load_document, load_youtube
from backend.vectorstore import get_embeddings
from backend.question_generator import generate_descriptive, generate_mcq
from backend.rag_chain import get_answer
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    UploadResponse,
)

load_dotenv()

VECTOR_STORE_BASE = Path("vector_store")
TEMP_UPLOAD_DIR = Path("temp_uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: wipe leftover stores from any previous crashed run
    if VECTOR_STORE_BASE.exists():
        shutil.rmtree(VECTOR_STORE_BASE)
    VECTOR_STORE_BASE.mkdir(parents=True, exist_ok=True)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    get_embeddings()  # warm up embedding model once at startup
    yield
    # Shutdown: clean everything
    if VECTOR_STORE_BASE.exists():
        shutil.rmtree(VECTOR_STORE_BASE)
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR)


app = FastAPI(title="EduCare RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_material(
    file: UploadFile = File(None),
    youtube_url: str = Form(None),
    old_session_id: str = Form(None),
):
    if not file and not youtube_url:
        raise HTTPException(status_code=400, detail="Provide a file or a YouTube URL.")

    # Delete the previous session's vector store before creating a new one
    if old_session_id:
        delete_session_store(old_session_id)

    session_id = str(uuid.uuid4())

    try:
        if youtube_url:
            docs = load_youtube(youtube_url.strip())
        else:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in ("pdf", "docx", "txt"):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

            temp_path = TEMP_UPLOAD_DIR / f"{session_id}.{ext}"
            temp_path.write_bytes(await file.read())

            docs = load_document(str(temp_path), ext)
            temp_path.unlink()

        _, chunk_count = create_vector_store(docs, session_id)

        return UploadResponse(
            session_id=session_id,
            message="Study material processed successfully.",
            chunks_created=chunk_count,
        )

    except HTTPException:
        raise
    except ValueError as e:
        delete_session_store(session_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        delete_session_store(session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not (VECTOR_STORE_BASE / req.session_id).exists():
        raise HTTPException(status_code=404, detail="Session not found. Please upload study material first.")

    result = get_answer(req.session_id, req.question)
    return ChatResponse(**result)


@app.post("/generate-questions", response_model=GenerateResponse)
def generate_questions(req: GenerateRequest):
    if not (VECTOR_STORE_BASE / req.session_id).exists():
        raise HTTPException(status_code=404, detail="Session not found. Please upload study material first.")

    try:
        config = req.model_dump()
        questions = generate_mcq(req.session_id, config) if req.type == "mcq" else generate_descriptive(req.session_id, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    metadata = {
        "type": req.type,
        "total_marks": req.total_marks,
        "duration_minutes": req.duration_minutes,
        "difficulty": req.difficulty,
        "topic_focus": req.topic_focus or "General",
    }
    return GenerateResponse(paper_type=req.type, questions=questions, metadata=metadata)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    delete_session_store(session_id)
    return {"message": f"Session {session_id} deleted."}
