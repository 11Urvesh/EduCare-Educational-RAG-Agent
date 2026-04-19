# EduCare RAG — Educational AI Agent

An AI-powered study assistant that lets you upload study material, chat with it, and auto-generate exam papers using Retrieval-Augmented Generation (RAG).

---

## Features

- **Upload Study Material** — PDF, DOCX, TXT files or YouTube video URLs (with captions)
- **Advanced PDF Parsing** — Handles scanned documents, tables, images via Docling (OCR + layout analysis)
- **Structure-Aware Chunking** — Splits on document headers first, then semantically within sections for precise retrieval
- **Chat with Material** — Ask questions and get answers grounded in uploaded content, with LLM reasoning and explanation
- **Generate Exam Papers** — Auto-generate MCQ or Descriptive question papers with configurable marks, duration, and difficulty
- **Download PDFs** — Export Question Paper and Answer Key as separate PDF files
- **Pluggable Vector Store** — ChromaDB by default, swap to Qdrant by editing one file

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| PDF Parsing | Docling (OCR + layout + table extraction) |
| Chunking | Structure-aware (MarkdownHeaderTextSplitter → SemanticChunker) |
| Vector DB | ChromaDB (pluggable via `backend/vectorstore.py`) |
| RAG Framework | LangChain |
| PDF Generation | fpdf2 |

---

## Project Structure

```
EduCare_RAG/
├── main.py                       # FastAPI backend server
├── .env                          # API keys (not committed)
├── .env.example                  # Template for environment variables
├── requirements.txt              # Python dependencies
├── backend/
│   ├── vectorstore.py            # Vector store abstraction (swap ChromaDB ↔ Qdrant here)
│   ├── ingestion.py              # Docling parsing + structure-aware chunking
│   ├── rag_chain.py              # RAG Q&A chain
│   ├── question_generator.py     # MCQ and descriptive question generation
│   ├── pdf_generator.py          # PDF export
│   └── schemas.py                # Pydantic request/response models
├── frontend/
│   └── app.py                    # Streamlit UI
├── prompts/
│   ├── chat_prompt.txt           # System prompt for chat
│   ├── mcq_prompt.txt            # System prompt for MCQ generation
│   └── descriptive_prompt.txt    # System prompt for descriptive generation
└── vector_store/                 # ChromaDB storage (auto-created, not committed)
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/11Urvesh/EduCare-Educational-RAG-Agent.git
cd EduCare-Educational-RAG-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Open `.env` and fill in your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

Get a Groq API key at [console.groq.com](https://console.groq.com)

### 5. Pre-download Docling models (first-time only)

On first run, Docling downloads layout and OCR models (~500MB) to your HuggingFace cache. Run this once before starting the server so it doesn't happen mid-request:

```bash
python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"
```

> **Windows users:** Enable Developer Mode (`Settings → Privacy & Security → For Developers → Developer Mode → ON`) before this step to allow symlinks in the HuggingFace cache.

### 6. Run the backend

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 7. Run the frontend

Open a second terminal:

```bash
streamlit run frontend/app.py
```

Frontend opens at `http://localhost:8501`.

> Start the **backend first**, then the frontend.

---

## RAG Pipeline

```
Upload File / YouTube URL
        ↓
   Docling (PDF/DOCX) or direct read (TXT)
   → Structured Markdown with headers, tables, sections
        ↓
   Has headers (#, ##, ###)?
   ├── YES → MarkdownHeaderTextSplitter → named sections
   │          → SemanticChunker within each section
   │          → Each chunk tagged: {source, section, page}
   │
   └── NO  → SemanticChunker on full text
              → Each chunk tagged: {source, page}
        ↓
   ChromaDB (embeddings: all-MiniLM-L6-v2)
        ↓
   Chat  → retrieve top-5 chunks → Groq LLM → Answer
   Generate → retrieve top-12 chunks → Groq LLM → JSON questions → PDF
```

---

## Swapping Vector Store (ChromaDB → Qdrant)

All vector store logic is isolated in `backend/vectorstore.py`. To switch to Qdrant:

```python
# backend/vectorstore.py — replace create_store() and load_store() only

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def create_store(docs, session_id):
    client = QdrantClient(url="http://localhost:6333")
    return QdrantVectorStore.from_documents(docs, get_embeddings(), client=client, collection_name=session_id)

def load_store(session_id):
    client = QdrantClient(url="http://localhost:6333")
    return QdrantVectorStore(client=client, collection_name=session_id, embedding=get_embeddings())
```

No other file needs to change.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload file or YouTube URL |
| POST | `/chat` | Ask a question |
| POST | `/generate-questions` | Generate exam paper |
| DELETE | `/session/{id}` | Delete a session |

---

## Notes

- Sessions are not persisted — vector stores are cleared on server restart
- YouTube URLs must have captions/transcripts enabled
- Docling models are cached in `~/.cache/huggingface/hub/` after first download
- Prompt templates are in `prompts/` — edit `.txt` files to tune LLM behaviour without touching Python code
