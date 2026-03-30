# EduCare RAG — Educational AI Agent

An AI-powered study assistant that lets you upload study material, chat with it, and auto-generate exam papers using Retrieval-Augmented Generation (RAG).

---

## Features

- **Upload Study Material** — PDF, DOCX, TXT files or YouTube video URLs (with captions)
- **Chat with Material** — Ask questions and get answers grounded strictly in the uploaded content
- **Generate Exam Papers** — Auto-generate MCQ or Descriptive question papers with configurable marks, duration, and difficulty
- **Download PDFs** — Export question paper and answer key as separate PDF files

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| PDF Generation | fpdf2 |

---

## Project Structure

```
EduCare_RAG/
├── main.py                   # FastAPI backend server
├── .env                      # API keys (not committed)
├── .env.example              # Template for environment variables
├── requirements.txt          # Python dependencies
├── backend/
│   ├── ingestion.py          # Document loading and vectorization
│   ├── rag_chain.py          # RAG Q&A chain
│   ├── question_generator.py # MCQ and descriptive question generation
│   ├── pdf_generator.py      # PDF export
│   └── schemas.py            # Pydantic request/response models
├── frontend/
│   └── app.py                # Streamlit UI
└── vector_store/             # ChromaDB storage (auto-created, not committed)
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/EduCare-RAG.git
cd EduCare-RAG
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
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

- Get a Groq API key at [console.groq.com](https://console.groq.com)
- OpenAI key is optional (not actively used in the current version)

### 5. Run the backend

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### 6. Run the frontend

Open a second terminal:

```bash
streamlit run frontend/app.py
```

Frontend opens at `http://localhost:8501`.

---

## How It Works

```
Upload File / YouTube URL
        ↓
   Chunk into 1000-char segments (200 overlap)
        ↓
   Generate embeddings (all-MiniLM-L6-v2)
        ↓
   Store in ChromaDB (per session)
        ↓
   Chat → retrieve top-5 chunks → Groq LLM → Answer
   Generate → retrieve top-12 chunks → Groq LLM → JSON questions → PDF
```

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
- For large question papers (25+ questions), generation may take 10–20 seconds
