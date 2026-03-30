import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

VECTOR_STORE_BASE = Path("vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_document(file_path: str, file_type: str) -> list:
    if file_type not in ("pdf", "docx", "txt"):
        raise ValueError(f"Unsupported file type: {file_type}")
    if file_type == "pdf":
        return PyPDFLoader(file_path).load()
    if file_type == "docx":
        return Docx2txtLoader(file_path).load()
    # txt — try UTF-8 first, fall back to latin-1
    try:
        return TextLoader(file_path, encoding="utf-8").load()
    except UnicodeDecodeError:
        return TextLoader(file_path, encoding="latin-1").load()


# def load_youtube(url: str) -> list:
#     return YoutubeLoader.from_youtube_url(url, add_video_info=False).load() 
def load_youtube(url: str) -> list:
    try:
        return YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
    except (NoTranscriptFound, TranscriptsDisabled):
        raise ValueError("This video has no captions/transcripts available. Please use a video with captions enabled.")


def create_vector_store(docs: list, session_id: str) -> tuple[object, int]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    persist_dir = str(VECTOR_STORE_BASE / session_id)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
        collection_name=session_id,
    )
    return vectorstore, len(chunks)


def delete_session_store(session_id: str) -> None:
    persist_dir = VECTOR_STORE_BASE / session_id
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
