from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_BASE = Path("vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── To swap to Qdrant later, replace only the two functions below ──
# from langchain_qdrant import QdrantVectorStore
# import qdrant_client


_embeddings: HuggingFaceEmbeddings | None = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def create_store(docs: list, session_id: str):
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=str(VECTOR_STORE_BASE / session_id),
        collection_name=session_id,
    )


def load_store(session_id: str):
    return Chroma(
        persist_directory=str(VECTOR_STORE_BASE / session_id),
        embedding_function=get_embeddings(),
        collection_name=session_id,
    )
