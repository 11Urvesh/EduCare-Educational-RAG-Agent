import re
import shutil
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from backend.vectorstore import VECTOR_STORE_BASE, create_store, get_embeddings

# Initialized once at import time — Docling downloads layout/OCR models on first run (~500MB)
_converter = DocumentConverter()

HEADERS_TO_SPLIT = [("#", "H1"), ("##", "H2"), ("###", "H3")]
TABLE_PATTERN = re.compile(r"(\|.+\|[ \t]*\n)+", re.MULTILINE)
MIN_SECTION_LENGTH = 300  # sections shorter than this are kept as a single chunk


# ── Helpers ─────────────────────────────────────────────────────────────────

def _has_headers(text: str) -> bool:
    return bool(re.search(r"^#{1,3}\s", text, re.MULTILINE))


def _extract_tables(text: str) -> tuple[str, list[str]]:
    """Replace markdown tables with placeholders so SemanticChunker never splits them."""
    tables: list[str] = []
    def replacer(m):
        tables.append(m.group(0))
        return f" TABLE_PLACEHOLDER_{len(tables) - 1} "
    return TABLE_PATTERN.sub(replacer, text), tables


def _restore_tables(text: str, tables: list[str]) -> str:
    for i, table in enumerate(tables):
        text = text.replace(f" TABLE_PLACEHOLDER_{i} ", f"\n\n{table}\n")
    return text


# ── Chunking strategies ──────────────────────────────────────────────────────

def _semantic_chunk(text: str, metadata: dict) -> list[Document]:
    """Semantic chunking with table protection."""
    cleaned, tables = _extract_tables(text)

    if not cleaned.strip():
        return [Document(page_content=text.strip(), metadata=metadata)]

    splitter = SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile")
    chunks = splitter.create_documents([cleaned], metadatas=[metadata])

    return [
        Document(
            page_content=_restore_tables(c.page_content, tables).strip(),
            metadata=c.metadata,
        )
        for c in chunks
        if c.page_content.strip()
    ]


def _structure_aware_chunk(markdown: str, source: str) -> list[Document]:
    """
    Split on headers first → semantic chunk within each section.
    Each chunk carries section title in metadata for better retrieval context.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    sections = header_splitter.split_text(markdown)

    chunks = []
    for section in sections:
        title = (
            section.metadata.get("H3")
            or section.metadata.get("H2")
            or section.metadata.get("H1")
            or "General"
        )
        metadata = {"source": source, "section": title}
        content = section.page_content.strip()

        if not content:
            continue

        if len(content) < MIN_SECTION_LENGTH:
            chunks.append(Document(page_content=content, metadata=metadata))
        else:
            chunks.extend(_semantic_chunk(content, metadata))

    return chunks


# ── Document loaders ─────────────────────────────────────────────────────────

def load_document(file_path: str, file_type: str) -> list[Document]:
    """
    Parse PDF/DOCX via Docling (handles text, scanned pages, tables, images).
    Parse TXT directly. Both routes detect headers and apply the right chunking strategy.
    """
    if file_type not in ("pdf", "docx", "txt"):
        raise ValueError(f"Unsupported file type: {file_type}")

    source = Path(file_path).name

    if file_type in ("pdf", "docx"):
        result = _converter.convert(file_path)
        markdown = result.document.export_to_markdown()
    else:
        try:
            markdown = Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            markdown = Path(file_path).read_text(encoding="latin-1")

    if _has_headers(markdown):
        return _structure_aware_chunk(markdown, source)
    return _semantic_chunk(markdown, {"source": source})


def load_youtube(url: str) -> list[Document]:
    """Extract YouTube transcript and semantically chunk it."""
    try:
        raw_docs = YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
    except (NoTranscriptFound, TranscriptsDisabled):
        raise ValueError(
            "This video has no captions/transcripts available. "
            "Please use a video with captions enabled."
        )
    text = "\n\n".join(doc.page_content for doc in raw_docs)
    return _semantic_chunk(text, {"source": url})


# ── Vector store ─────────────────────────────────────────────────────────────

def create_vector_store(docs: list[Document], session_id: str) -> tuple[object, int]:
    """Embed and store pre-chunked documents. No splitting done here."""
    store = create_store(docs, session_id)
    return store, len(docs)


def delete_session_store(session_id: str) -> None:
    persist_dir = VECTOR_STORE_BASE / session_id
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
