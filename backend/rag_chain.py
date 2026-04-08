import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_BASE = Path("vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

CHAT_PROMPT = ChatPromptTemplate.from_template(
    (PROMPTS_DIR / "chat_prompt.txt").read_text(encoding="utf-8")
)


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _load_vectorstore(session_id: str) -> Chroma:
    return Chroma(
        persist_directory=str(VECTOR_STORE_BASE / session_id),
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        collection_name=session_id,
    )


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_answer(session_id: str, question: str) -> dict:
    vectorstore = _load_vectorstore(session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | CHAT_PROMPT
        | _get_llm()
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    # Fetch sources separately
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "study material") for doc in docs})

    return {"answer": answer, "sources": sources}
