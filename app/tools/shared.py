import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Load .env early so QDRANT_* vars are available at import time
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # type: ignore
except Exception:
    pass

# Disable HuggingFace tokenizers parallelism warnings (fork-safe)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DENSE_MODEL = os.getenv("DENSE_EMBEDDINGS_MODEL", "ai-forever/FRIDA")
SPARSE_MODEL = os.getenv("SPARSE_EMBEDDINGS_MODEL", "Qdrant/bm25")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)

dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)

# Import lazily to avoid cost at import time in some environments
from langchain_qdrant import FastEmbedSparse  # noqa: E402  # type: ignore

sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant documents found."
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", None) or {}
        chapter_title = meta.get("chapter_title") or ""
        chapter_num = meta.get("chapter_number") or ""
        article_title = meta.get("article_title") or ""
        article_num = meta.get("article_number") or ""
        content = getattr(d, "page_content", "") or ""
        line = (
            f"[{i}]\n"
            f"Глава: {chapter_title} (номер: {chapter_num})\n"
            f"Статья: {article_title} (номер: {article_num})\n"
            f"Содержание: {content}"
        )
        lines.append(line)
    return "\n\n".join(lines)
