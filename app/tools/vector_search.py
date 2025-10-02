import os
from typing import List

from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient


# Single collection and single vector store (simple, no wrappers)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "uk_1996")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DENSE_MODEL = os.getenv("DENSE_EMBEDDINGS_MODEL", "ai-forever/FRIDA")
SPARSE_MODEL = os.getenv("SPARSE_EMBEDDINGS_MODEL", "Qdrant/bm25")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)

dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)


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


@tool("search_uk_1996", return_direct=False)
def search_uk_1996(query: str, top_k: int = 3) -> str:
    """
    Search within Qdrant collection 'uk_1996' (Уголовный кодекс РФ) using hybrid retrieval.
    Returns concise citations with article and chapter context when available.
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        docs = vector_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


if __name__ == "__main__":
    # Minimal CLI test
    import sys

    queries = [
        "мошенничество",
        # "Статья 159",
        # "тайное хищение чужого имущества",
    ]
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    print(f"Testing collection: {COLLECTION_NAME} at {QDRANT_URL}")
    for q in queries:
        print(f"\n== Query: {q}")
        result = search_uk_1996.invoke({"query": q, "top_k": 3})
        print(result)
