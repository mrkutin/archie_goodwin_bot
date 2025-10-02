import os
from typing import Dict, List, Optional

from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient


_VECTORSTORES: Dict[str, QdrantVectorStore] = {}


def _get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    host = os.getenv("QDRANT_HOST")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
    if host:
        return QdrantClient(host=host, port=port, api_key=api_key, prefer_grpc=True)
    return QdrantClient(url="http://localhost:6333", api_key=api_key, prefer_grpc=True)


def _get_vectorstore(collection_name: str) -> QdrantVectorStore:
    if collection_name in _VECTORSTORES:
        return _VECTORSTORES[collection_name]

    client = _get_qdrant_client()

    dense_model = os.getenv("DENSE_EMBEDDINGS_MODEL", "ai-forever/FRIDA")
    sparse_model = os.getenv("SPARSE_EMBEDDINGS_MODEL", "Qdrant/bm25")

    dense_embeddings = HuggingFaceEmbeddings(model_name=dense_model)
    sparse_embeddings = FastEmbedSparse(model_name=sparse_model)

    vs = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    _VECTORSTORES[collection_name] = vs
    return vs


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant documents found."
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", None) or {}
        # Ingest uses article_number/title and chapter_number/title
        title = meta.get("article_title") or meta.get("title") or "Untitled"
        article_num = meta.get("article_number")
        chapter_num = meta.get("chapter_number")
        chapter_title = meta.get("chapter_title")
        source = meta.get("source") or meta.get("url") or ""
        content = getattr(d, "page_content", "") or ""
        heading = title
        if article_num:
            heading = f"Статья {article_num}: {title}" if title else f"Статья {article_num}"
        if chapter_num:
            heading = f"Глава {chapter_num} — {chapter_title} | {heading}" if chapter_title else f"Глава {chapter_num} | {heading}"
        snippet = str(content).strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:397] + "..."
        lines.append(f"[{i}] {heading}\nSource: {source}\nExcerpt: {snippet}")
    return "\n\n".join(lines)


def _resolve_collection(default_name: str, override: Optional[str]) -> str:
    if override and override.strip():
        return override
    env_name = os.getenv("QDRANT_COLLECTION")
    return env_name or default_name


# Pure function for retrieval, safe for CLI usage without Tool wrapper

def search_kb_raw(query: str, top_k: int = 5, collection: Optional[str] = None) -> str:
    if not query or not query.strip():
        return "Query is empty."
    try:
        collection_name = _resolve_collection("uk_1996", collection)
        vs = _get_vectorstore(collection_name)
        docs = vs.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("search_kb", return_direct=False)
def search_kb(query: str, top_k: int = 5, collection: Optional[str] = None) -> str:
    """
    Hybrid search (dense + BM25) in a Qdrant collection (named vectors: dense/sparse).
    - query: text to search
    - top_k: number of results
    - collection: collection name (default env QDRANT_COLLECTION or 'uk_1996')
    """
    return search_kb_raw(query=query, top_k=top_k, collection=collection)


if __name__ == "__main__":
    # Simple CLI test harness
    import sys

    collection = _resolve_collection("uk_1996", None)
    queries = [
        # "мошенничество",
        # "Статья 159",
        # "ответственность за кражу",
        "тайное хищение чужого имущества",
    ]
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    print(f"Testing collection: {collection}")
    for q in queries:
        print(f"\n== Query: {q}")
        result = search_kb_raw(query=q, top_k=3, collection=collection)
        print(result)
