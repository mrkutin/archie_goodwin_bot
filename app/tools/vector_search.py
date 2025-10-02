import os
from typing import List

from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


# Disable HuggingFace tokenizers parallelism warnings (fork-safe)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
def search_uk_1996(query: str, top_k: int = 5) -> str:
    """
    Search within Qdrant collection 'uk_1996' (Уголовный кодекс РФ) using hybrid retrieval.
    Returns full article text when available.
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        docs = vector_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_uk_1996_by_article", return_direct=False)
def get_uk_1996_by_article(article_number: str) -> str:
    """
    Retrieve exact article by number from 'uk_1996' using metadata filter.
    - article_number: e.g., "159"
    Returns full article text with chapter/article titles when available.
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    # Normalize to digits only (e.g., strip 'ст.' prefix, spaces)
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        # Some ingesters store metadata nested under "metadata.*"; also type may be str or int.
        should_conditions = [
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
        ]
        # Also try numeric match
        try:
            num_val = int(digits)
            should_conditions.extend(
                [
                    FieldCondition(key="metadata.article_number", match=MatchValue(value=num_val)),
                ]
            )
        except Exception:
            pass

        flt = Filter(should=should_conditions)
        # Use raw client scroll to avoid requiring a vector
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            with_payload=True,
            limit=10,
        )
        if not points:
            return "No article found with the specified number."
        lines: List[str] = []
        for idx, p in enumerate(points, start=1):
            payload = p.payload or {}
            meta = payload.get("metadata") or payload
            chapter_title = meta.get("chapter_title") or ""
            chapter_num = meta.get("chapter_number") or ""
            article_title = meta.get("article_title") or ""
            article_num = meta.get("article_number") or digits
            content = payload.get("page_content") or payload.get("text") or ""
            lines.append(
                f"[{idx}]\n"
                f"Глава: {chapter_title} (номер: {chapter_num})\n"
                f"Статья: {article_title} (номер: {article_num})\n"
                f"Содержание: {content}"
            )
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Lookup failed: {exc}"


if __name__ == "__main__":
    # Hardcoded tests (no CLI args)
    print(f"Testing collection: {COLLECTION_NAME} at {QDRANT_URL}")

    # print("\n== Semantic query: мошенничество")
    # print(search_uk_1996.invoke({"query": "мошенничество", "top_k": 3}))

    print("\n== Article number lookup: 1")
    print(get_uk_1996_by_article.invoke({"article_number": "1"}))
