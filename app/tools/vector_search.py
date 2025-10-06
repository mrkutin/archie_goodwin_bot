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
    Semantic hybrid search in Qdrant collection 'uk_1996' (Уголовный кодекс РФ).

    - Use this for meaning-based queries (понятия, состав преступления, формулировки).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
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
    Exact lookup by article number in 'uk_1996' (Уголовный кодекс РФ).

    - Use when the query references a specific article (e.g., "статья 159", "ст. 105").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "159" (digits only or with "ст." prefix)
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        should_conditions = [
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
        ]
        try:
            num_val = int(digits)
            should_conditions.append(FieldCondition(key="metadata.article_number", match=MatchValue(value=num_val)))
        except Exception:
            pass

        flt = Filter(should=should_conditions)
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


# GK 1994 tools (Гражданский кодекс РФ)

@tool("search_gk_1994", return_direct=False)
def search_gk_1994(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'gk_1994' (Гражданский кодекс РФ).

    - Use for meaning-based civil-law queries (договор, обязательства, собственность и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        gk_store = QdrantVectorStore(
            client=client,
            collection_name="gk_1994",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = gk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_gk_1994_by_article", return_direct=False)
def get_gk_1994_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'gk_1994' (Гражданский кодекс РФ).

    - Use when the query references a specific article (e.g., "статья 10 ГК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "10"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="gk_1994",
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


# APK 2002 tools (Арбитражный процессуальный кодекс РФ)

@tool("search_apk_2002", return_direct=False)
def search_apk_2002(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'apk_2002' (Арбитражный процессуальный кодекс РФ).

    - Use for meaning-based procedural queries (исковое заявление, подсудность, сроки и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        apk_store = QdrantVectorStore(
            client=client,
            collection_name="apk_2002",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = apk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_apk_2002_by_article", return_direct=False)
def get_apk_2002_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'apk_2002' (Арбитражный процессуальный кодекс РФ).

    - Use when the query references a specific article (e.g., "статья 1 АПК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "1"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="apk_2002",
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


# BK 1998 tools (Бюджетный кодекс РФ)

@tool("search_bk_1998", return_direct=False)
def search_bk_1998(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'bk_1998' (Бюджетный кодекс Российской Федерации).

    - Use for meaning-based budget-law queries (ассигнования, бюджет, межбюджетные трансферты и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        bk_store = QdrantVectorStore(
            client=client,
            collection_name="bk_1998",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = bk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_bk_1998_by_article", return_direct=False)
def get_bk_1998_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'bk_1998' (Бюджетный кодекс Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 6 БК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "6"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="bk_1998",
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


# GPK 2002 tools (Гражданский процессуальный кодекс РФ)

@tool("search_gpk_2002", return_direct=False)
def search_gpk_2002(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'gpk_2002' (Гражданский процессуальный кодекс Российской Федерации).

    - Use for meaning-based civil procedure queries (исковое производство, подсудность, доказательства и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        gpk_store = QdrantVectorStore(
            client=client,
            collection_name="gpk_2002",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = gpk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_gpk_2002_by_article", return_direct=False)
def get_gpk_2002_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'gpk_2002' (Гражданский процессуальный кодекс Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 131 ГПК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "131"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="gpk_2002",
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


# GSK 2004 tools (Градостроительный кодекс РФ)

@tool("search_gsk_2004", return_direct=False)
def search_gsk_2004(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'gsk_2004' (Градостроительный кодекс Российской Федерации).

    - Use for meaning-based urban planning queries (застройка, разрешение на строительство, ПЗЗ и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        gsk_store = QdrantVectorStore(
            client=client,
            collection_name="gsk_2004",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = gsk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_gsk_2004_by_article", return_direct=False)
def get_gsk_2004_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'gsk_2004' (Градостроительный кодекс Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 51 ГрК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "51"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="gsk_2004",
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


# KOAP 2001 tools (Кодекс об административных правонарушениях Российской Федерации)

@tool("search_koap_2001", return_direct=False)
def search_koap_2001(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'koap_2001' (Кодекс об административных правонарушениях Российской Федерации).

    - Use for meaning-based administrative-offense queries (состав, санкции, субъекты и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        koap_store = QdrantVectorStore(
            client=client,
            collection_name="koap_2001",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = koap_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_koap_2001_by_article", return_direct=False)
def get_koap_2001_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'koap_2001' (Кодекс об административных правонарушениях Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 12.9 КоАП").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "12.9" or "19.3"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    # Keep digits and dot for KoAP compound articles (e.g., 12.9)
    normalized = "".join(ch for ch in str(article_number) if (ch.isdigit() or ch == "."))
    if not normalized:
        return "Article number must contain digits."
    try:
        should_conditions = [
            FieldCondition(key="metadata.article_number", match=MatchValue(value=normalized)),
        ]
        # Also try pure integer if no dot
        if "." not in normalized:
            try:
                should_conditions.append(FieldCondition(key="metadata.article_number", match=MatchValue(value=int(normalized))))
            except Exception:
                pass
        flt = Filter(should=should_conditions)
        points, _ = client.scroll(
            collection_name="koap_2001",
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
            article_num = meta.get("article_number") or normalized
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


# KTM 1999 tools (Кодекс торгового мореплавания Российской Федерации)

@tool("search_ktm_1999", return_direct=False)
def search_ktm_1999(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'ktm_1999' (Кодекс торгового мореплавания Российской Федерации).

    - Use for maritime law queries (перевозка грузов, чартер, капитан, аварии и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        ktm_store = QdrantVectorStore(
            client=client,
            collection_name="ktm_1999",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = ktm_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_ktm_1999_by_article", return_direct=False)
def get_ktm_1999_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'ktm_1999' (Кодекс торгового мореплавания Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 142 КТМ").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "142"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="ktm_1999",
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


# KVVT 2001 tools (Кодекс внутреннего водного транспорта Российской Федерации)

@tool("search_kvvt_2001", return_direct=False)
def search_kvvt_2001(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'kvvt_2001' (Кодекс внутреннего водного транспорта Российской Федерации).

    - Use for inland water transport queries (перевозка, судоходство, порт, путевой лист и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        kvvt_store = QdrantVectorStore(
            client=client,
            collection_name="kvvt_2001",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = kvvt_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_kvvt_2001_by_article", return_direct=False)
def get_kvvt_2001_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'kvvt_2001' (Кодекс внутреннего водного транспорта Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 65 КВВТ").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "65"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="kvvt_2001",
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


# LK 2006 tools (Лесной кодекс Российской Федерации)

@tool("search_lk_2006", return_direct=False)
def search_lk_2006(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'lk_2006' (Лесной кодекс Российской Федерации).

    - Use for forest law queries (лесные участки, аренда, вырубка, воспроизводство и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        lk_store = QdrantVectorStore(
            client=client,
            collection_name="lk_2006",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = lk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_lk_2006_by_article", return_direct=False)
def get_lk_2006_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'lk_2006' (Лесной кодекс Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 29 ЛК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "29"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="lk_2006",
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


# NK 1998 tools (Налоговый кодекс Российской Федерации)

@tool("search_nk_1998", return_direct=False)
def search_nk_1998(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'nk_1998' (Налоговый кодекс Российской Федерации).

    - Use for tax law queries (НДС, налог на прибыль, вычеты, объекты налогообложения и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        nk_store = QdrantVectorStore(
            client=client,
            collection_name="nk_1998",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = nk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_nk_1998_by_article", return_direct=False)
def get_nk_1998_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'nk_1998' (Налоговый кодекс Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 169 НК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "169"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="nk_1998",
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


# SK 1995 tools (Семейный кодекс Российской Федерации)

@tool("search_sk_1995", return_direct=False)
def search_sk_1995(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'sk_1995' (Семейный кодекс Российской Федерации).

    - Use for family law queries (брак, развод, алименты, родители и дети и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.

    Args:
    - query: natural-language query in Russian
    - top_k: number of results (default 5)
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        sk_store = QdrantVectorStore(
            client=client,
            collection_name="sk_1995",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = sk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_sk_1995_by_article", return_direct=False)
def get_sk_1995_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'sk_1995' (Семейный кодекс Российской Федерации).

    - Use when the query references a specific article (e.g., "статья 80 СК").
    - Matches metadata.article_number (string or integer).
    - Returns full article text with chapter/article titles when available.

    Args:
    - article_number: e.g., "80"
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="sk_1995",
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


# TK 2001 tools (Трудовой кодекс Российской Федерации)

@tool("search_tk_2001", return_direct=False)
def search_tk_2001(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in Qdrant collection 'tk_2001' (Трудовой кодекс Российской Федерации).

    - Use for labor law queries (трудовой договор, увольнение, отпуск, заработная плата и т.п.).
    - Returns up to top_k most relevant articles with full text and basic metadata.
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        tk_store = QdrantVectorStore(
            client=client,
            collection_name="tk_2001",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = tk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_tk_2001_by_article", return_direct=False)
def get_tk_2001_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'tk_2001' (Трудовой кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="tk_2001",
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


# UIK 1997 tools (Уголовно-исполнительный кодекс Российской Федерации)

@tool("search_uik_1997", return_direct=False)
def search_uik_1997(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in 'uik_1997' (Уголовно-исполнительный кодекс Российской Федерации).
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        uik_store = QdrantVectorStore(
            client=client,
            collection_name="uik_1997",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = uik_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_uik_1997_by_article", return_direct=False)
def get_uik_1997_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'uik_1997' (Уголовно-исполнительный кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="uik_1997",
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


# UPK 2001 tools (Уголовно-процессуальный кодекс Российской Федерации)

@tool("search_upk_2001", return_direct=False)
def search_upk_2001(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in 'upk_2001' (Уголовно-процессуальный кодекс Российской Федерации).
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        upk_store = QdrantVectorStore(
            client=client,
            collection_name="upk_2001",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = upk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_upk_2001_by_article", return_direct=False)
def get_upk_2001_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'upk_2001' (Уголовно-процессуальный кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="upk_2001",
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


# VK 2006 tools (Водный кодекс Российской Федерации)

@tool("search_vk_2006", return_direct=False)
def search_vk_2006(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in 'vk_2006' (Водный кодекс Российской Федерации).
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        vk_store = QdrantVectorStore(
            client=client,
            collection_name="vk_2006",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = vk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_vk_2006_by_article", return_direct=False)
def get_vk_2006_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'vk_2006' (Водный кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="vk_2006",
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


# VOZK 1997 tools (Воздушный кодекс Российской Федерации)

@tool("search_vozk_1997", return_direct=False)
def search_vozk_1997(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in 'vozk_1997' (Воздушный кодекс Российской Федерации).
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        vozk_store = QdrantVectorStore(
            client=client,
            collection_name="vozk_1997",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = vozk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_vozk_1997_by_article", return_direct=False)
def get_vozk_1997_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'vozk_1997' (Воздушный кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="vozk_1997",
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


# ZHK 2004 tools (Жилищный кодекс Российской Федерации)

@tool("search_zhk_2004", return_direct=False)
def search_zhk_2004(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in 'zhk_2004' (Жилищный кодекс Российской Федерации).
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        zhk_store = QdrantVectorStore(
            client=client,
            collection_name="zhk_2004",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = zhk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_zhk_2004_by_article", return_direct=False)
def get_zhk_2004_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'zhk_2004' (Жилищный кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="zhk_2004",
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


# ZK 2001 tools (Земельный кодекс Российской Федерации)

@tool("search_zk_2001", return_direct=False)
def search_zk_2001(query: str, top_k: int = 5) -> str:
    """
    Semantic hybrid search in 'zk_2001' (Земельный кодекс Российской Федерации).
    """
    if not query or not query.strip():
        return "Query is empty."
    try:
        zk_store = QdrantVectorStore(
            client=client,
            collection_name="zk_2001",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docs = zk_store.similarity_search(query, k=max(1, int(top_k)))
        return _format_docs(docs)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool("get_zk_2001_by_article", return_direct=False)
def get_zk_2001_by_article(article_number: str) -> str:
    """
    Exact lookup by article number in 'zk_2001' (Земельный кодекс Российской Федерации).
    """
    if not article_number or not str(article_number).strip():
        return "Article number is empty."
    digits = "".join(ch for ch in str(article_number) if ch.isdigit())
    if not digits:
        return "Article number must contain digits."
    try:
        flt = Filter(should=[
            FieldCondition(key="metadata.article_number", match=MatchValue(value=digits)),
            FieldCondition(key="metadata.article_number", match=MatchValue(value=int(digits))) if digits.isdigit() else None,
        ])
        flt.should = [c for c in flt.should if c is not None]
        points, _ = client.scroll(
            collection_name="zk_2001",
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

    # UK tests
    # print("\n== Semantic query: мошенничество")
    # print(search_uk_1996.invoke({"query": "мошенничество", "top_k": 3}))

    print("\n== Article number lookup (UK): 1")
    print(get_uk_1996_by_article.invoke({"article_number": "1"}))

    # GK tests
    print("\n== Semantic query (GK): договор купли-продажи")
    print(search_gk_1994.invoke({"query": "договор купли-продажи", "top_k": 3}))

    print("\n== Article number lookup (GK): 10")
    print(get_gk_1994_by_article.invoke({"article_number": "10"}))

    # APK tests
    print("\n== Semantic query (APK): исковое заявление")
    print(search_apk_2002.invoke({"query": "исковое заявление", "top_k": 3}))

    print("\n== Article number lookup (APK): 1")
    print(get_apk_2002_by_article.invoke({"article_number": "1"}))
