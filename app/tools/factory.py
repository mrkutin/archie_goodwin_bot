import re
from typing import Callable, Tuple

from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Reuse initialized resources from vector_search
from app.tools.vector_search import (
    client,
    dense_embeddings,
    sparse_embeddings,
    _format_docs,
)


def _normalize_article_number(raw: str, allow_fractional: bool) -> str:
    if allow_fractional:
        return "".join(ch for ch in str(raw) if ch.isdigit() or ch == ".")
    return "".join(ch for ch in str(raw) if ch.isdigit())


def create_code_tools(
    collection_name: str,
    code_key: str,
    full_display_name: str,
    allow_fractional_articles: bool = False,
) -> Tuple[Callable[..., str], Callable[..., str]]:
    """
    Build two tools for a legal code collection:
    - search_<code_key>: semantic hybrid search returning formatted docs
    - get_<code_key>_by_article: exact article lookup (by metadata.article_number)
    """

    search_tool_name = f"search_{code_key}"
    exact_tool_name = f"get_{code_key}_by_article"

    @tool(search_tool_name, return_direct=False)
    def search_tool(query: str, top_k: int = 5) -> str:  # type: ignore
        """
        Semantic hybrid search in the given collection.
        Returns up to top_k most relevant articles with full text and basic metadata.
        """
        if not query or not str(query).strip():
            return "Query is empty."
        try:
            store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode="HYBRID",
                vector_name="dense",
                sparse_vector_name="sparse",
            )
            docs = store.similarity_search(query, k=max(1, int(top_k)))
            return _format_docs(docs)
        except Exception as exc:  # pragma: no cover
            return f"Search failed: {exc}"

    @tool(exact_tool_name, return_direct=False)
    def exact_tool(article_number: str) -> str:  # type: ignore
        """
        Exact lookup by article number in the given collection.
        Matches metadata.article_number (string or integer; fractional allowed if configured).
        Returns full article text with chapter/article titles when available.
        """
        if not article_number or not str(article_number).strip():
            return "Article number is empty."
        normalized = _normalize_article_number(article_number, allow_fractional_articles)
        if not normalized:
            return "Article number must contain digits."
        try:
            should = [
                FieldCondition(key="metadata.article_number", match=MatchValue(value=normalized))
            ]
            if "." not in normalized:
                try:
                    should.append(
                        FieldCondition(
                            key="metadata.article_number", match=MatchValue(value=int(normalized))
                        )
                    )
                except Exception:
                    pass
            flt = Filter(should=should)
            points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                with_payload=True,
                limit=10,
            )
            if not points:
                return "No article found with the specified number."
            lines = []
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
        except Exception as exc:  # pragma: no cover
            return f"Lookup failed: {exc}"

    return search_tool, exact_tool
