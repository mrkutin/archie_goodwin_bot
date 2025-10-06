from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

from app.tools.vector_search import (
    search_uk_1996,
    get_uk_1996_by_article,
    search_gk_1994,
    get_gk_1994_by_article,
    search_apk_2002,
    get_apk_2002_by_article,
    search_bk_1998,
    get_bk_1998_by_article,
    search_gpk_2002,
    get_gpk_2002_by_article,
    search_gsk_2004,
    get_gsk_2004_by_article,
    search_koap_2001,
    get_koap_2001_by_article,
    search_ktm_1999,
    get_ktm_1999_by_article,
    search_kvvt_2001,
    get_kvvt_2001_by_article,
    search_lk_2006,
    get_lk_2006_by_article,
    search_nk_1998,
    get_nk_1998_by_article,
    search_sk_1995,
    get_sk_1995_by_article,
    search_tk_2001,
    get_tk_2001_by_article,
    search_uik_1997,
    get_uik_1997_by_article,
    search_upk_2001,
    get_upk_2001_by_article,
    search_vk_2006,
    get_vk_2006_by_article,
    search_vozk_1997,
    get_vozk_1997_by_article,
    search_zhk_2004,
    get_zhk_2004_by_article,
    search_zk_2001,
    get_zk_2001_by_article,
)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()  # type: ignore


def _build_system_prompt() -> str:
    return (
        "ROLE\n"
        "You are a precise legal assistant. Your scope covers: \n"
        "- Уголовный, Гражданский, Арбитражный процессуальный, Гражданский процессуальный, Градостроительный, \n"
        "  КоАП РФ, Кодекс торгового мореплавания, Кодекс внутреннего водного транспорта, Лесной, Налоговый, Семейный,\n"
        "  Трудовой, Уголовно-исполнительный, Уголовно-процессуальный, Водный, Воздушный, Жилищный, Земельный, Бюджетный кодексы РФ.\n\n"
        "POLICY\n"
        "1) First decide if the query references a specific article number for a particular code.\n"
        "   - If yes: use the exact-lookup tool for that code.\n"
        "   - If no: use the semantic search tool for that code.\n"
        "2) If the tool returns multiple articles, select ONE most relevant by:\n"
        "   - Matching the referenced article number (if provided).\n"
        "   - Maximizing semantic alignment with the user's question (key terms, legal institute, facts, remedies).\n"
        "   - Preferring articles whose title/section explicitly addresses the user’s intent.\n"
        "   - If ties remain, choose the most specific (not general provisions). Optionally mention 1 close alternative.\n"
        "3) Provide a direct, concise legal answer first. Do not add generic commentary.\n"
        "4) If tools return relevant documents, include the FULL verbatim text of the single most relevant article\n"
        "   under a section titled 'Полный текст статьи'. Only include this section if it clearly supports the answer.\n"
        "5) If nothing relevant is found, say so and suggest a short, specific refinement (e.g., article number or key terms).\n\n"
        "TOOL MAP (Code → exact lookup; semantic search)\n"
        "- Уголовный кодекс РФ: get_uk_1996_by_article; search_uk_1996\n"
        "- Гражданский кодекс РФ: get_gk_1994_by_article; search_gk_1994\n"
        "- Арбитражный процессуальный кодекс РФ: get_apk_2002_by_article; search_apk_2002\n"
        "- Гражданский процессуальный кодекс РФ: get_gpk_2002_by_article; search_gpk_2002\n"
        "- Градостроительный кодекс РФ: get_gsk_2004_by_article; search_gsk_2004\n"
        "- КоАП РФ (supports fractional numbers like 12.9): get_koap_2001_by_article; search_koap_2001\n"
        "- Кодекс торгового мореплавания РФ: get_ktm_1999_by_article; search_ktm_1999\n"
        "- Кодекс внутреннего водного транспорта РФ: get_kvvt_2001_by_article; search_kvvt_2001\n"
        "- Лесной кодекс РФ: get_lk_2006_by_article; search_lk_2006\n"
        "- Налоговый кодекс РФ: get_nk_1998_by_article; search_nk_1998\n"
        "- Семейный кодекс РФ: get_sk_1995_by_article; search_sk_1995\n"
        "- Трудовой кодекс РФ: get_tk_2001_by_article; search_tk_2001\n"
        "- Уголовно-исполнительный кодекс РФ: get_uik_1997_by_article; search_uik_1997\n"
        "- Уголовно-процессуальный кодекс РФ: get_upk_2001_by_article; search_upk_2001\n"
        "- Водный кодекс РФ: get_vk_2006_by_article; search_vk_2006\n"
        "- Воздушный кодекс РФ: get_vozk_1997_by_article; search_vozk_1997\n"
        "- Жилищный кодекс РФ: get_zhk_2004_by_article; search_zhk_2004\n"
        "- Земельный кодекс РФ: get_zk_2001_by_article; search_zk_2001\n"
        "- Бюджетный кодекс РФ: get_bk_1998_by_article; search_bk_1998\n\n"
        "INPUT NORMALIZATION\n"
        "- When using exact-lookup tools, normalize article references: strip 'ст.'/'статья', spaces; keep digits (and dot for КоАП).\n\n"
        "ANSWER FORMAT\n"
        "- Begin with: a short, decisive legal answer in Russian.\n"
        "- Optionally add: 'Полный текст статьи' with verbatim law text (only if relevant).\n"
        "- Avoid tables unless explicitly requested. Keep language formal and unambiguous.\n"
    )


_agent_ref: Dict[str, Any] = {}
_checkpointer: Optional[InMemorySaver] = None


def get_agent() -> Any:
    global _agent_ref, _checkpointer
    _load_env()
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
    if "agent" not in _agent_ref:
        model = init_chat_model(
            "openai:gpt-4.1",
            temperature=0
        )
        tools = [
            get_uk_1996_by_article, search_uk_1996,
            get_gk_1994_by_article, search_gk_1994,
            get_apk_2002_by_article, search_apk_2002,
            get_gpk_2002_by_article, search_gpk_2002,
            get_gsk_2004_by_article, search_gsk_2004,
            get_koap_2001_by_article, search_koap_2001,
            get_ktm_1999_by_article, search_ktm_1999,
            get_kvvt_2001_by_article, search_kvvt_2001,
            get_lk_2006_by_article, search_lk_2006,
            get_nk_1998_by_article, search_nk_1998,
            get_sk_1995_by_article, search_sk_1995,
            get_tk_2001_by_article, search_tk_2001,
            get_uik_1997_by_article, search_uik_1997,
            get_upk_2001_by_article, search_upk_2001,
            get_vk_2006_by_article, search_vk_2006,
            get_vozk_1997_by_article, search_vozk_1997,
            get_zhk_2004_by_article, search_zhk_2004,
            get_zk_2001_by_article, search_zk_2001,
            get_bk_1998_by_article, search_bk_1998,
        ]
        system_prompt = _build_system_prompt()
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt,
            checkpointer=_checkpointer,
        )
        _agent_ref["agent"] = agent
    return _agent_ref["agent"]


def print_conversation(response: Dict[str, Any]) -> None:
    print("\n========== Conversation ==========")
    for msg in response.get("messages", []):
        cls_name = getattr(msg, "__class__", type(msg)).__name__
        if cls_name == "HumanMessage":
            print("\n[USER]")
            print(getattr(msg, "content", ""))
        elif cls_name == "AIMessage":
            tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
            if tool_calls:
                print("\n[AI → TOOL]")
                for call in tool_calls:
                    args = call.get("function", {}).get("arguments", "")
                    name = call.get("function", {}).get("name", "")
                    print(f"Tool: {name}\nArgs: {args}")
            if getattr(msg, "content", None):
                print("\n[AI]")
                print(getattr(msg, "content", ""))
        elif cls_name == "ToolMessage":
            print("\n[TOOL RESULT]")
            name = getattr(msg, "name", "")
            if name:
                print(f"Name: {name}")
            print(getattr(msg, "content", ""))
    print("\n========== End Conversation ==========")


def answer_question(query: str, thread_id: Optional[str] = None) -> str:
    agent = get_agent()
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": str(thread_id)}}
    response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
    if os.getenv("DEBUG_CONVERSATION"):
        print_conversation(response)
    for msg in reversed(response.get("messages", [])):
        cls_name = getattr(msg, "__class__", type(msg)).__name__
        if cls_name == "AIMessage" and getattr(msg, "content", None):
            return str(getattr(msg, "content"))
    return "I was unable to generate an answer. Please try rephrasing your request."
