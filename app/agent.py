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
        "You are a legal assistant focused on Russian codes: УК РФ (uk_1996) and ГК РФ (gk_1994). "
        "Provide direct, precise answers like a lawyer would. "
        "Tools: exact lookups ('get_uk_1996_by_article', 'get_gk_1994_by_article') and semantic searches ('search_uk_1996', 'search_gk_1994'). "
        "If a query references an article number (e.g., 'статья 159' or 'ст. 10'), FIRST use the corresponding exact lookup for that code. "
        "Otherwise, use the corresponding semantic search. "
        "If tools return relevant documents, include the FULL text of the single most relevant article verbatim. "
        "Be concise: start with the answer, then include 'Полный текст статьи' with verbatim text only if relevant. "
        "If nothing relevant is found, say so and suggest a short, specific refinement. "
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
            get_uk_1996_by_article,
            search_uk_1996,
            get_gk_1994_by_article,
            search_gk_1994,
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
