from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

from app.tools.registry import ALL_TOOLS

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
        "- Арбитражный процессуальный\n"
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
        "- Арбитражный процессуальный кодекс РФ: get_apk_2002_by_article; search_apk_2002\n"
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
        tools = ALL_TOOLS
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
