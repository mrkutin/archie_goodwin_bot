from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

from app.tools.vector_search import search_uk_1996

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()  # type: ignore


def _build_system_prompt() -> str:
    today_utc_str = datetime.now(timezone.utc).date().isoformat()
    return (
        f"Today is: {today_utc_str} (UTC). "
        "Always use UTC for any date/time reasoning. "
        "You are a legal assistant that helps users find and understand the exact text of Russian Criminal Code (УК РФ) articles. "
        "You have access to a tool named 'search_uk_1996' that searches a Qdrant collection 'uk_1996' containing the Criminal Code articles. "
        "When a user asks a legal question or references an article, FIRST call 'search_uk_1996' with the user's query to retrieve the most relevant article. "
        "If the tool returns documents, choose the most relevant one and include its FULL article text verbatim in your answer. "
        "After the full text, add a concise, neutral summary and commentary to explain the article's meaning and applicability. "
        "Prefer primary sources returned by the tool and cite them inline as [1]. If multiple articles are equally relevant, select the best one and mention any close alternatives. "
        "If no documents are found, say so and suggest a refined query or clarification (e.g., article number, key terms, jurisdiction details). "
        "Your commentary is informational only and not legal advice. Include a brief non-legal-advice disclaimer when appropriate. "
        "Be structured and concise: use short sections with clear headings: 'Полный текст статьи' followed by the verbatim law text, then 'Краткое пояснение' for your summary/comments. "
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
        tools = [search_uk_1996]
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
    print("\n--- Conversation Trace ---\n")
    for msg in response.get("messages", []):
        cls_name = getattr(msg, "__class__", type(msg)).__name__
        if cls_name == "HumanMessage":
            print(f"User: {getattr(msg, 'content', '')}")
        elif cls_name == "AIMessage":
            tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
            if tool_calls:
                print("AI: (decides to use tool)")
                for call in tool_calls:
                    args = call.get("function", {}).get("arguments", "")
                    name = call.get("function", {}).get("name", "")
                    print(f"  Calls tool: {name} with args: {args}")
            elif getattr(msg, "content", None):
                print(f"AI: {getattr(msg, 'content', '')}")
        elif cls_name == "ToolMessage":
            print(f"Tool [{getattr(msg, 'name', '')}]: {getattr(msg, 'content', '')}")
    print("\n--- Final Response ---\n")
    for msg in reversed(response.get("messages", [])):
        cls_name = getattr(msg, "__class__", type(msg)).__name__
        if cls_name == "AIMessage" and getattr(msg, "content", None):
            print(f"AI: {getattr(msg, 'content', '')}")
            break


def answer_question(query: str, thread_id: Optional[str] = None) -> str:
    agent = get_agent()
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": str(thread_id)}}
    if os.getenv("DEBUG_CONVERSATION"):
        print(f"User: {query}")
    response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
    if os.getenv("DEBUG_CONVERSATION"):
        print_conversation(response)
    for msg in reversed(response.get("messages", [])):
        cls_name = getattr(msg, "__class__", type(msg)).__name__
        if cls_name == "AIMessage" and getattr(msg, "content", None):
            ai_text = str(getattr(msg, "content"))
            if os.getenv("DEBUG_CONVERSATION"):
                print(f"AI: {ai_text}")
            return ai_text
    return "I was unable to generate an answer. Please try rephrasing your request."
