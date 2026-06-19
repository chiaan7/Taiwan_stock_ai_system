from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit is an app dependency
    st = None


MEMORY_KEY = "taiwan_stock_assistant_memory"
_FALLBACK_STATE: dict[str, Any] = {}


def _default_memory(enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "messages": [],
        "current_stock_id": "",
        "current_stock_name": "",
        "current_focus": "",
        "previous_question": "",
        "previous_answer_summary": "",
        "referenced_sources": [],
    }


def _state() -> Any:
    if st is not None:
        try:
            return st.session_state
        except Exception:
            pass
    return _FALLBACK_STATE


def initialize_memory(enabled: bool = True) -> dict[str, Any]:
    state = _state()
    if MEMORY_KEY not in state:
        state[MEMORY_KEY] = _default_memory(enabled=enabled)
    memory = state[MEMORY_KEY]
    defaults = _default_memory(enabled=enabled)
    for key, value in defaults.items():
        memory.setdefault(key, value)
    return memory


def get_memory() -> dict[str, Any]:
    return deepcopy(initialize_memory())


def clear_memory() -> dict[str, Any]:
    state = _state()
    enabled = bool(initialize_memory().get("enabled", True))
    state[MEMORY_KEY] = _default_memory(enabled=enabled)
    return get_memory()


def is_memory_enabled() -> bool:
    return bool(initialize_memory().get("enabled", True))


def set_memory_enabled(enabled: bool) -> dict[str, Any]:
    memory = initialize_memory(enabled=enabled)
    memory["enabled"] = bool(enabled)
    return get_memory()


def update_memory(
    *,
    user_message: str | None = None,
    assistant_message: str | None = None,
    stock_id: str | None = None,
    stock_name: str | None = None,
    focus: str | None = None,
    answer_summary: str | None = None,
    referenced_sources: list[str] | None = None,
) -> dict[str, Any]:
    memory = initialize_memory()
    if not memory.get("enabled", True):
        return get_memory()

    if user_message:
        memory["messages"].append({"role": "user", "content": user_message})
        memory["previous_question"] = user_message
    if assistant_message:
        memory["messages"].append({"role": "assistant", "content": assistant_message})
        memory["previous_answer_summary"] = answer_summary or summarize_answer(assistant_message)
    if stock_id:
        memory["current_stock_id"] = str(stock_id)
    if stock_name:
        memory["current_stock_name"] = str(stock_name)
    if focus:
        memory["current_focus"] = str(focus)
    if referenced_sources is not None:
        memory["referenced_sources"] = list(referenced_sources)

    memory["messages"] = memory["messages"][-12:]
    return get_memory()


def summarize_answer(answer: str, max_chars: int = 220) -> str:
    text = " ".join((answer or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
