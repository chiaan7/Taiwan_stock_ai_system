from __future__ import annotations

import re
from typing import Any


CONTINUATION_MARKERS = (
    "那",
    "剛才",
    "剛剛",
    "前面",
    "上述",
    "這個",
    "這些",
    "它",
    "他",
    "呢",
    "風險",
)

FOCUS_KEYWORDS = (
    ("外資", ("外資", "foreign")),
    ("投信", ("投信",)),
    ("自營商", ("自營商",)),
    ("籌碼", ("籌碼", "法人", "買賣超", "三大法人")),
    ("新聞風險", ("新聞", "風險", "利空", "疑慮")),
    ("情緒", ("情緒", "PTT", "社群", "討論")),
    ("股價", ("股價", "價格", "走勢", "漲跌", "收盤")),
)


def rewrite_query(
    original_query: str,
    memory: dict[str, Any] | None = None,
    *,
    fallback_stock_id: str = "",
    fallback_stock_name: str = "",
) -> dict[str, Any]:
    memory = memory or {}
    query = _normalize(original_query)
    detected_focus = detect_focus(query) or str(memory.get("current_focus") or "整體")
    memory_enabled = bool(memory.get("enabled", True))

    stock_id = _extract_stock_id(query) or fallback_stock_id
    stock_name = fallback_stock_name
    used_memory = False

    if memory_enabled and is_continuation_query(query):
        memory_stock_id = str(memory.get("current_stock_id") or "")
        memory_stock_name = str(memory.get("current_stock_name") or "")
        if memory_stock_id or memory_stock_name:
            used_memory = True
        if not stock_id and memory_stock_id:
            stock_id = memory_stock_id
        if not stock_name and memory_stock_name:
            stock_name = memory_stock_name
        if detected_focus == "整體" and memory.get("current_focus"):
            detected_focus = str(memory["current_focus"])

    rewritten_query = _build_rewritten_query(query, stock_id, stock_name, detected_focus, used_memory)
    rag_query = _build_rag_query(rewritten_query, stock_id, stock_name, detected_focus)

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "rag_query": rag_query,
        "used_memory": used_memory,
        "detected_focus": detected_focus,
    }


def is_continuation_query(query: str) -> bool:
    text = _normalize(query)
    if not text:
        return False
    if _extract_stock_id(text):
        return False
    return any(marker in text for marker in CONTINUATION_MARKERS) or len(text) <= 8


def detect_focus(query: str) -> str:
    text = _normalize(query)
    for focus, keywords in FOCUS_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return focus
    return "整體"


def _build_rewritten_query(
    query: str,
    stock_id: str,
    stock_name: str,
    focus: str,
    used_memory: bool,
) -> str:
    subject = " ".join(part for part in (stock_id, stock_name) if part).strip()
    if not used_memory or not subject:
        return query

    templates = {
        "外資": f"{subject} 近期外資買賣超狀況如何？",
        "投信": f"{subject} 近期投信買賣超狀況如何？",
        "自營商": f"{subject} 近期自營商買賣超狀況如何？",
        "籌碼": f"{subject} 近期三大法人籌碼狀況如何？",
        "新聞風險": f"{subject} 近期新聞中有哪些可能風險？",
        "情緒": f"{subject} 近期新聞與 PTT 情緒如何？",
        "股價": f"{subject} 近期股價走勢如何？",
    }
    return templates.get(focus, f"{subject} {query}")


def _build_rag_query(rewritten_query: str, stock_id: str, stock_name: str, focus: str) -> str:
    focus_terms = {
        "外資": "外資 買賣超 三大法人 籌碼",
        "投信": "投信 買賣超 三大法人 籌碼",
        "自營商": "自營商 買賣超 三大法人 籌碼",
        "籌碼": "三大法人 外資 投信 自營商 買賣超",
        "新聞風險": "新聞 風險 疑慮 利空 事件",
        "情緒": "新聞 PTT 情緒 討論 市場看法",
        "股價": "股價 走勢 收盤 漲跌 成交量",
    }
    subject = " ".join(part for part in (stock_id, stock_name) if part).strip()
    parts = []
    if subject and subject not in rewritten_query:
        parts.append(subject)
    parts.extend([rewritten_query, focus_terms.get(focus, "市場 資訊")])
    return " ".join(part for part in parts if part).strip()


def _extract_stock_id(query: str) -> str:
    match = re.search(r"\b\d{4,6}\b", query)
    return match.group(0) if match else ""


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
