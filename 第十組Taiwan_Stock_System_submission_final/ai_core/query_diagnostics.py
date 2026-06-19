from __future__ import annotations

import re
from typing import Any


SOURCE_TYPES = ("news", "ptt", "glossary", "analysis_rule")


def build_query_diagnostics(
    *,
    result: dict[str, Any],
    rewritten: dict[str, Any],
    rag: Any,
    answer: str,
    memory_enabled: bool,
    openai_key_present: bool,
    gemini_key_present: bool,
) -> dict[str, Any]:
    evidence = list(getattr(rag, "evidence", []) or [])
    scores = [_safe_float(item.get("score")) for item in evidence]
    scores = [score for score in scores if score is not None]
    source_counts = {source_type: 0 for source_type in SOURCE_TYPES}
    for item in evidence:
        source_type = str(item.get("source_type", ""))
        if source_type in source_counts:
            source_counts[source_type] += 1

    stock_id = str(result.get("stock_id", ""))
    stock_name = str(result.get("stock_name", ""))
    query_type = str(getattr(rag, "query_type", "comprehensive"))
    focus = _diagnostic_focus(rewritten, query_type)
    original_query = str(rewritten.get("original_query", ""))
    rewritten_query = str(rewritten.get("rewritten_query", original_query))
    used_memory = bool(rewritten.get("used_memory", False))
    rewrite_executed = used_memory or _normalize(original_query) != _normalize(rewritten_query)
    rag_status = str(getattr(rag, "status", "insufficient_evidence"))
    insufficient = rag_status != "ok" or not evidence

    different_stock_documents = [
        item
        for item in evidence
        if item.get("source_type") in {"news", "ptt"}
        and item.get("stock_id")
        and str(item.get("stock_id")) != stock_id
    ]

    api_key_present = bool(openai_key_present or gemini_key_present)
    answer_mode = _detect_answer_mode(
        answer,
        openai_key_present=openai_key_present,
        gemini_key_present=gemini_key_present,
    )
    references = sorted(set(re.findall(r"\[R(\d+)(?:\]|｜)", answer or "")), key=int)
    has_limitations = "資料限制" in (answer or "") or "資料不足" in (answer or "")
    advice_rule_triggered = any(
        phrase in (answer or "")
        for phrase in ("不構成投資建議", "不提供投資建議", "不能直接視為買進訊號", "不等於投資建議")
    )
    prediction_rule_triggered = any(
        phrase in (answer or "")
        for phrase in ("不預測確切未來股價", "不預測", "確切未來股價", "不得預測")
    )
    counts = result.get("counts", {}) if isinstance(result.get("counts"), dict) else {}

    summary = _build_summary(
        stock_id=stock_id,
        stock_name=stock_name,
        used_memory=used_memory,
        evidence_count=len(evidence),
        reference_count=len(references),
        answer_mode=answer_mode,
        insufficient=insufficient,
    )

    return {
        "original_query": original_query,
        "stock_id": stock_id,
        "stock_name": stock_name,
        "focus": focus,
        "memory_enabled": bool(memory_enabled),
        "used_memory": used_memory,
        "query_rewriting_executed": rewrite_executed,
        "rewritten_query": rewritten_query,
        "rag_query": str(rewritten.get("rag_query", getattr(rag, "query", ""))),
        "rag_success": not insufficient,
        "evidence_count": len(evidence),
        "top_k": int(getattr(rag, "top_k", 0) or 0),
        "max_score": max(scores) if scores else None,
        "average_score": sum(scores) / len(scores) if scores else None,
        "source_counts": source_counts,
        "different_stock_documents": len(different_stock_documents),
        "insufficient_evidence": insufficient,
        "answer_mode": answer_mode,
        "api_key_present": api_key_present,
        "has_references": bool(references),
        "reference_count": len(references),
        "has_limitations": has_limitations,
        "advice_rule_triggered": advice_rule_triggered,
        "prediction_rule_triggered": prediction_rule_triggered,
        "price_rows": counts.get("price_rows", _data_length(result, "price_history")),
        "chip_rows": counts.get("chip_rows", _data_length(result, "institutional_trading")),
        "news_items": counts.get("news_items", _data_length(result, "news")),
        "ptt_items": counts.get("ptt_items", _data_length(result, "ptt")),
        "data_mode": result.get("data_mode") or "未提供",
        "updated_at": result.get("crawled_at") or "未提供",
        "summary": summary,
    }


def _diagnostic_focus(rewritten: dict[str, Any], query_type: str) -> str:
    if query_type == "news":
        return "新聞"
    if query_type == "ptt":
        return "PTT"
    if query_type == "glossary":
        return "金融名詞"
    detected = str(rewritten.get("detected_focus", "整體"))
    return "綜合市場" if detected in {"", "整體"} else detected


def _detect_answer_mode(answer: str, *, openai_key_present: bool, gemini_key_present: bool) -> str:
    fallback_markers = (
        "此為資訊整理與教學展示",
        "名詞解釋不等於投資建議",
        "本系統尚未直接串接完整財報資料庫",
        "規則式摘要",
    )
    if not openai_key_present and not gemini_key_present:
        return "fallback"
    if any(marker in (answer or "") for marker in fallback_markers):
        return "fallback"
    if openai_key_present:
        return "OpenAI"
    if gemini_key_present:
        return "Gemini"
    return "fallback"


def _build_summary(
    *,
    stock_id: str,
    stock_name: str,
    used_memory: bool,
    evidence_count: int,
    reference_count: int,
    answer_mode: str,
    insufficient: bool,
) -> str:
    stock_text = " ".join(part for part in (stock_id, stock_name) if part) or "未辨識股票"
    if insufficient:
        return f"本輪辨識 {stock_text}，但未取得足夠相關證據，系統已採保守回答，目前使用 {answer_mode} 模式。"
    memory_text = "使用 Memory 補全追問" if used_memory else "未使用 Memory 補全"
    return (
        f"本輪成功辨識 {stock_text}，{memory_text}，共取得 {evidence_count} 筆 RAG 證據，"
        f"回答引用 {reference_count} 筆來源，目前使用 {answer_mode} 模式。"
    )


def _data_length(result: dict[str, Any], key: str) -> int | str:
    try:
        return len(result.get("data", {}).get(key, []))
    except Exception:
        return "未提供"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
