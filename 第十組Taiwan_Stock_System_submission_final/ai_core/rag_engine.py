from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120
MAX_CHUNKS_PER_DOCUMENT = 2
DEFAULT_KNOWLEDGE_ROOT = Path("data/knowledge")
SOURCE_TYPE_LABELS = {
    "news": "新聞",
    "ptt": "PTT",
    "glossary": "金融名詞",
    "analysis_rule": "分析規則",
}


def load_rag_documents(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    documents: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return documents


def load_static_knowledge(root: str | Path = DEFAULT_KNOWLEDGE_ROOT) -> list[dict[str, Any]]:
    knowledge_root = Path(root)
    documents: list[dict[str, Any]] = []

    glossary = _read_json_list(knowledge_root / "finance_glossary.json")
    for item in glossary:
        term = _clean_text(str(item.get("term", "")))
        if not term:
            continue
        definition = _clean_text(str(item.get("definition", "")))
        interpretation = _clean_text(str(item.get("interpretation", "")))
        risk_note = _clean_text(str(item.get("risk_note", "")))
        content = (
            f"名詞：{term}。定義：{definition}。"
            f"解讀：{interpretation}。解讀限制：{risk_note}"
        )
        documents.append(
            {
                "document_id": f"glossary:{term}",
                "source_type": "glossary",
                "stock_id": "",
                "stock_name": "",
                "title": term,
                "source": "金融名詞庫",
                "published_at": "",
                "url": "",
                "crawl_time": "",
                "category": item.get("category", "金融名詞"),
                "content": content,
                "text": content,
            }
        )

    rules = _read_json_list(knowledge_root / "analysis_rules.json")
    for item in rules:
        title = _clean_text(str(item.get("title", "")))
        rule = _clean_text(str(item.get("rule", "")))
        if not title or not rule:
            continue
        documents.append(
            {
                "document_id": f"analysis_rule:{item.get('rule_id', title)}",
                "source_type": "analysis_rule",
                "stock_id": "",
                "stock_name": "",
                "title": title,
                "source": "系統分析規則",
                "published_at": "",
                "url": "",
                "crawl_time": "",
                "category": item.get("category", "分析規則"),
                "content": rule,
                "text": f"{title}。{rule}",
            }
        )
    return documents


def build_rag_index(
    documents: list[dict[str, Any]],
    output_path: str | Path,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Build a small local TF-IDF index from crawler RAG documents."""
    payload = build_rag_index_payload(documents, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_rag_index_payload(
    documents: list[dict[str, Any]],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Build an in-memory index payload without writing files."""
    prepared_docs: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()

    unique_documents = _dedupe_documents(documents)
    for document_id, document in enumerate(unique_documents):
        text = _clean_text(str(document.get("content") or document.get("text", "")))
        if not text:
            continue
        chunks = _chunk_text(text, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)[:MAX_CHUNKS_PER_DOCUMENT]
        source_type = _normalize_source_type(document)
        stable_document_id = str(document.get("document_id") or f"document:{document_id}")
        for chunk_index, chunk in enumerate(chunks):
            tokens = _tokenize(chunk)
            if not tokens:
                continue
            term_counts = Counter(tokens)
            document_frequency.update(term_counts.keys())
            prepared_docs.append(
                {
                    "id": f"{document_id}:{chunk_index}",
                    "document_id": stable_document_id,
                    "reference_id": str(document.get("reference_id") or f"D{document_id + 1}C{chunk_index + 1}"),
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "source_type": source_type,
                    "stock_id": str(document.get("stock_id", "")),
                    "stock_name": str(document.get("stock_name", "")),
                    "source": document.get("source", ""),
                    "title": document.get("title", ""),
                    "url": document.get("url", ""),
                    "published_at": document.get("published_at", ""),
                    "crawl_time": document.get("crawl_time", ""),
                    "category": document.get("category", ""),
                    "content": chunk,
                    "sentiment": document.get("sentiment", ""),
                    "text": chunk,
                    "term_counts": dict(term_counts),
                    "length": len(tokens),
                }
            )

    total_docs = len(prepared_docs)
    idf = {
        term: math.log((1 + total_docs) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "document_count": len(unique_documents),
        "chunk_count": total_docs,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "idf": idf,
        "documents": prepared_docs,
    }
    return payload


def load_rag_index(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"document_count": 0, "idf": {}, "documents": []}
    return json.loads(source.read_text(encoding="utf-8"))


def search_rag_index(
    index: dict[str, Any],
    query: str,
    top_k: int = 5,
    diversify_documents: bool = True,
    stock_id: str = "",
    query_type: str | None = None,
) -> list[dict[str, Any]]:
    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return []

    resolved_query_type = query_type or detect_query_type(query)
    preferred_types = _preferred_source_types(resolved_query_type)
    requested_stock_id = stock_id or _extract_stock_id(query)
    idf = index.get("idf", {})
    scored: list[dict[str, Any]] = []
    for document in index.get("documents", []):
        source_type = str(document.get("source_type", ""))
        document_stock_id = str(document.get("stock_id", ""))
        if source_type in {"news", "ptt"} and requested_stock_id and document_stock_id and document_stock_id != requested_stock_id:
            continue
        if preferred_types and source_type not in preferred_types:
            continue

        term_counts = document.get("term_counts", {})
        length = max(int(document.get("length", 1)), 1)
        score = 0.0
        for term, query_count in query_terms.items():
            tf = float(term_counts.get(term, 0)) / length
            score += tf * float(idf.get(term, 0.0)) * query_count
        if score <= 0:
            continue
        if source_type in {"news", "ptt"}:
            score += _recency_boost(str(document.get("published_at", "")))
        result = {key: value for key, value in document.items() if key not in {"term_counts", "length"}}
        result["score"] = round(score, 6)
        scored.append(result)

    scored.sort(key=lambda item: item["score"], reverse=True)
    top_k = max(top_k, 1)
    if not diversify_documents:
        return scored[:top_k]

    selected: list[dict[str, Any]] = []
    seen_documents: set[Any] = set()
    for item in scored:
        document_id = item.get("document_id", item.get("id"))
        if document_id in seen_documents:
            continue
        selected.append(item)
        seen_documents.add(document_id)
        if len(selected) >= top_k:
            return selected
    return selected


def detect_query_type(query: str) -> str:
    text = _clean_text(query).lower()
    if any(keyword.lower() in text for keyword in ("ptt", "社群", "討論", "鄉民", "市場情緒")):
        return "ptt"
    if any(keyword in text for keyword in ("是什麼", "什麼意思", "代表什麼", "名詞", "如何解讀", "怎麼解讀")):
        return "glossary"
    if any(keyword in text for keyword in ("新聞", "消息", "事件", "有哪些重要", "近期事件")):
        return "news"
    return "comprehensive"


def build_rag_context(
    results: list[dict[str, Any]],
    max_chars: int = 3000,
) -> str:
    return build_referenced_rag_context(format_rag_evidence(results), max_chars=max_chars)


def format_rag_evidence(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        text = _clean_text(str(result.get("text", "")))
        if not text:
            continue
        evidence.append(
            {
                "rank": rank,
                "reference_id": f"R{rank}",
                "score": result.get("score", 0),
                "source_type": result.get("source_type", ""),
                "stock_id": result.get("stock_id", ""),
                "stock_name": result.get("stock_name", ""),
                "source": result.get("source", ""),
                "title": result.get("title", ""),
                "published_at": result.get("published_at", ""),
                "url": result.get("url", ""),
                "category": result.get("category", ""),
                "content": text,
            }
        )
    return evidence


def build_referenced_rag_context(
    evidence: list[dict[str, Any]],
    max_chars: int = 3000,
) -> str:
    if not evidence:
        return "目前檢索證據不足，未找到可引用的 RAG 文件。"

    blocks: list[str] = []
    used_chars = 0
    for item in evidence:
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        content = _clean_text(str(item.get("content", "")))
        excerpt = content[: min(remaining, 650)].rstrip()
        source_type = str(item.get("source_type", ""))
        label = SOURCE_TYPE_LABELS.get(source_type, source_type or "資料")
        stock_text = " ".join(
            part for part in (str(item.get("stock_id", "")), str(item.get("stock_name", ""))) if part
        )
        block = (
            f"[{item.get('reference_id', '')}｜{label}]\n"
            f"股票：{stock_text or '不適用'}\n"
            f"標題：{item.get('title', '')}\n"
            f"來源：{item.get('source', '')}\n"
            f"分類：{item.get('category', '')}\n"
            f"日期：{item.get('published_at', '') or '未提供'}\n"
            f"URL：{item.get('url', '') or '無'}\n"
            f"檢索分數：{item.get('score', 0)}\n"
            f"內容：{excerpt}"
        )
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks) or "目前檢索證據不足，未找到可引用的 RAG 文件。"


def build_rag_evidence_bundle(
    index: dict[str, Any],
    query: str,
    top_k: int = 5,
    max_context_chars: int = 3000,
    stock_id: str = "",
) -> dict[str, Any]:
    query_type = detect_query_type(query)
    results = search_rag_index(
        index,
        query,
        top_k=top_k,
        stock_id=stock_id,
        query_type=query_type,
    )
    evidence = format_rag_evidence(results)
    if evidence:
        context = build_referenced_rag_context(evidence, max_chars=max_context_chars)
    else:
        context = _insufficient_evidence_message(query_type)
    source_counts = Counter(item.get("source_type", "") for item in evidence)
    return {
        "status": "ok" if evidence else "insufficient_evidence",
        "query": query,
        "query_type": query_type,
        "top_k": max(top_k, 1),
        "document_count": int(index.get("document_count", 0) or 0),
        "chunk_count": len(index.get("documents", [])),
        "evidence_count": len(evidence),
        "source_counts": dict(source_counts),
        "raw_results": results,
        "evidence": evidence,
        "context": context,
    }


def save_retrieval_results(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _tokenize(text: str) -> list[str]:
    text = _clean_text(text).lower()
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            tokens.extend(match)
            tokens.extend(match[i : i + 2] for i in range(max(len(match) - 1, 0)))
        else:
            tokens.append(match)
    return [token for token in tokens if token.strip()]


def _clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "")
    return value.strip()


def _chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    chunk_chars = max(chunk_chars, 200)
    chunk_overlap = min(max(chunk_overlap, 0), chunk_chars // 2)
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind("；", start, end), text.rfind("，", start, end))
            if boundary > start + chunk_chars * 0.55:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _dedupe_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for document in documents:
        key = _document_key(document)
        if not key:
            continue
        if key not in unique or _document_quality(document) > _document_quality(unique[key]):
            unique[key] = document
    return list(unique.values())


def _document_key(document: dict[str, Any]) -> str:
    document_id = _clean_text(str(document.get("document_id", "")))
    if document_id:
        return f"document_id:{document_id}"
    url = _clean_text(str(document.get("url", "")))
    if url:
        return f"url:{url}"
    title = _clean_text(str(document.get("title", ""))).lower()
    title = re.sub(r"\s+", "", title)
    if title:
        return f"title:{title}"
    text = _clean_text(str(document.get("text", "")))
    return f"text:{text[:80]}" if text else ""


def _document_quality(document: dict[str, Any]) -> int:
    return (20 if document.get("url") else 0) + min(len(str(document.get("text", ""))), 10000)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _normalize_source_type(document: dict[str, Any]) -> str:
    source_type = _clean_text(str(document.get("source_type", ""))).lower()
    if source_type:
        return source_type
    source = _clean_text(str(document.get("source", ""))).lower()
    return "ptt" if "ptt" in source else "news"


def _preferred_source_types(query_type: str) -> set[str]:
    if query_type == "news":
        return {"news"}
    if query_type == "ptt":
        return {"ptt"}
    if query_type == "glossary":
        return {"glossary", "analysis_rule"}
    return set()


def _extract_stock_id(query: str) -> str:
    match = re.search(r"\b\d{4,6}\b", query)
    return match.group(0) if match else ""


def _recency_boost(value: str) -> float:
    try:
        published = datetime.strptime(value[:10], "%Y-%m-%d")
    except Exception:
        return 0.0
    age_days = max((datetime.now() - published).days, 0)
    if age_days <= 30:
        return 0.04
    if age_days <= 90:
        return 0.02
    if age_days <= 180:
        return 0.01
    return 0.0


def _insufficient_evidence_message(query_type: str) -> str:
    if query_type in {"news", "ptt"}:
        return "目前沒有取得足夠的近期新聞或討論證據。"
    if query_type == "glossary":
        return "目前沒有取得足夠的金融名詞或分析規則證據。"
    return "目前檢索證據不足，未找到可引用的 RAG 文件。"
