from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_core.llm_analyzer import analyze_stock
from ai_core.rag_engine import (
    build_rag_evidence_bundle,
    build_rag_index,
    build_rag_index_payload,
    load_static_knowledge,
    save_retrieval_results,
)


@dataclass
class RagArtifacts:
    context: str
    results: list[dict[str, Any]]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    query: str = ""
    document_count: int = 0
    chunk_count: int = 0
    top_k: int = 5
    query_type: str = "comprehensive"
    source_counts: dict[str, int] = field(default_factory=dict)
    index_path: Path | None = None
    retrieval_path: Path | None = None


@dataclass
class AnalysisArtifacts:
    answer: str
    report_path: Path | None
    rag: RagArtifacts


def build_query(result: dict[str, Any], question: str) -> str:
    return f"{result.get('stock_name', '')} {result.get('news_keyword', '')} {question}".strip()


def build_rag_artifacts(
    rag_documents: list[dict[str, Any]],
    query: str,
    output_root: str | Path | None = None,
    top_k: int = 5,
    max_context_chars: int = 3000,
    stock_id: str = "",
) -> RagArtifacts:
    combined_documents = [*rag_documents, *load_static_knowledge()]
    if output_root:
        root = Path(output_root)
        index_path = root / "rag_index.json"
        index = build_rag_index(combined_documents, index_path)
    else:
        root = None
        index_path = None
        index = build_rag_index_payload(combined_documents)

    bundle = build_rag_evidence_bundle(
        index,
        query,
        top_k=top_k,
        max_context_chars=max_context_chars,
        stock_id=stock_id,
    )
    results = bundle["raw_results"]
    context = bundle["context"]

    retrieval_path = None
    if root:
        retrieval_path = save_retrieval_results(results, root / "rag_retrieval.json")

    return RagArtifacts(
        context=context,
        results=results,
        evidence=bundle["evidence"],
        status=bundle["status"],
        query=query,
        document_count=bundle["document_count"],
        chunk_count=bundle["chunk_count"],
        top_k=bundle["top_k"],
        query_type=bundle["query_type"],
        source_counts=bundle["source_counts"],
        index_path=index_path,
        retrieval_path=retrieval_path,
    )


def analyze_result(
    result: dict[str, Any],
    question: str | None = None,
    prefer_llm: bool = True,
    output_root: str | Path | None = None,
    use_rag: bool = True,
    rag_top_k: int = 5,
    rag_query: str | None = None,
) -> AnalysisArtifacts:
    question = question or f"{result['stock_name']} 最近市場資訊如何？"
    rag = (
        build_rag_artifacts(
            result["data"]["rag_documents"],
            rag_query or build_query(result, question),
            output_root=output_root,
            top_k=rag_top_k,
            stock_id=str(result.get("stock_id", "")),
        )
        if use_rag
        else RagArtifacts(context="尚未建立或檢索 RAG 文件。", results=[], status="disabled")
    )

    answer = analyze_stock(
        result["stock_id"],
        result["stock_name"],
        question,
        result["data"]["price_history"],
        result["data"]["institutional_trading"],
        result["data"]["news"],
        result["data"]["ptt"],
        snapshot=None,
        prefer_llm=prefer_llm,
        rag_context=rag.context,
    )

    report_path = None
    if output_root:
        report_path = Path(output_root) / "analysis_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(answer, encoding="utf-8")

    return AnalysisArtifacts(answer=answer, report_path=report_path, rag=rag)
