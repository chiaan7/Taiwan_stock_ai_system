from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd

from ai_core.analysis_pipeline import analyze_result
from ai_core.query_diagnostics import build_query_diagnostics
from ai_core.query_rewriter import rewrite_query
from ai_core.rag_engine import (
    build_rag_evidence_bundle,
    build_rag_index_payload,
    load_static_knowledge,
    search_rag_index,
)
from scrapers.crawler import EMPTY_PRICE_COLUMNS, _choose_dataframe_source, _valid_price_data
from scrapers.news_spider import NewsItem, build_rag_documents


def main() -> None:
    news = [
        NewsItem(
            title="台積電 AI 需求帶動先進製程",
            url="https://example.com/tsmc-ai",
            source="Test News",
            published_at="2026-05-24",
            sentiment="正向",
            content="台積電受惠 AI 晶片需求，市場關注先進製程與產能規劃。",
            keyword="台積電",
        )
    ]
    ptt = [
        NewsItem(
            title="[討論] 台積電近期量價與法人動向",
            url="https://www.ptt.cc/bbs/Stock/test.html",
            source="PTT Stock",
            published_at="2026-05-24",
            sentiment="中立",
            content="PTT 討論台積電近期股價、外資買賣超與先進製程消息，但看法並不一致。",
            keyword="台積電",
            stock_id="2330",
        )
    ]
    history = pd.DataFrame(
        [
            {"date": "2026-05-20", "close": 1000, "volume": 1000},
            {"date": "2026-05-21", "close": 1010, "volume": 1100},
            {"date": "2026-05-22", "close": 1020, "volume": 1200},
            {"date": "2026-05-23", "close": 1015, "volume": 1050},
            {"date": "2026-05-24", "close": 1030, "volume": 1300},
        ]
    )
    chip_data = pd.DataFrame(
        [
            {"date": "2026-05-20", "foreign_net": 100, "investment_trust_net": 20, "dealer_net": -10, "total_net": 110},
            {"date": "2026-05-21", "foreign_net": 200, "investment_trust_net": 30, "dealer_net": 10, "total_net": 240},
            {"date": "2026-05-22", "foreign_net": -50, "investment_trust_net": 10, "dealer_net": 5, "total_net": -35},
            {"date": "2026-05-23", "foreign_net": 150, "investment_trust_net": 20, "dealer_net": 0, "total_net": 170},
            {"date": "2026-05-24", "foreign_net": 80, "investment_trust_net": 15, "dealer_net": 5, "total_net": 100},
        ]
    )
    dynamic_documents = build_rag_documents("2330", "台積電", news, ptt)
    result = {
        "stock_id": "2330",
        "stock_name": "台積電",
        "news_keyword": "台積電",
        "data_mode": "示範資料",
        "crawled_at": "2026-05-24T12:00:00",
        "counts": {
            "price_rows": len(history),
            "chip_rows": len(chip_data),
            "news_items": len(news),
            "ptt_items": len(ptt),
            "rag_documents": len(dynamic_documents),
        },
        "data": {
            "price_history": history,
            "institutional_trading": chip_data,
            "news": news,
            "ptt": ptt,
            "rag_documents": dynamic_documents,
        },
    }
    index = build_rag_index_payload([*dynamic_documents, *load_static_knowledge()])
    retrieval = search_rag_index(index, "台積電 AI 先進製程", top_k=3)
    assert retrieval, "RAG retrieval should return at least one document"
    evidence_bundle = build_rag_evidence_bundle(index, "台積電 AI 先進製程", top_k=3)
    assert evidence_bundle["evidence"], "RAG evidence should be formatted"
    assert evidence_bundle["evidence"][0]["reference_id"] == "R1", "Evidence should use R-style reference ids"

    glossary_bundle = build_rag_evidence_bundle(index, "外資買超是什麼意思？", top_k=3)
    assert glossary_bundle["evidence"], "Glossary question should retrieve static knowledge"
    assert glossary_bundle["evidence"][0]["source_type"] in {"glossary", "analysis_rule"}
    glossary_answer = analyze_result(result, "外資買超是什麼意思？", prefer_llm=False)
    assert "買超" in glossary_answer.answer
    assert "不代表股價一定上漲" in glossary_answer.answer
    glossary_rewrite = rewrite_query(
        "外資買超是什麼意思？",
        {"enabled": False},
        fallback_stock_id="2330",
        fallback_stock_name="台積電",
    )
    glossary_diagnostics = build_query_diagnostics(
        result=result,
        rewritten=glossary_rewrite,
        rag=glossary_answer.rag,
        answer=glossary_answer.answer,
        memory_enabled=False,
        openai_key_present=False,
        gemini_key_present=False,
    )
    assert glossary_diagnostics["focus"] == "金融名詞"
    assert glossary_diagnostics["source_counts"]["glossary"] + glossary_diagnostics["source_counts"]["analysis_rule"] > 0
    assert glossary_diagnostics["answer_mode"] == "fallback"
    assert "AIza" not in str(glossary_diagnostics) and "sk-" not in str(glossary_diagnostics)

    news_bundle = build_rag_evidence_bundle(index, "2330 台積電近期有哪些重要新聞？", top_k=3, stock_id="2330")
    assert news_bundle["evidence"], "News question should retrieve news"
    assert all(item["source_type"] == "news" for item in news_bundle["evidence"])
    assert all(item["stock_id"] == "2330" and item["title"] and item["published_at"] for item in news_bundle["evidence"])

    ptt_bundle = build_rag_evidence_bundle(index, "PTT 最近在討論台積電什麼？", top_k=3, stock_id="2330")
    assert ptt_bundle["evidence"], "PTT question should retrieve PTT posts"
    assert all(item["source_type"] == "ptt" for item in ptt_bundle["evidence"])

    comprehensive = analyze_result(result, "台積電最近的市場狀況如何？", prefer_llm=False)
    assert "一、數據事實" in comprehensive.answer
    assert "五、資料限制" in comprehensive.answer
    complete_rewrite = rewrite_query(
        "台積電最近的市場狀況如何？",
        {"enabled": True},
        fallback_stock_id="2330",
        fallback_stock_name="台積電",
    )
    complete_diagnostics = build_query_diagnostics(
        result=result,
        rewritten=complete_rewrite,
        rag=comprehensive.rag,
        answer=comprehensive.answer,
        memory_enabled=True,
        openai_key_present=False,
        gemini_key_present=False,
    )
    assert complete_diagnostics["stock_id"] == "2330"
    assert complete_diagnostics["stock_name"] == "台積電"
    assert complete_diagnostics["evidence_count"] > 0
    assert sum(complete_diagnostics["source_counts"].values()) > 0

    empty_bundle = build_rag_evidence_bundle(
        build_rag_index_payload([]),
        "2330 台積電近期有哪些重要新聞？",
        top_k=3,
        stock_id="2330",
    )
    assert empty_bundle["status"] == "insufficient_evidence"
    assert "目前沒有取得足夠的近期新聞或討論證據" in empty_bundle["context"]
    insufficient_diagnostics = build_query_diagnostics(
        result=result,
        rewritten=complete_rewrite,
        rag=SimpleNamespace(
            evidence=[],
            status="insufficient_evidence",
            top_k=3,
            query_type="news",
            query=complete_rewrite["rag_query"],
        ),
        answer="目前資料不足。五、資料限制：沒有足夠證據。",
        memory_enabled=True,
        openai_key_present=False,
        gemini_key_present=False,
    )
    assert insufficient_diagnostics["insufficient_evidence"]
    assert not insufficient_diagnostics["rag_success"]

    memory = {
        "enabled": True,
        "current_stock_id": "2330",
        "current_stock_name": "台積電",
        "current_focus": "整體",
    }
    rewritten = rewrite_query("那外資呢？", memory)
    assert rewritten["used_memory"], "Follow-up question should use memory"
    assert "2330 台積電" in rewritten["rewritten_query"], "Rewritten query should include current stock"
    followup_diagnostics = build_query_diagnostics(
        result=result,
        rewritten=rewritten,
        rag=comprehensive.rag,
        answer=comprehensive.answer,
        memory_enabled=True,
        openai_key_present=False,
        gemini_key_present=False,
    )
    assert followup_diagnostics["used_memory"]
    assert followup_diagnostics["query_rewriting_executed"]
    assert "2330 台積電" in followup_diagnostics["rewritten_query"]
    assert "外資" in followup_diagnostics["rewritten_query"]
    disabled_rewrite = rewrite_query("那外資呢？", {**memory, "enabled": False})
    assert not disabled_rewrite["used_memory"], "Disabled memory should not be used"

    root = Path(".tmp_smoke_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        cache_path = root / "price_history.csv"
        cached_price = pd.DataFrame(
            [{"date": "2026-05-24", "open": 100, "high": 110, "low": 99, "close": 108, "adjclose": 108, "volume": 1000}]
        )
        cached_price.to_csv(cache_path, index=False)
        chosen_price, source_status = _choose_dataframe_source(
            "price",
            pd.DataFrame(columns=EMPTY_PRICE_COLUMNS),
            cache_path,
            root / "missing_demo.csv",
            EMPTY_PRICE_COLUMNS,
            _valid_price_data,
            "simulated price failure",
            allow_demo=False,
        )
        assert len(chosen_price) == 1, "Failed source should fall back to existing cache"
        assert source_status["used_cache"], "Source status should indicate cache fallback"
    finally:
        shutil.rmtree(root, ignore_errors=True)

    analysis = analyze_result(result, "台積電最近狀況如何？", prefer_llm=False)
    assert "資料限制" in analysis.answer, "Analysis should include limitation section"
    assert analysis.rag.evidence, "Analysis should expose RAG evidence"
    print("offline smoke test passed")


if __name__ == "__main__":
    main()
