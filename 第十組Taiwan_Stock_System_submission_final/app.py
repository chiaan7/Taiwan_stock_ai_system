from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

try:
    import streamlit as st
except ImportError:
    print("請先安裝套件：pip install -r requirements.txt")
    print("安裝後執行：streamlit run app.py")
    raise SystemExit(1)

import pandas as pd
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ai_core.analysis_pipeline import analyze_result
from ai_core.api_health import check_gemini_api, get_gemini_api_key
from ai_core.memory_manager import (
    clear_memory,
    get_memory,
    is_memory_enabled,
    set_memory_enabled,
    update_memory,
)
from ai_core.query_rewriter import rewrite_query
from ai_core.query_diagnostics import build_query_diagnostics
from scrapers.crawler import crawl_stock
from scrapers.news_spider import NewsItem, summarize_sentiment
from scrapers.yfinance_client import (
    STOCK_NAMES,
    build_snapshot,
    get_stock_name,
    normalize_stock_id,
    summarize_price,
)


DATA_ROOT = Path("data/raw_data")
DEMO_ROOT = Path("data/demo")
DEMO_STOCK_ID = "2330"
MIN_RAG_DOCUMENTS = 10
EMPTY_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adjclose", "volume"]
EMPTY_CHIP_COLUMNS = ["date", "foreign_net", "investment_trust_net", "dealer_net", "total_net"]
INSTITUTION_COLUMNS = [
    ("foreign_net", "外資"),
    ("investment_trust_net", "投信"),
    ("dealer_net", "自營商"),
]


if load_dotenv:
    load_dotenv()


st.set_page_config(page_title="台股市場資訊儀表板", layout="wide")


def main() -> None:
    inject_custom_css()
    render_header()
    config = render_sidebar()
    apply_runtime_api_key(config["runtime_api_key"])

    if config["query_clicked"] or config["refresh_clicked"]:
        st.session_state["active_dashboard_config"] = config.copy()
    elif st.session_state.get("active_dashboard_config"):
        saved_config = st.session_state["active_dashboard_config"].copy()
        saved_config["query_clicked"] = True
        saved_config["refresh_clicked"] = False
        saved_config["runtime_api_key"] = config["runtime_api_key"]
        config = saved_config
    else:
        st.info("輸入股票代號後按「查詢」。錄影展示建議先使用已有快取資料，例如 2330 台積電。")
        render_system_notes()
        return

    stock_id = normalize_stock_id(config["stock_id"])
    if not is_valid_stock_id(stock_id):
        st.error("股票代號格式不正確。請輸入 4 到 6 位數字，例如 2330。")
        return

    stock_name = config["stock_name"] or get_stock_name(stock_id)
    result: dict[str, Any] | None = None
    status_messages: list[str] = []

    if config["refresh_clicked"]:
        show_crawl_plan(config)
        status_messages.append("正在更新資料")
        with st.spinner("正在更新資料，請稍候..."):
            result = run_crawler_safely(stock_id, stock_name, config)
        if result:
            result["data_mode"] = "即時爬取"
            st.success("資料已更新，以下使用最新資料呈現。")
        else:
            st.warning("網路或資料來源暫時不穩，已改為嘗試使用既有快取或示範資料。")
            result = load_best_available_result(stock_id)
    else:
        cache_status = inspect_cache(stock_id)
        if cache_status["usable"]:
            status_messages.append(f"使用{cache_status['mode']}")
            result = load_result_by_status(cache_status)
            if cache_status["mode"] == "示範資料" and stock_id != DEMO_STOCK_ID:
                st.info("查詢股票目前沒有足夠快取資料，畫面先使用 2330 台積電示範資料，方便確認系統展示效果。")
        else:
            status_messages.append("快取不足，準備爬蟲")
            st.warning("快取與示範資料都不足。為了避免錄影時中斷，系統不會自動爬蟲；請按左側「重新抓取資料」。")
            render_cache_summary(stock_id, cache_status)
            render_system_notes()
            return

    if not result:
        st.error("目前沒有可用資料。請確認股票代號，或稍後按「重新抓取資料」。")
        return

    prefer_llm = has_any_llm_key()
    if not prefer_llm:
        status_messages.append("使用 fallback 分析")

    render_status_strip(status_messages)
    updated_result = render_update_console(result, config)
    if updated_result:
        result = updated_result
    render_dashboard(result, config, prefer_llm=prefer_llm)


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
:root {
    --ts-primary: #3F755F;
    --ts-primary-dark: #2F5F4C;
    --ts-text: #23352E;
    --ts-muted: #6B7D72;
    --ts-bg: #F7F8F5;
    --ts-card: #FFFFFF;
    --ts-soft: #EDF4EE;
    --ts-border: #DDE8DF;
    --ts-gold: #D8A84E;
    --ts-shadow: 0 14px 36px rgba(47, 95, 76, 0.10);
    --ts-radius: 22px;
}

[data-testid="stAppViewContainer"],
.stApp {
    background:
        radial-gradient(circle at top left, rgba(237, 244, 238, 0.95), transparent 34rem),
        linear-gradient(180deg, #F7F8F5 0%, #F1F5EF 100%);
    color: var(--ts-text);
}

[data-testid="stHeader"] {
    background: rgba(247, 248, 245, 0.86);
    border-bottom: 1px solid rgba(221, 232, 223, 0.70);
    backdrop-filter: blur(14px);
}

.block-container {
    max-width: 1320px;
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: var(--ts-text);
    letter-spacing: 0;
}

h1 {
    padding-bottom: 0.35rem;
    font-weight: 760;
}

h1 + div,
h1 + p,
[data-testid="stCaptionContainer"] {
    color: var(--ts-muted);
}

h2 {
    margin-top: 1.6rem;
    padding-top: 0.4rem;
    border-top: 1px solid rgba(221, 232, 223, 0.86);
}

h3 {
    color: var(--ts-primary-dark);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EDF4EE 0%, #F7F8F5 100%);
    border-right: 1px solid var(--ts-border);
}

[data-testid="stSidebarContent"] {
    padding: 1.35rem 1rem 2rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--ts-primary-dark);
}

[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--ts-card);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius);
    box-shadow: var(--ts-shadow);
}

[data-testid="stMetric"] {
    padding: 1rem 1.05rem;
}

[data-testid="stMetricLabel"] {
    color: var(--ts-muted);
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: var(--ts-primary-dark);
    font-weight: 760;
}

[data-testid="stAlert"] {
    border-radius: 18px;
    border: 1px solid var(--ts-border);
    box-shadow: 0 8px 24px rgba(47, 95, 76, 0.07);
}

[data-testid="stAlert"] > div {
    color: var(--ts-text);
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 999px;
    border: 1px solid rgba(63, 117, 95, 0.24);
    background: #FFFFFF;
    color: var(--ts-primary-dark);
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(47, 95, 76, 0.08);
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}

div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--ts-primary);
    color: var(--ts-primary-dark);
    box-shadow: 0 10px 24px rgba(47, 95, 76, 0.16);
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--ts-primary) 0%, var(--ts-primary-dark) 100%);
    border-color: var(--ts-primary-dark);
    color: #FFFFFF;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    border-radius: 16px;
    border-color: var(--ts-border);
    background: #FFFFFF;
    color: var(--ts-text);
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--ts-primary);
    box-shadow: 0 0 0 3px rgba(63, 117, 95, 0.12);
}

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--ts-border);
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(47, 95, 76, 0.08);
    overflow: hidden;
}

[data-testid="stExpander"] details > summary {
    color: var(--ts-primary-dark);
    font-weight: 720;
}

[data-testid="stTabs"] {
    background: rgba(255, 255, 255, 0.58);
    border: 1px solid rgba(221, 232, 223, 0.74);
    border-radius: 999px;
    padding: 0.25rem;
}

[data-testid="stTabs"] button {
    border-radius: 999px;
    color: var(--ts-muted);
    font-weight: 700;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background: var(--ts-primary);
    color: #FFFFFF;
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"] {
    background: var(--ts-card);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius);
    box-shadow: var(--ts-shadow);
    padding: 0.7rem;
}

[data-testid="stDataFrame"] {
    overflow: hidden;
}

hr {
    border-color: var(--ts-border);
}

code,
pre {
    border-radius: 16px;
    background: #F1F6F1 !important;
    color: #244C3E !important;
}

a {
    color: var(--ts-primary-dark);
    font-weight: 650;
}

a:hover {
    color: var(--ts-gold);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title("台股市場資訊儀表板")
    st.caption("整合股價、法人籌碼、新聞與 PTT 討論，快速掌握個股近期狀態。")


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("查詢")
        preset_label = st.selectbox(
            "常用股票",
            ["自訂"] + [f"{sid} {name}" for sid, name in STOCK_NAMES.items()],
            index=1,
        )
        default_stock = "2330" if preset_label == "自訂" else preset_label.split()[0]
        stock_id = st.text_input("股票代號", value=default_stock).strip()
        stock_name = st.text_input("股票名稱", value=get_stock_name(stock_id)).strip()
        question = st.text_area("觀察問題", value=f"{stock_name or stock_id} 最近狀況如何？", height=78)

        query_clicked = st.button("查詢", type="primary", use_container_width=True)
        refresh_clicked = st.button("重新抓取資料", use_container_width=True)

        with st.expander("進階爬蟲設定", expanded=False):
            days = st.slider("新聞/PTT 天數", min_value=1, max_value=93, value=7)
            chip_days = st.slider("法人資料天數", min_value=3, max_value=60, value=10)
            news_limit = st.slider("新聞上限", min_value=5, max_value=150, value=15, step=5)
            ptt_pages = st.slider("PTT 掃描頁數", min_value=1, max_value=120, value=20)
            news_source = st.selectbox("新聞來源", ["yahoo", "google", "both"], index=0)
            yahoo_mode = st.selectbox("Yahoo 新聞模式", ["rss", "auto", "scroll"], index=0)
            price_range = st.selectbox("股價區間", ["1mo", "3mo", "6mo", "1y"], index=0)
            st.caption(
                f"預計範圍：近 {days} 天、新聞上限 {news_limit} 篇、PTT {ptt_pages} 頁、"
                f"新聞來源 {news_source}、法人 {chip_days} 筆。"
            )

        render_api_controls()
        memory_enabled, rag_top_k = render_memory_controls()
        runtime_api_key = st.session_state.get("runtime_api_key", "")

    return {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "question": question,
        "query_clicked": query_clicked,
        "refresh_clicked": refresh_clicked,
        "days": days,
        "chip_days": chip_days,
        "news_limit": news_limit,
        "ptt_pages": ptt_pages,
        "news_source": news_source,
        "yahoo_mode": yahoo_mode,
        "price_range": price_range,
        "runtime_api_key": runtime_api_key,
        "memory_enabled": memory_enabled,
        "rag_top_k": rag_top_k,
    }


def render_api_controls() -> None:
    gemini_key, gemini_key_name = get_gemini_api_key()
    has_key = bool(os.getenv("OPENAI_API_KEY") or gemini_key)
    with st.expander("API 狀態", expanded=False):
        st.caption("有 API key 會使用模型產生摘要；沒有也會使用 fallback 分析。")
        if has_key:
            st.success(f"已偵測到 API key：{gemini_key_name if gemini_key else 'OPENAI_API_KEY'}")
        else:
            runtime_key = st.text_input(
                "請在此輸入您的 Google AI Studio API key，本次執行期間暫時使用，不會寫入檔案。",
                type="password",
            )
            if runtime_key:
                st.session_state["runtime_api_key"] = runtime_key
                st.success("本次工作階段已暫時使用輸入的 API key，不會寫入檔案。")
            else:
                st.info("未偵測到 API key，將使用 fallback 分析。")

        if st.button("測試 Gemini API", use_container_width=True):
            apply_runtime_api_key(st.session_state.get("runtime_api_key", ""))
            result = check_gemini_api(timeout=20)
            if result.status == "ok":
                st.success(f"{result.message} ({result.model}, {result.latency_ms}ms)")
            else:
                st.warning(f"{result.message} ({result.model})")


def render_memory_controls() -> tuple[bool, int]:
    with st.expander("AI 記憶與 RAG 測試", expanded=False):
        enabled = st.checkbox("啟用 session-level Memory", value=is_memory_enabled())
        set_memory_enabled(enabled)
        st.caption("Memory 只保存在目前 Streamlit session，不會寫入資料庫或檔案。")
        top_k = st.slider("Top-k 證據", min_value=1, max_value=10, value=5)
        if st.button("清除對話記憶", use_container_width=True):
            clear_memory()
            st.success("已清除目前 session 的對話記憶。")
        return enabled, top_k


def apply_runtime_api_key(api_key: str) -> None:
    if api_key and not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = api_key


def is_valid_stock_id(stock_id: str) -> bool:
    return bool(re.fullmatch(r"\d{4,6}", stock_id or ""))


def inspect_cache(stock_id: str) -> dict[str, Any]:
    root = DATA_ROOT / stock_id
    rag_path = root / "rag_documents.jsonl"
    rag_count = count_jsonl_rows(rag_path)
    demo_root = DEMO_ROOT / DEMO_STOCK_ID
    demo_rag_path = demo_root / "rag_documents.jsonl"
    demo_rag_count = count_jsonl_rows(demo_rag_path)
    raw_usable = rag_path.exists() and rag_count >= MIN_RAG_DOCUMENTS
    demo_usable = demo_rag_path.exists() and demo_rag_count >= MIN_RAG_DOCUMENTS
    if raw_usable:
        mode = "快取資料"
        usable_root = root
        usable_stock_id = stock_id
    elif demo_usable:
        mode = "示範資料"
        usable_root = demo_root
        usable_stock_id = DEMO_STOCK_ID
    else:
        mode = "快取不足"
        usable_root = root
        usable_stock_id = stock_id
    return {
        "root": root,
        "rag_path": rag_path,
        "rag_count": rag_count,
        "demo_root": demo_root,
        "demo_rag_count": demo_rag_count,
        "exists": root.exists(),
        "usable": raw_usable or demo_usable,
        "mode": mode,
        "usable_root": usable_root,
        "usable_stock_id": usable_stock_id,
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_cached_result(stock_id: str) -> dict[str, Any] | None:
    return load_result_from_folder(DATA_ROOT / stock_id, stock_id, "crawl_summary.json", "快取資料")


@st.cache_data(ttl=900, show_spinner=False)
def load_demo_result() -> dict[str, Any] | None:
    return load_result_from_folder(DEMO_ROOT / DEMO_STOCK_ID, DEMO_STOCK_ID, "demo_result.json", "示範資料")


def load_result_by_status(cache_status: dict[str, Any]) -> dict[str, Any] | None:
    if cache_status.get("mode") == "示範資料":
        return load_demo_result()
    return load_cached_result(str(cache_status.get("usable_stock_id") or ""))


def load_best_available_result(stock_id: str) -> dict[str, Any] | None:
    cache_status = inspect_cache(stock_id)
    if cache_status["usable"]:
        return load_result_by_status(cache_status)
    return None


def load_result_from_folder(root: Path, stock_id: str, summary_name: str, data_mode: str) -> dict[str, Any] | None:
    if not root.exists():
        return None

    summary = read_json(root / summary_name, default={})
    stock_name = summary.get("stock_name") or get_stock_name(stock_id)
    news_items = load_news_items(root / "news.json")
    ptt_items = load_news_items(root / "ptt_posts.json")
    rag_documents = load_jsonl(root / "rag_documents.jsonl")
    if not news_items and not ptt_items and rag_documents:
        news_items, ptt_items = news_items_from_rag_documents(rag_documents)
    errors = read_json(root / "crawl_errors.json", default=[])
    history = read_csv(root / "price_history.csv", EMPTY_PRICE_COLUMNS)
    chip_data = read_csv(root / "institutional_trading.csv", EMPTY_CHIP_COLUMNS)
    snapshot = summary.get("snapshot") or asdict(build_snapshot(stock_id, history))

    return {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "data_mode": data_mode,
        "source_root": str(root),
        "news_keyword": summary.get("news_keyword") or stock_name,
        "crawled_at": summary.get("crawled_at", ""),
        "crawl_config": summary.get("crawl_config", {}),
        "snapshot": snapshot,
        "price_summary": summary.get("price_summary", summarize_price(history)),
        "sentiment_summary": summary.get("sentiment_summary", summarize_sentiment(news_items + ptt_items)),
        "errors": errors if isinstance(errors, list) else [],
        "source_status": summary.get("source_status", []),
        "counts": {
            "price_rows": int(len(history)),
            "chip_rows": int(len(chip_data)),
            "news_items": int(len(news_items)),
            "ptt_items": int(len(ptt_items)),
            "rag_documents": int(len(rag_documents)),
            "errors": int(len(errors)) if isinstance(errors, list) else 0,
        },
        "data": {
            "price_history": history,
            "institutional_trading": chip_data,
            "news": news_items,
            "ptt": ptt_items,
            "rag_documents": rag_documents,
        },
    }


def run_crawler_safely(stock_id: str, stock_name: str, config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        load_cached_result.clear()
        return crawl_stock(
            stock_id,
            stock_name=stock_name,
            price_range=config["price_range"],
            chip_days=config["chip_days"],
            days=config["days"],
            news_limit=config["news_limit"],
            ptt_pages=config["ptt_pages"],
            news_source=config["news_source"],
            yahoo_mode=config["yahoo_mode"],
            output_dir=DATA_ROOT,
            save=True,
        )
    except Exception as exc:
        st.error(f"更新資料時發生問題：{friendly_error(exc)}")
        return None


def render_dashboard(result: dict[str, Any], config: dict[str, Any], prefer_llm: bool) -> None:
    history = result["data"]["price_history"]
    chip_data = result["data"]["institutional_trading"]
    news_items = result["data"]["news"]
    ptt_items = result["data"]["ptt"]
    snapshot = SimpleNamespace(**result["snapshot"])
    question = config["question"]

    st.subheader("1. 個股概覽")
    render_overview_metrics(result, snapshot, history, chip_data, news_items, ptt_items)

    st.subheader("2. 股價走勢")
    render_price_chart(history)
    render_price_chip_comparison(history, chip_data)

    st.subheader("3. 三大法人買賣超")
    render_chip_chart(chip_data)

    st.subheader("4. 新聞與 PTT 情緒")
    render_sentiment_chart(news_items, ptt_items)

    st.subheader("5. AI 近期觀察摘要")
    render_analysis(result, config, prefer_llm)

    st.subheader("6. 資料來源")
    render_sources(news_items, ptt_items, result.get("crawled_at", ""))

    render_quality_summary(result)
    render_financial_glossary()
    render_system_notes()


def render_overview_metrics(
    result: dict[str, Any],
    snapshot: SimpleNamespace,
    history: pd.DataFrame,
    chip_data: pd.DataFrame,
    news_items: list[NewsItem],
    ptt_items: list[NewsItem],
) -> None:
    price_summary = summarize_price(history)
    sentiment = summarize_sentiment(news_items + ptt_items)
    cols = st.columns(5)
    cols[0].metric("股票", f"{result['stock_id']} {result['stock_name']}")
    cols[1].metric("最新價", format_number(snapshot.last_price), metric_delta(snapshot.change_pct))
    cols[2].metric("區間趨勢", str(price_summary["trend"]))
    cols[3].metric("法人近五筆", chip_total_label(chip_data))
    cols[4].metric("情緒", str(sentiment["label"]), f"新聞 {len(news_items)} / PTT {len(ptt_items)}")

    crawled_at = result.get("crawled_at") or "未記錄"
    data_mode = result.get("data_mode") or "快取資料"
    st.caption(f"目前資料模式：{data_mode}；資料時間：{crawled_at}。摘要僅供資訊整理，不構成投資建議。")


def render_price_chart(history: pd.DataFrame) -> None:
    if history.empty or "close" not in history.columns:
        st.info("目前資料不足，無法繪製圖表。")
        return
    view = history.dropna(subset=["close"]).copy()
    if view.empty:
        st.info("目前資料不足，無法繪製圖表。")
        return
    st.line_chart(view.set_index("date")["close"], height=320)


def render_price_chip_comparison(history: pd.DataFrame, chip_data: pd.DataFrame) -> None:
    if history.empty or chip_data.empty or "close" not in history.columns or "total_net" not in chip_data.columns:
        st.info("目前資料不足，無法繪製股價與法人對照圖。")
        return

    price = history[["date", "close"]].copy()
    chip = chip_data[["date", "total_net"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    chip["date"] = pd.to_datetime(chip["date"], errors="coerce")
    merged = pd.merge(price.dropna(), chip.dropna(), on="date", how="inner").tail(20)
    if merged.empty:
        st.info("目前資料不足，無法繪製股價與法人對照圖。")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=merged["date"],
            y=merged["total_net"],
            name="三大法人合計買賣超",
            yaxis="y2",
            marker_color="#4C78A8",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["close"],
            name="收盤價",
            mode="lines+markers",
            line={"color": "#F58518", "width": 3},
            yaxis="y1",
        )
    )
    fig.update_layout(
        title="股價走勢與三大法人合計買賣超對照",
        height=360,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend={"orientation": "h", "y": 1.08},
        yaxis={"title": "收盤價"},
        yaxis2={"title": "買賣超股數", "overlaying": "y", "side": "right", "showgrid": False},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("此圖用於比較法人資金方向與股價變化是否一致，不代表預測股價。")


def render_chip_chart(chip_data: pd.DataFrame) -> None:
    required = {"foreign_net", "investment_trust_net", "dealer_net"}
    if chip_data.empty or not required.issubset(chip_data.columns):
        st.info("目前資料不足，無法繪製圖表。")
        return
    render_chip_interpretation(chip_data)
    view = chip_data.tail(10).copy()
    view = view.rename(
        columns={
            "foreign_net": "外資",
            "investment_trust_net": "投信",
            "dealer_net": "自營商",
        }
    )
    st.bar_chart(view.set_index("date")[["外資", "投信", "自營商"]], height=320)
    with st.expander("法人明細", expanded=False):
        table = chip_data.tail(10).rename(
            columns={
                "date": "日期",
                "foreign_net": "外資",
                "investment_trust_net": "投信",
                "dealer_net": "自營商",
                "total_net": "合計",
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_chip_interpretation(chip_data: pd.DataFrame) -> None:
    summary = build_chip_interpretation(chip_data)
    st.markdown("#### 籌碼解讀")

    cols = st.columns(4)
    cols[0].metric("近 5 筆三大法人合計", signed_amount(summary["total_net"]))
    cols[1].metric("主要影響來源", summary["major_source"])
    cols[2].metric("目前狀態", summary["status"])
    cols[3].metric("連續方向", summary["streak"])

    st.info(
        f"{summary['plain_text']} 法人買超或賣超只能代表近期資金方向，"
        "不代表股價一定上漲或下跌，仍需搭配股價、成交量與新聞事件判斷。"
    )
    with st.expander("近五筆法人方向", expanded=False):
        st.dataframe(summary["detail"], use_container_width=True, hide_index=True)


def build_chip_interpretation(chip_data: pd.DataFrame) -> dict[str, Any]:
    recent = chip_data.tail(5).copy()
    for col, _label in INSTITUTION_COLUMNS + [("total_net", "三大法人")]:
        if col in recent.columns:
            recent[col] = pd.to_numeric(recent[col], errors="coerce").fillna(0)

    totals = {label: int(recent[col].sum()) for col, label in INSTITUTION_COLUMNS if col in recent.columns}
    total_net = int(recent["total_net"].sum()) if "total_net" in recent.columns else int(sum(totals.values()))
    if totals and any(value != 0 for value in totals.values()):
        major_source = max(totals, key=lambda label: abs(totals[label]))
    else:
        major_source = "無明顯來源"

    if total_net > 0:
        status = "法人籌碼偏買超"
        plain_text = "外資、投信與自營商近五筆合計偏買超，代表近期法人資金較偏向買方。"
    elif total_net < 0:
        status = "法人籌碼偏賣超"
        plain_text = "外資、投信與自營商近五筆合計偏賣超，代表近期法人資金較偏向賣方。"
    else:
        status = "法人籌碼大致中性"
        plain_text = "外資、投信與自營商近五筆合計接近持平，法人資金方向尚不明顯。"

    detail_rows = []
    for col, label in INSTITUTION_COLUMNS:
        if col not in recent.columns:
            continue
        detail_rows.append(
            {
                "法人": label,
                "近五筆方向": "、".join(direction_word(value) for value in recent[col].tolist()),
                "近五筆合計": signed_amount(totals.get(label, 0)),
            }
        )

    return {
        "total_net": total_net,
        "major_source": major_source,
        "status": status,
        "plain_text": plain_text,
        "streak": consecutive_direction_label(chip_data.get("total_net", pd.Series(dtype=float))),
        "detail": pd.DataFrame(detail_rows),
    }


def consecutive_direction_label(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    if not values:
        return "資料不足"
    latest_sign = number_sign(values[-1])
    if latest_sign == 0:
        return "最新一筆持平"
    count = 0
    for value in reversed(values):
        if number_sign(value) != latest_sign:
            break
        count += 1
    direction = "買超" if latest_sign > 0 else "賣超"
    if count >= 2:
        return f"連續 {count} 日{direction}"
    return "方向不連續"


def direction_word(value: Any) -> str:
    sign = number_sign(value)
    if sign > 0:
        return "買"
    if sign < 0:
        return "賣"
    return "平"


def number_sign(value: Any) -> int:
    try:
        number = float(value)
    except Exception:
        return 0
    if number > 0:
        return 1
    if number < 0:
        return -1
    return 0


def signed_amount(value: Any) -> str:
    try:
        number = int(float(value))
    except Exception:
        return "資料不足"
    if number > 0:
        return f"買超 {number:,} 股"
    if number < 0:
        return f"賣超 {abs(number):,} 股"
    return "持平"


def render_sentiment_chart(news_items: list[NewsItem], ptt_items: list[NewsItem]) -> None:
    rows = []
    for label, items in (("新聞", news_items), ("PTT", ptt_items)):
        summary = summarize_sentiment(items)
        for sentiment in ("正向", "中立", "負向"):
            rows.append({"來源": label, "情緒": sentiment, "筆數": int(summary[sentiment])})
    sentiment_df = pd.DataFrame(rows)
    if sentiment_df["筆數"].sum() == 0:
        st.info("目前資料不足，無法繪製圖表。")
        return
    pivot = sentiment_df.pivot(index="情緒", columns="來源", values="筆數").fillna(0)
    st.bar_chart(pivot, height=280)

    col_news, col_ptt = st.columns(2)
    with col_news:
        st.caption("近期新聞")
        render_item_list(news_items[:5])
    with col_ptt:
        st.caption("PTT 討論")
        render_item_list(ptt_items[:5])


def render_analysis(result: dict[str, Any], config: dict[str, Any], prefer_llm: bool) -> None:
    question = config["question"]
    memory_enabled = bool(config.get("memory_enabled", True))
    top_k = int(config.get("rag_top_k", 5))
    memory = get_memory()
    if not memory_enabled:
        memory["enabled"] = False
    rewritten = rewrite_query(
        question,
        memory,
        fallback_stock_id=str(result.get("stock_id", "")),
        fallback_stock_name=str(result.get("stock_name", "")),
    )

    render_memory_and_query_state(memory_enabled, memory, rewritten)

    try:
        analysis = analyze_result(
            result,
            question=rewritten["rewritten_query"],
            prefer_llm=prefer_llm,
            output_root=None,
            use_rag=True,
            rag_top_k=top_k,
            rag_query=rewritten["rag_query"],
        )
        if not prefer_llm:
            st.info("未使用 API key，本段為 fallback 分析。")
        render_rag_evidence(analysis.rag)
        st.markdown(analysis.answer)
        try:
            gemini_key, _ = get_gemini_api_key()
            diagnostics = build_query_diagnostics(
                result=result,
                rewritten=rewritten,
                rag=analysis.rag,
                answer=analysis.answer,
                memory_enabled=memory_enabled,
                openai_key_present=bool(os.getenv("OPENAI_API_KEY")),
                gemini_key_present=bool(gemini_key),
            )
            render_query_diagnostics(diagnostics)
        except Exception:
            with st.expander("本輪查詢診斷", expanded=False):
                st.caption("本輪診斷資訊暫時無法取得，不影響原本回答。")
        update_memory(
            user_message=question,
            assistant_message=analysis.answer,
            stock_id=str(result.get("stock_id", "")),
            stock_name=str(result.get("stock_name", "")),
            focus=rewritten["detected_focus"],
            referenced_sources=[item["reference_id"] for item in analysis.rag.evidence],
        )
    except Exception as exc:
        st.warning(f"摘要產生失敗，已改用簡易說明：{friendly_error(exc)}")
        st.write("目前可用資料有限，請先查看股價、法人籌碼、新聞與 PTT 區塊。")


def render_memory_and_query_state(
    memory_enabled: bool,
    memory: dict[str, Any],
    rewritten: dict[str, Any],
) -> None:
    with st.expander("Memory 與 Query Rewriting", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Memory", "啟用" if memory_enabled else "關閉")
        cols[1].metric("本輪使用 Memory", "是" if rewritten["used_memory"] else "否")
        cols[2].metric("目前股票", f"{memory.get('current_stock_id') or '尚未記錄'} {memory.get('current_stock_name') or ''}")
        cols[3].metric("偵測主題", rewritten["detected_focus"])

        st.write("原始問題")
        st.code(rewritten["original_query"], language="text")
        st.write("補全後問題")
        st.code(rewritten["rewritten_query"], language="text")
        st.write("實際 RAG query")
        st.code(rewritten["rag_query"], language="text")


def render_rag_evidence(rag: Any) -> None:
    with st.expander("Top-k RAG 檢索證據", expanded=True):
        source_labels = {
            "news": "新聞",
            "ptt": "PTT",
            "glossary": "金融名詞",
            "analysis_rule": "分析規則",
        }
        st.caption(
            f"本機文字檢索 RAG；文件數 {rag.document_count}，"
            f"Chunk 數 {rag.chunk_count}，Top-k {rag.top_k}，引用來源 {len(rag.evidence)}。"
        )
        source_summary = "、".join(
            f"{source_labels.get(source_type, source_type)} {count} 筆"
            for source_type, count in rag.source_counts.items()
            if source_type
        )
        if source_summary:
            st.caption(f"本次檢索來源：{source_summary}")
        if rag.status != "ok" or not rag.evidence:
            st.warning("目前檢索證據不足，AI 摘要僅能作為有限資料整理。")
            st.code(rag.context, language="text")
            return

        rows = []
        for item in rag.evidence:
            rows.append(
                {
                    "rank": item["rank"],
                    "reference_id": item["reference_id"],
                    "score": item["score"],
                    "source_type": source_labels.get(item.get("source_type", ""), item.get("source_type", "")),
                    "stock_id": item.get("stock_id", ""),
                    "stock_name": item.get("stock_name", ""),
                    "source": item["source"],
                    "title": item["title"],
                    "published_at": item["published_at"],
                    "url": item["url"],
                    "category": item.get("category", ""),
                    "content": shorten_text(item["content"], 180),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("顯示實際送入 LLM / fallback 的 RAG Context", expanded=False):
            st.code(rag.context, language="text")


def render_query_diagnostics(diagnostics: dict[str, Any]) -> None:
    with st.expander("本輪查詢診斷", expanded=False):
        metrics = st.columns(4)
        metrics[0].metric(
            "股票辨識",
            f"{diagnostics.get('stock_id') or '未提供'} {diagnostics.get('stock_name') or ''}".strip(),
        )
        metrics[1].metric("Memory", "已使用" if diagnostics.get("used_memory") else "未使用")
        metrics[2].metric("RAG 證據數", diagnostics.get("evidence_count", 0))
        metrics[3].metric("回答模式", diagnostics.get("answer_mode", "未提供"))

        st.caption(
            f"問題焦點：{diagnostics.get('focus', '未提供')}｜"
            f"Memory 狀態：{'啟用' if diagnostics.get('memory_enabled') else '關閉'}｜"
            f"Query Rewriting：{'已執行' if diagnostics.get('query_rewriting_executed') else '未執行'}"
        )
        st.write("原始問題")
        st.code(str(diagnostics.get("original_query", "未提供")), language="text")
        st.write("補全後問題")
        st.code(str(diagnostics.get("rewritten_query", "未提供")), language="text")
        st.write("實際 RAG Query")
        st.code(str(diagnostics.get("rag_query", "未提供")), language="text")

        rag_metrics = st.columns(4)
        rag_metrics[0].metric("RAG 狀態", "成功" if diagnostics.get("rag_success") else "證據不足")
        rag_metrics[1].metric("Top-k", diagnostics.get("top_k", 0))
        rag_metrics[2].metric("最高分數", format_score(diagnostics.get("max_score")))
        rag_metrics[3].metric("平均分數", format_score(diagnostics.get("average_score")))

        source_counts = diagnostics.get("source_counts", {})
        source_cols = st.columns(4)
        source_cols[0].metric("新聞", source_counts.get("news", 0))
        source_cols[1].metric("PTT", source_counts.get("ptt", 0))
        source_cols[2].metric("金融名詞", source_counts.get("glossary", 0))
        source_cols[3].metric("分析規則", source_counts.get("analysis_rule", 0))

        technical_rows = [
            {"診斷項目": "不同股票的動態文件", "結果": diagnostics.get("different_stock_documents", 0)},
            {"診斷項目": "證據不足狀態", "結果": "是" if diagnostics.get("insufficient_evidence") else "否"},
            {"診斷項目": "API Key", "結果": "已提供" if diagnostics.get("api_key_present") else "未提供"},
            {"診斷項目": "回答含引用", "結果": "是" if diagnostics.get("has_references") else "否"},
            {"診斷項目": "引用 reference 數", "結果": diagnostics.get("reference_count", 0)},
            {"診斷項目": "包含資料限制", "結果": "是" if diagnostics.get("has_limitations") else "否"},
            {"診斷項目": "投資建議安全規則", "結果": "已觸發" if diagnostics.get("advice_rule_triggered") else "未偵測"},
            {"診斷項目": "價格預測安全規則", "結果": "已觸發" if diagnostics.get("prediction_rule_triggered") else "未偵測"},
        ]
        st.dataframe(pd.DataFrame(technical_rows), use_container_width=True, hide_index=True)

        data_cols = st.columns(4)
        data_cols[0].metric("股價資料", format_record_count(diagnostics.get("price_rows")))
        data_cols[1].metric("法人資料", format_record_count(diagnostics.get("chip_rows")))
        data_cols[2].metric("新聞資料", format_record_count(diagnostics.get("news_items")))
        data_cols[3].metric("PTT 資料", format_record_count(diagnostics.get("ptt_items")))
        st.caption(
            f"資料模式：{diagnostics.get('data_mode', '未提供')}｜"
            f"最後更新：{diagnostics.get('updated_at', '未提供')}"
        )

        if diagnostics.get("insufficient_evidence"):
            st.warning(diagnostics.get("summary", "本輪未取得足夠相關證據。"))
        else:
            st.info(diagnostics.get("summary", "本輪查詢診斷完成。"))


def render_sources(news_items: list[NewsItem], ptt_items: list[NewsItem], crawled_at: str) -> None:
    source_counts: dict[str, int] = {}
    for item in news_items + ptt_items:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
    if source_counts:
        source_text = "、".join(f"{source} {count} 筆" for source, count in source_counts.items())
        st.write(source_text)
    else:
        st.info("目前沒有可顯示的新聞或社群來源。")
    if crawled_at:
        st.caption(f"最近更新：{crawled_at}")


def render_quality_summary(result: dict[str, Any]) -> None:
    counts = result.get("counts", {})
    errors = result.get("errors", [])
    with st.expander("資料品質摘要", expanded=False):
        quality_df = pd.DataFrame(
            [
                {"項目": "股價筆數", "數量": counts.get("price_rows", 0)},
                {"項目": "法人筆數", "數量": counts.get("chip_rows", 0)},
                {"項目": "新聞筆數", "數量": counts.get("news_items", 0)},
                {"項目": "PTT 筆數", "數量": counts.get("ptt_items", 0)},
                {"項目": "RAG 文件筆數", "數量": counts.get("rag_documents", 0)},
                {"項目": "錯誤訊息", "數量": len(errors)},
            ]
        )
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
        if counts.get("news_items", 0) < 5 or counts.get("ptt_items", 0) < 3:
            st.warning("本次新聞或社群資料量較少，因此 AI 摘要僅能作為資訊整理，不代表完整市場意見。")
        if errors:
            st.caption("錯誤紀錄")
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)


def render_financial_glossary() -> None:
    with st.expander("金融名詞小字典", expanded=False):
        st.markdown(
            """
- **外資**：國外法人或外國機構投資人。
- **投信**：基金公司或資產管理機構。
- **自營商**：券商使用自有資金進行交易的部位。
- **買超**：買進股數大於賣出股數。
- **賣超**：賣出股數大於買進股數。
- **籌碼**：市場中股票主要由誰持有、誰正在買賣的狀態。
            """.strip()
        )


def render_item_list(items: list[NewsItem]) -> None:
    if not items:
        st.info("目前資料不足。")
        return
    for item in items:
        date_text = item.display_date or item.published_at or ""
        suffix = f" · {date_text}" if date_text else ""
        title = item.title or "未命名資料"
        if item.url:
            st.markdown(f"- [{title}]({item.url}) `{item.sentiment}`{suffix}")
        else:
            st.markdown(f"- {title} `{item.sentiment}`{suffix}")


def render_status_strip(messages: list[str]) -> None:
    if not messages:
        return
    cols = st.columns(len(messages))
    for col, message in zip(cols, messages):
        col.info(message)


def render_update_console(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    st.subheader("資料更新控制台")
    counts = result.get("counts", {})
    crawled_at = result.get("crawled_at") or "未記錄"
    mode = result.get("data_mode") or "快取資料"

    cols = st.columns(6)
    cols[0].metric("目前資料來源", mode)
    cols[1].metric("股價資料", f"{counts.get('price_rows', 0)} 筆")
    cols[2].metric("法人資料", f"{counts.get('chip_rows', 0)} 筆")
    cols[3].metric("新聞資料", f"{counts.get('news_items', 0)} 筆")
    cols[4].metric("PTT 資料", f"{counts.get('ptt_items', 0)} 筆")
    cols[5].metric("RAG 文件", f"{counts.get('rag_documents', 0)} 筆")

    st.caption(f"最後更新時間：{crawled_at}")
    if result.get("source_root"):
        st.caption(f"資料位置：`{result['source_root']}`")
    render_source_status(result.get("source_status", []))

    if st.button("重新抓取最新資料", key="main_refresh_latest", use_container_width=True):
        stock_id = normalize_stock_id(config["stock_id"])
        if not is_valid_stock_id(stock_id):
            st.error("股票代號格式不正確，請輸入 4 到 6 位數字。")
            return None
        stock_name = config["stock_name"] or get_stock_name(stock_id)
        show_crawl_plan(config)
        with st.spinner("正在重新抓取最新資料，若資料來源忙碌會保留目前畫面..."):
            refreshed = run_crawler_safely(stock_id, stock_name, config)
        if refreshed:
            refreshed["data_mode"] = "即時爬取"
            st.success("已重新抓取最新資料。")
            st.session_state["active_dashboard_config"] = {
                **config,
                "query_clicked": True,
                "refresh_clicked": False,
            }
            return refreshed
        st.warning("更新資料失敗，已保留目前可用資料。")
    return None


def render_source_status(source_status: list[dict[str, Any]]) -> None:
    if not source_status:
        return
    status_label = {
        "success": "成功",
        "failed": "失敗",
        "cache": "使用快取",
        "demo": "使用 demo",
    }
    rows = []
    for item in source_status:
        rows.append(
            {
                "來源": item.get("source", ""),
                "狀態": status_label.get(str(item.get("status", "")), item.get("status", "")),
                "筆數": item.get("record_count", 0),
                "使用快取": "是" if item.get("used_cache") else "否",
                "使用 demo": "是" if item.get("used_demo") else "否",
                "更新時間": item.get("updated_at", ""),
                "錯誤訊息": item.get("error_message", ""),
            }
        )
    with st.expander("各資料來源狀態", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_crawl_plan(config: dict[str, Any]) -> None:
    st.info(
        "預計爬取範圍："
        f"近 {config['days']} 天、新聞上限 {config['news_limit']} 篇、PTT {config['ptt_pages']} 頁、"
        f"新聞來源 {config['news_source']}、法人資料 {config['chip_days']} 筆。"
    )


def render_cache_summary(stock_id: str, cache_status: dict[str, Any]) -> None:
    st.write(f"快取位置：`data/raw_data/{stock_id}`")
    st.write(f"目前 RAG 文件筆數：{cache_status['rag_count']}，展示建議至少 {MIN_RAG_DOCUMENTS} 筆。")


def render_system_notes() -> None:
    with st.expander("系統說明", expanded=False):
        st.write("本儀表板優先使用既有快取資料，只有按下「重新抓取資料」才會連網更新。")
        st.write("若未設定 API key，系統會使用 fallback 分析，仍可完成展示。")


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except Exception:
        return rows
    return rows


def count_jsonl_rows(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame(columns=columns)
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df
    except Exception:
        return pd.DataFrame(columns=columns)


def load_news_items(path: Path) -> list[NewsItem]:
    rows = read_json(path, default=[])
    if not isinstance(rows, list):
        return []
    allowed = set(NewsItem.__dataclass_fields__.keys())
    items: list[NewsItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        data = {key: row.get(key) for key in allowed if key in row}
        try:
            items.append(NewsItem(**data))
        except TypeError:
            continue
    return items


def news_items_from_rag_documents(documents: list[dict[str, Any]]) -> tuple[list[NewsItem], list[NewsItem]]:
    news_items: list[NewsItem] = []
    ptt_items: list[NewsItem] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        source = str(doc.get("source") or "資料來源")
        item = NewsItem(
            title=str(doc.get("title") or "未命名資料"),
            url=str(doc.get("url") or ""),
            source=source,
            published_at=str(doc.get("published_at") or "") or None,
            display_date=str(doc.get("display_date") or ""),
            sentiment=str(doc.get("sentiment") or "中立"),
            content=str(doc.get("text") or doc.get("content") or ""),
            keyword=str(doc.get("stock_name") or doc.get("keyword") or ""),
            list_score=safe_int_or_none(doc.get("list_score")),
            push_count=safe_int_or_none(doc.get("push_count")),
            boo_count=safe_int_or_none(doc.get("boo_count")),
            arrow_count=safe_int_or_none(doc.get("arrow_count")),
            reply_count=safe_int_or_none(doc.get("reply_count")),
        )
        if "PTT" in source.upper():
            ptt_items.append(item)
        else:
            news_items.append(item)
    return news_items, ptt_items


def safe_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def shorten_text(value: str, max_chars: int = 180) -> str:
    text = " ".join((value or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_score(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "未提供"
        return f"{float(value):.4f}"
    except Exception:
        return "未提供"


def format_record_count(value: Any) -> str:
    if value is None or value == "未提供":
        return "未提供"
    try:
        return f"{int(value)} 筆"
    except Exception:
        return "未提供"


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def metric_delta(value: Any) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
        return f"{float(value):+.2f}%"
    except Exception:
        return None


def chip_total_label(chip_data: pd.DataFrame) -> str:
    if chip_data.empty or "total_net" not in chip_data.columns:
        return "資料不足"
    total = int(chip_data.tail(5)["total_net"].sum())
    if total > 0:
        return f"買超 {total:,}"
    if total < 0:
        return f"賣超 {abs(total):,}"
    return "持平"


def has_any_llm_key() -> bool:
    gemini_key, _ = get_gemini_api_key()
    return bool(os.getenv("OPENAI_API_KEY") or gemini_key)


def friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name in {"HTTPError", "ConnectionError", "Timeout", "ReadTimeout"}:
        return "網路連線或資料來源暫時無法回應。"
    if name in {"ValueError", "KeyError"}:
        return "資料格式不完整，請重新抓取或改查其他股票。"
    return "系統暫時無法完成這個動作，請稍後再試。"


if __name__ == "__main__":
    main()
