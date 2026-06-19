from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

from ai_core.prompts import ANALYSIS_TEMPLATE, FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from ai_core.api_health import DEFAULT_GEMINI_MODEL, get_gemini_api_key
from scrapers.news_spider import NewsItem, summarize_sentiment
from scrapers.yfinance_client import StockSnapshot, summarize_price


def build_context(
    stock_id: str,
    stock_name: str,
    question: str,
    history: pd.DataFrame,
    chip_data: pd.DataFrame,
    news_items: list[NewsItem],
    ptt_items: list[NewsItem],
    rag_context: str | None = None,
) -> str:
    price_context = _format_price_context(history)
    chip_context = _format_chip_context(chip_data)
    news_context = _format_items(news_items) or "未取得近期新聞。"
    sentiment_context = _format_sentiment(news_items + ptt_items)
    rag_context = rag_context or "尚未建立或檢索 RAG 文件。"
    answer_guidance = _answer_guidance(question)

    return ANALYSIS_TEMPLATE.format(
        stock_id=stock_id,
        stock_name=stock_name,
        question=question,
        rag_context=rag_context,
        price_context=price_context,
        chip_context=chip_context,
        news_context=news_context,
        sentiment_context=sentiment_context,
        answer_guidance=answer_guidance,
    )


def analyze_stock(
    stock_id: str,
    stock_name: str,
    question: str,
    history: pd.DataFrame,
    chip_data: pd.DataFrame,
    news_items: list[NewsItem],
    ptt_items: list[NewsItem],
    snapshot: StockSnapshot | None = None,
    prefer_llm: bool = True,
    rag_context: str | None = None,
) -> str:
    context = build_context(
        stock_id,
        stock_name,
        question,
        history,
        chip_data,
        news_items,
        ptt_items,
        rag_context=rag_context,
    )
    if prefer_llm:
        llm_answer = _try_llm(context)
        if llm_answer:
            return llm_answer
    return rule_based_analysis(stock_name, history, chip_data, news_items, ptt_items, snapshot, question=question, rag_context=rag_context)


def rule_based_analysis(
    stock_name: str,
    history: pd.DataFrame,
    chip_data: pd.DataFrame,
    news_items: list[NewsItem],
    ptt_items: list[NewsItem],
    snapshot: StockSnapshot | None = None,
    question: str | None = None,
    rag_context: str | None = None,
) -> str:
    question = question or ""
    price = summarize_price(history)
    sentiment = summarize_sentiment(news_items + ptt_items)
    chip_summary = _chip_direction(chip_data)
    news_summary = _news_summary(news_items)
    if _is_glossary_question(question):
        return _rule_based_glossary_analysis(question, rag_context)
    if _is_financial_topic_question(question):
        return _rule_based_financial_topic_analysis(
            stock_name,
            question,
            history,
            chip_data,
            news_items,
            ptt_items,
            snapshot,
            rag_context,
        )

    price_line = ""
    if snapshot and snapshot.last_price is not None:
        change_pct = f"{snapshot.change_pct:.2f}%" if snapshot.change_pct is not None else "N/A"
        price_line = f"目前收盤/最新價約 {snapshot.last_price:.2f}，單日變動 {change_pct}。"
    elif price["return_pct"] is not None:
        price_line = f"區間報酬約 {price['return_pct']:.2f}%，股價趨勢判讀為{price['trend']}。"
    else:
        price_line = "股價資料不足，無法判斷價格趨勢。"

    return "\n\n".join(
        [
            f"一、數據事實：{price_line} {chip_summary}",
            (
                f"二、近期事件與討論：{news_summary} "
                f"目前取得資料的情緒以「{sentiment['label']}」為主 "
                f"(正向 {sentiment['正向']}、中立 {sentiment['中立']}、負向 {sentiment['負向']})。"
                f" 檢索證據摘要：{_shorten_context(rag_context or '', 420)}"
            ),
            (
                f"三、綜合解讀：{stock_name} 的價格面顯示「{price['trend']}」。"
                "法人買賣超只能描述近期資金方向；同期新聞可能是市場關注因素之一，"
                "但僅憑時間重疊無法確認與股價具有直接因果關係。"
            ),
            (
                "四、名詞補充：買超代表特定期間買進股數大於賣出股數，賣超則相反。"
                "法人買超不代表股價一定上漲，PTT 情緒也只代表目前取得文章的討論傾向。"
            ),
            (
                "五、資料限制：此為資訊整理與教學展示，不構成投資建議，也不預測確切未來股價。"
                "若網路、TWSE、Yahoo 或 PTT 無法連線，分析會以已取得資料與規則式摘要為主。"
            ),
        ]
    )


def _try_llm(context: str) -> str | None:
    return _try_openai(context) or _try_gemini(context)


def _answer_guidance(question: str) -> str:
    return "\n".join(
        [
            "一、數據事實：只整理結構化股價、成交量與三大法人資料。",
            "二、近期事件與討論：使用 RAG 新聞/PTT 證據並標示 [R1]、[R2]。",
            "三、綜合解讀：區分同時發生與直接因果，避免過度推論。",
            "四、名詞補充：使用金融名詞與分析規則證據。",
            "五、資料限制：說明日期、樣本不足、不提供投資建議且不預測確切股價。",
        ]
    )


def _is_financial_topic_question(question: str) -> bool:
    keywords = (
        "財報",
        "營收",
        "獲利",
        "eps",
        "EPS",
        "毛利",
        "營益",
        "題材",
        "展望",
        "未來",
        "成長動能",
        "法說",
    )
    return any(keyword in question for keyword in keywords)


def _is_glossary_question(question: str) -> bool:
    return any(keyword in question for keyword in ("是什麼", "什麼意思", "代表什麼", "如何解讀", "怎麼解讀"))


def _rule_based_glossary_analysis(question: str, rag_context: str | None) -> str:
    context = _shorten_context(rag_context or "", 900)
    return "\n\n".join(
        [
            "一、數據事實：這是一個金融名詞解釋問題，不需要用股價或法人每日數字作為定義。",
            f"二、近期事件與討論：本題主要使用金融名詞與分析規則證據。{context}",
            (
                "三、綜合解讀：買超表示特定期間買進股數大於賣出股數，可用來觀察近期資金方向；"
                "但法人買超不代表股價一定上漲，也不能直接視為買進訊號。"
            ),
            "四、名詞補充：外資買超是指外資在特定期間的買進股數大於賣出股數。",
            "五、資料限制：名詞解釋不等於投資建議，實際市場判讀仍需搭配股價、成交量、資料日期與其他資訊。",
        ]
    )


def _try_openai(context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _try_gemini(context: str) -> str | None:
    api_key, _ = get_gemini_api_key()
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES + "\n\n" + context
    try:
        response = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        parts = payload["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts).strip() or None
    except Exception:
        return None


def _format_price_context(history: pd.DataFrame) -> str:
    if history.empty:
        return "未取得股價資料。"
    view = history.dropna(subset=["close"]).tail(5)
    if view.empty:
        return "未取得有效收盤價資料。"
    latest_close = float(view.iloc[-1]["close"])
    first_close = float(view.iloc[0]["close"])
    five_day_return = ((latest_close - first_close) / first_close * 100) if first_close else None
    latest_volume = view.iloc[-1].get("volume")
    first_volume = view.iloc[0].get("volume")
    volume_change = None
    try:
        if first_volume and not pd.isna(first_volume) and not pd.isna(latest_volume):
            volume_change = (float(latest_volume) - float(first_volume)) / float(first_volume) * 100
    except Exception:
        volume_change = None
    latest_date = view.iloc[-1].get("date")
    return (
        f"最新資料日期: {latest_date}; 最新收盤價: {latest_close:.2f}; "
        f"近五筆漲跌幅: {_format_optional_pct(five_day_return)}; "
        f"最新成交量: {_format_optional_number(latest_volume)}; "
        f"近五筆成交量變化: {_format_optional_pct(volume_change)}"
    )


def _format_chip_context(chip_data: pd.DataFrame) -> str:
    if chip_data.empty:
        return "未取得三大法人買賣超資料。"
    recent = chip_data.tail(5)
    foreign = int(pd.to_numeric(recent["foreign_net"], errors="coerce").fillna(0).sum())
    trust = int(pd.to_numeric(recent["investment_trust_net"], errors="coerce").fillna(0).sum())
    dealer = int(pd.to_numeric(recent["dealer_net"], errors="coerce").fillna(0).sum())
    total = int(pd.to_numeric(recent["total_net"], errors="coerce").fillna(0).sum())
    direction = "偏買超" if total > 0 else "偏賣超" if total < 0 else "大致持平"
    return (
        f"近五筆外資合計: {foreign:,} 股; 投信合計: {trust:,} 股; "
        f"自營商合計: {dealer:,} 股; 三大法人合計: {total:,} 股; "
        f"法人整體方向: {direction}"
    )


def _format_items(items: list[NewsItem]) -> str:
    return "\n".join(f"- [{item.source}] {item.title} ({item.sentiment})" for item in items)


def _format_sentiment(items: list[NewsItem]) -> str:
    summary = summarize_sentiment(items)
    return (
        f"整體: {summary['label']}; "
        f"正向: {summary['正向']}; 中立: {summary['中立']}; 負向: {summary['負向']}"
    )


def _chip_direction(chip_data: pd.DataFrame) -> str:
    if chip_data.empty:
        return "目前未取得三大法人買賣超資料，因此無法判斷外資、投信與自營商的近期方向。"

    recent = chip_data.tail(5)
    total = int(recent["total_net"].sum())
    foreign = int(recent["foreign_net"].sum())
    trust = int(recent["investment_trust_net"].sum())
    dealer = int(recent["dealer_net"].sum())
    direction = "偏買超" if total > 0 else "偏賣超" if total < 0 else "大致持平"
    return (
        f"近 {len(recent)} 筆三大法人合計為 {direction}，"
        f"外資 {foreign:,} 股、投信 {trust:,} 股、自營商 {dealer:,} 股，"
        f"合計 {total:,} 股。"
    )


def _news_summary(news_items: list[NewsItem]) -> str:
    if not news_items:
        return "目前未取得新聞資料，無法用近期事件輔助判斷。"
    top_titles = "；".join(item.title for item in news_items[:3])
    sentiment = summarize_sentiment(news_items)
    return f"共整理 {len(news_items)} 則新聞，標題情緒以「{sentiment['label']}」為主。重點包含：{top_titles}。"


def _rule_based_financial_topic_analysis(
    stock_name: str,
    question: str,
    history: pd.DataFrame,
    chip_data: pd.DataFrame,
    news_items: list[NewsItem],
    ptt_items: list[NewsItem],
    snapshot: StockSnapshot | None,
    rag_context: str | None,
) -> str:
    price = summarize_price(history)
    sentiment = summarize_sentiment(news_items + ptt_items)
    topic_items = _select_topic_items(news_items + ptt_items, question, limit=5)
    topic_titles = "；".join(item.title for item in topic_items) if topic_items else "目前資料中沒有明確相關標題。"
    rag_note = _shorten_context(rag_context or "", 260)

    if snapshot and snapshot.last_price is not None:
        price_line = f"最新價約 {snapshot.last_price:.2f}，單日變動 {snapshot.change_pct:.2f}%。" if snapshot.change_pct is not None else f"最新價約 {snapshot.last_price:.2f}。"
    elif price["return_pct"] is not None:
        price_line = f"區間報酬約 {price['return_pct']:.2f}%，價格趨勢為{price['trend']}。"
    else:
        price_line = "目前沒有足夠股價資料。"

    return "\n\n".join(
        [
            (
                f"一、數據事實：{_chip_direction(chip_data)} {price_line} "
                "本系統尚未直接串接完整財報資料庫，因此不能提供未取得的 EPS、毛利率或逐季損益數字。"
            ),
            f"二、近期事件與討論：已取得資料中較相關的線索包含：{topic_titles}。RAG 檢索摘要：{rag_note}",
            (
                "三、綜合解讀：新聞題材可能是市場關注因素之一，但不能只因新聞與股價同期出現就認定直接因果。"
                f"目前新聞/PTT 情緒整體為「{sentiment['label']}」"
                f"(正向 {sentiment['正向']}、中立 {sentiment['中立']}、負向 {sentiment['負向']})。"
            ),
            "四、名詞補充：題材是市場近期關注的事件或敘事，不等於已實現的營收或獲利。",
            (
                "五、資料限制：若要正式判斷財報與未來題材，需要補充季報、月營收、法說會與公司公告。"
                "本段不構成投資建議，也不預測確切未來股價。"
            ),
        ]
    )


def _select_topic_items(items: list[NewsItem], question: str, limit: int = 5) -> list[NewsItem]:
    terms = [term for term in ("財報", "營收", "獲利", "EPS", "eps", "毛利", "題材", "AI", "電動車", "伺服器", "展望", "法說") if term in question]
    if not terms:
        terms = ["財報", "營收", "獲利", "題材", "AI", "電動車", "伺服器", "展望", "法說"]
    selected = []
    for item in items:
        text = f"{item.title} {item.content}"
        if any(term in text for term in terms):
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected or items[:limit]


def _shorten_context(value: str, max_chars: int) -> str:
    value = " ".join((value or "").split())
    if not value:
        return "未檢索到明確相關內容。"
    return value[:max_chars].rstrip() + ("..." if len(value) > max_chars else "")


def _format_optional_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "資料不足"
        return f"{float(value):.2f}%"
    except Exception:
        return "資料不足"


def _format_optional_number(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "資料不足"
        return f"{int(float(value)):,}"
    except Exception:
        return "資料不足"
