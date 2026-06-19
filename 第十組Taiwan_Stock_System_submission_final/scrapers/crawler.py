from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scrapers.news_spider import (
    NewsItem,
    build_rag_documents,
    fetch_google_news,
    fetch_ptt_stock_posts,
    fetch_yahoo_news,
    fetch_yahoo_news_scroll,
    summarize_sentiment,
)
from scrapers.yfinance_client import (
    StockSnapshot,
    build_snapshot,
    fetch_institutional_trading,
    fetch_price_history,
    get_stock_name,
    normalize_stock_id,
    summarize_price,
)


DEFAULT_OUTPUT_DIR = Path("data/raw_data")
DEMO_OUTPUT_DIR = Path("data/demo")
DEMO_STOCK_ID = "2330"
EMPTY_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adjclose", "volume"]
EMPTY_CHIP_COLUMNS = ["date", "foreign_net", "investment_trust_net", "dealer_net", "total_net"]


def crawl_stock(
    stock_id: str,
    stock_name: str | None = None,
    news_keyword: str | None = None,
    price_range: str = "3mo",
    chip_days: int = 10,
    news_limit: int = 50,
    ptt_pages: int = 80,
    days: int = 7,
    news_source: str = "yahoo",
    yahoo_mode: str = "rss",
    yahoo_scroll_rounds: int = 20,
    max_content_chars: int = 6000,
    strict_dates: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save: bool = True,
) -> dict[str, Any]:
    """Run the crawler pipeline for one Taiwan stock."""
    stock_id = normalize_stock_id(stock_id)
    stock_name = stock_name or get_stock_name(stock_id)
    news_keyword = news_keyword or stock_name or stock_id
    crawled_at = datetime.now().isoformat(timespec="seconds")
    output_root = Path(output_dir) / stock_id
    demo_root = DEMO_OUTPUT_DIR / DEMO_STOCK_ID
    allow_demo = stock_id == DEMO_STOCK_ID
    errors: list[dict[str, str]] = []
    source_status: list[dict[str, Any]] = []
    progress = ProgressReporter()

    progress.phase(1, 4, "Yahoo Finance price")
    print(f"[1/4] Fetching {price_range} Yahoo Finance price history for {stock_id} {stock_name}...")
    price_error = ""
    try:
        new_history = fetch_price_history(stock_id, range_value=price_range)
    except Exception as exc:
        new_history = pd.DataFrame(columns=EMPTY_PRICE_COLUMNS)
        price_error = f"{type(exc).__name__}: {exc}"
        errors.append({"source": "price", "message": price_error, "url": "Yahoo Finance chart API"})
    history, status = _choose_dataframe_source(
        "price",
        new_history,
        output_root / "price_history.csv",
        demo_root / "price_history.csv",
        EMPTY_PRICE_COLUMNS,
        _valid_price_data,
        price_error or "股價資料為空或必要欄位不足",
        allow_demo,
    )
    source_status.append(status)
    snapshot = build_snapshot(stock_id, history)
    progress.done("Yahoo Finance price", f"{len(history)} rows")

    progress.phase(2, 4, "TWSE institutional trading")
    print(f"[2/4] Fetching TWSE institutional trading for {stock_id}...")
    chip_error_start = len(errors)
    try:
        new_chip_data = fetch_institutional_trading(stock_id, lookback_days=chip_days, errors=errors)
    except Exception as exc:
        new_chip_data = pd.DataFrame(columns=EMPTY_CHIP_COLUMNS)
        errors.append({"source": "institutional_trading", "message": f"{type(exc).__name__}: {exc}", "url": "TWSE/FinMind"})
    chip_error = _latest_error_message(errors, chip_error_start) or "三大法人資料為空或必要欄位不足"
    chip_data, status = _choose_dataframe_source(
        "institutional_trading",
        new_chip_data,
        output_root / "institutional_trading.csv",
        demo_root / "institutional_trading.csv",
        EMPTY_CHIP_COLUMNS,
        _valid_chip_data,
        chip_error,
        allow_demo,
    )
    source_status.append(status)
    progress.done("TWSE institutional trading", f"{len(chip_data)} rows")

    progress.phase(3, 4, f"News ({news_source})")
    months_for_yahoo = max(1, (days + 30) // 31)

    print(f"[3/4] Fetching news from the last {days} days for keyword: {news_keyword}")
    news_error_start = len(errors)
    try:
        new_news_items = _fetch_news_by_source(
            stock_id=stock_id,
            keyword=news_keyword,
            limit=news_limit,
            days=days,
            months=months_for_yahoo,
            source=news_source,
            mode=yahoo_mode,
            scroll_rounds=yahoo_scroll_rounds,
            progress=progress.update,
            errors=errors,
        )
    except Exception as exc:
        new_news_items = []
        errors.append({"source": "news", "message": f"{type(exc).__name__}: {exc}", "url": news_source})
    new_news_items = _postprocess_items(
        new_news_items,
        max_content_chars=max_content_chars,
        strict_dates=strict_dates,
        stock_id=stock_id,
        crawl_time=crawled_at,
    )
    news_error = _latest_error_message(errors, news_error_start) or "新聞資料為空或必要欄位不足"
    news_items, status = _choose_items_source(
        "news",
        new_news_items,
        output_root / "news.json",
        demo_root,
        _valid_news_items,
        news_error,
        allow_demo,
    )
    source_status.append(status)
    progress.done("News", f"{len(news_items)} items")

    progress.phase(4, 4, "PTT Stock")
    print(f"[4/4] Fetching PTT Stock posts from the last {days} days for keyword: {news_keyword}")
    ptt_error_start = len(errors)
    try:
        new_ptt_items = fetch_ptt_stock_posts(
            news_keyword,
            stock_id=stock_id,
            max_pages=ptt_pages,
            days=days,
            content_search=True,
            progress=progress.update,
            errors=errors,
        )
    except Exception as exc:
        new_ptt_items = []
        errors.append({"source": "ptt", "message": f"{type(exc).__name__}: {exc}", "url": "PTT Stock"})
    new_ptt_items = _postprocess_items(
        new_ptt_items,
        max_content_chars=max_content_chars,
        strict_dates=strict_dates,
        stock_id=stock_id,
        crawl_time=crawled_at,
    )
    ptt_error = _latest_error_message(errors, ptt_error_start) or "PTT 資料為空、無正文或明顯無關"
    ptt_items, status = _choose_items_source(
        "ptt",
        new_ptt_items,
        output_root / "ptt_posts.json",
        demo_root,
        _valid_ptt_items,
        ptt_error,
        allow_demo,
    )
    source_status.append(status)
    progress.done("PTT Stock", f"{len(ptt_items)} items")

    rag_documents = build_rag_documents(stock_id, stock_name, news_items, ptt_items)

    result = {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "news_keyword": news_keyword,
        "crawled_at": crawled_at,
        "crawl_config": {
            "price_range": price_range,
            "days": days,
            "news_source": news_source,
            "yahoo_month_window": months_for_yahoo,
            "chip_days": chip_days,
            "news_limit": news_limit,
            "ptt_pages": ptt_pages,
            "ptt_content_search": True,
            "yahoo_mode": yahoo_mode,
            "yahoo_scroll_rounds": yahoo_scroll_rounds,
            "max_content_chars": max_content_chars,
            "strict_dates": strict_dates,
        },
        "snapshot": asdict(snapshot),
        "price_summary": summarize_price(history),
        "sentiment_summary": summarize_sentiment(news_items + ptt_items),
        "errors": errors,
        "source_status": source_status,
        "counts": {
            "price_rows": int(len(history)),
            "chip_rows": int(len(chip_data)),
            "news_items": int(len(news_items)),
            "ptt_items": int(len(ptt_items)),
            "rag_documents": int(len(rag_documents)),
            "errors": int(len(errors)),
            "undated_news_items": _count_undated(news_items),
            "undated_ptt_items": _count_undated(ptt_items),
        },
        "data": {
            "price_history": history,
            "institutional_trading": chip_data,
            "news": news_items,
            "ptt": ptt_items,
            "rag_documents": rag_documents,
        },
    }

    if save:
        saved_files = save_crawl_result(result, output_dir)
        result["saved_files"] = [str(path) for path in saved_files]

    return result


def crawl_keyword(
    keyword: str,
    days: int = 7,
    news_source: str = "yahoo",
    news_limit: int = 50,
    ptt_pages: int = 80,
    yahoo_mode: str = "rss",
    yahoo_scroll_rounds: int = 20,
    max_content_chars: int = 6000,
    strict_dates: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save: bool = True,
) -> dict[str, Any]:
    """Collect market news and PTT posts by keyword without stock price/chip data."""
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword cannot be empty")
    keyword_id = _safe_output_name(f"keyword_{keyword}")
    crawled_at = datetime.now().isoformat(timespec="seconds")
    progress = ProgressReporter()
    errors: list[dict[str, str]] = []
    months_for_yahoo = max(1, (days + 30) // 31)

    progress.phase(1, 2, f"News ({news_source})")
    print(f"[1/2] Fetching news from the last {days} days for keyword: {keyword}")
    news_items = _fetch_news_by_source(
        stock_id="",
        keyword=keyword,
        limit=news_limit,
        days=days,
        months=months_for_yahoo,
        source=news_source,
        mode="rss" if yahoo_mode == "scroll" else yahoo_mode,
        scroll_rounds=yahoo_scroll_rounds,
        progress=progress.update,
        errors=errors,
    )
    news_items = _postprocess_items(
        news_items,
        max_content_chars=max_content_chars,
        strict_dates=strict_dates,
        stock_id=keyword_id,
        crawl_time=crawled_at,
    )
    progress.done("News", f"{len(news_items)} items")

    progress.phase(2, 2, "PTT Stock")
    print(f"[2/2] Fetching PTT Stock posts from the last {days} days for keyword: {keyword}")
    ptt_items = fetch_ptt_stock_posts(
        keyword,
        stock_id="",
        max_pages=ptt_pages,
        days=days,
        content_search=True,
        progress=progress.update,
        errors=errors,
    )
    ptt_items = _postprocess_items(
        ptt_items,
        max_content_chars=max_content_chars,
        strict_dates=strict_dates,
        stock_id=keyword_id,
        crawl_time=crawled_at,
    )
    progress.done("PTT Stock", f"{len(ptt_items)} items")

    price_history = pd.DataFrame(columns=EMPTY_PRICE_COLUMNS)
    chip_data = pd.DataFrame(columns=EMPTY_CHIP_COLUMNS)
    rag_documents = build_rag_documents(keyword_id, keyword, news_items, ptt_items)

    result = {
        "stock_id": keyword_id,
        "stock_name": keyword,
        "news_keyword": keyword,
        "crawled_at": crawled_at,
        "crawl_config": {
            "keyword_only": True,
            "days": days,
            "news_source": news_source,
            "yahoo_month_window": months_for_yahoo,
            "news_limit": news_limit,
            "ptt_pages": ptt_pages,
            "ptt_content_search": True,
            "yahoo_mode": yahoo_mode,
            "yahoo_scroll_rounds": yahoo_scroll_rounds,
            "max_content_chars": max_content_chars,
            "strict_dates": strict_dates,
        },
        "snapshot": asdict(
            StockSnapshot(keyword_id, keyword, None, None, None, None, None, "keyword mode")
        ),
        "price_summary": summarize_price(price_history),
        "sentiment_summary": summarize_sentiment(news_items + ptt_items),
        "errors": errors,
        "counts": {
            "price_rows": 0,
            "chip_rows": 0,
            "news_items": int(len(news_items)),
            "ptt_items": int(len(ptt_items)),
            "rag_documents": int(len(rag_documents)),
            "errors": int(len(errors)),
            "undated_news_items": _count_undated(news_items),
            "undated_ptt_items": _count_undated(ptt_items),
        },
        "data": {
            "price_history": price_history,
            "institutional_trading": chip_data,
            "news": news_items,
            "ptt": ptt_items,
            "rag_documents": rag_documents,
        },
    }

    if save:
        saved_files = save_crawl_result(result, output_dir)
        result["saved_files"] = [str(path) for path in saved_files]

    return result


def save_crawl_result(result: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_root = Path(output_dir) / str(result["stock_id"])
    output_root.mkdir(parents=True, exist_ok=True)

    history: pd.DataFrame = result["data"]["price_history"]
    chip_data: pd.DataFrame = result["data"]["institutional_trading"]
    news_items: list[NewsItem] = result["data"]["news"]
    ptt_items: list[NewsItem] = result["data"]["ptt"]
    rag_documents: list[dict[str, Any]] = result["data"]["rag_documents"]
    status_map = {item.get("source"): item for item in result.get("source_status", []) if isinstance(item, dict)}

    paths = [
        output_root / "price_history.csv",
        output_root / "institutional_trading.csv",
        output_root / "news.json",
        output_root / "news_text.jsonl",
        output_root / "ptt_posts.json",
        output_root / "rag_documents.jsonl",
        output_root / "crawl_errors.json",
        output_root / "crawl_summary.json",
    ]

    written_paths: list[Path] = []
    if _should_write_source(status_map, "price", history):
        _save_dataframe(history, paths[0])
        written_paths.append(paths[0])
    if _should_write_source(status_map, "institutional_trading", chip_data):
        _save_dataframe(chip_data, paths[1])
        written_paths.append(paths[1])
    if _should_write_source(status_map, "news", news_items):
        _write_json([asdict(item) for item in news_items], paths[2])
        _write_jsonl(_build_news_text_rows(news_items), paths[3])
        written_paths.extend([paths[2], paths[3]])
    if _should_write_source(status_map, "ptt", ptt_items):
        _write_json([asdict(item) for item in ptt_items], paths[4])
        written_paths.append(paths[4])
    if rag_documents and (not status_map or any(_is_fresh_success(status_map, source) for source in ("news", "ptt"))):
        _write_jsonl(rag_documents, paths[5])
        written_paths.append(paths[5])
    _write_json(result.get("errors", []), paths[6])
    written_paths.append(paths[6])

    summary = {key: value for key, value in result.items() if key not in {"data", "saved_files"}}
    _write_json(summary, paths[7])
    written_paths.append(paths[7])
    return written_paths


def print_crawl_report(result: dict[str, Any]) -> None:
    snapshot = result["snapshot"]
    counts = result["counts"]
    sentiment = result["sentiment_summary"]

    print("\n=== Crawl Result ===")
    print(f"Stock: {result['stock_id']} {result['stock_name']}")
    print(f"News keyword: {result['news_keyword']}")
    print(f"Price rows: {counts['price_rows']}")
    print(f"Chip rows: {counts['chip_rows']}")
    print(f"News items: {counts['news_items']}")
    print(f"PTT items: {counts['ptt_items']}")
    print(f"RAG documents: {counts['rag_documents']}")
    print(f"Errors: {counts['errors']}")
    print(f"Undated news/PTT: {counts['undated_news_items']} / {counts['undated_ptt_items']}")
    print(f"Last price: {_format_number(snapshot.get('last_price'))}")
    print(f"Change pct: {_format_number(snapshot.get('change_pct'))}%")
    print(f"Sentiment: {sentiment['label']}")
    if result.get("saved_files"):
        print("\nSaved files:")
        for path in result["saved_files"]:
            print(f"- {path}")


def write_analysis_report(
    result: dict[str, Any],
    question: str | None = None,
    prefer_llm: bool = True,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    use_rag: bool = True,
    rag_top_k: int = 5,
) -> Path:
    from ai_core.analysis_pipeline import analyze_result

    output_root = Path(output_dir) / str(result["stock_id"])
    artifacts = analyze_result(
        result,
        question=question,
        prefer_llm=prefer_llm,
        output_root=output_root,
        use_rag=use_rag,
        rag_top_k=rag_top_k,
    )
    return artifacts.report_path or output_root / "analysis_report.txt"


def write_rag_index(result: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    from ai_core.rag_engine import build_rag_index, load_static_knowledge

    output_root = Path(output_dir) / str(result["stock_id"])
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "rag_index.json"
    build_rag_index([*result["data"]["rag_documents"], *load_static_knowledge()], path)
    return path


def resolve_stock_ids(args: argparse.Namespace) -> list[str]:
    raw_values: list[str] = []
    if args.stocks:
        raw_values.extend(args.stocks.split(","))
    if args.stocks_file:
        raw_values.extend(Path(args.stocks_file).read_text(encoding="utf-8").splitlines())
    if args.stock_option:
        raw_values.append(args.stock_option)
    if args.stock_id:
        raw_values.append(args.stock_id)

    if not raw_values:
        typed = input("請輸入股票代號，可用逗號分隔多檔，例如 2330,2317：").strip()
        raw_values.extend(typed.split(","))

    stocks: list[str] = []
    for value in raw_values:
        value = value.strip()
        if not value or value.startswith("#"):
            continue
        stock_id = normalize_stock_id(value)
        if stock_id and stock_id not in stocks:
            stocks.append(stock_id)

    if not stocks:
        raise SystemExit("沒有可爬取的股票代號。")
    return stocks


def _safe_output_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned.strip("_") or "keyword"


def _source_status(
    source: str,
    status: str,
    record_count: int,
    error_message: str = "",
    used_cache: bool = False,
    used_demo: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "record_count": int(record_count),
        "source": source,
        "error_message": error_message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "used_cache": bool(used_cache),
        "used_demo": bool(used_demo),
    }


def _valid_price_data(df: pd.DataFrame) -> bool:
    return not df.empty and {"date", "close"}.issubset(df.columns) and df["close"].notna().any()


def _valid_chip_data(df: pd.DataFrame) -> bool:
    required = {"date", "foreign_net", "investment_trust_net", "dealer_net", "total_net"}
    return not df.empty and required.issubset(df.columns)


def _valid_news_items(items: list[NewsItem]) -> bool:
    return any(item.title and item.source for item in items)


def _valid_ptt_items(items: list[NewsItem]) -> bool:
    return any(item.title and item.source == "PTT Stock" and item.content for item in items)


def _read_dataframe(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame(columns=columns)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)


def _read_news_items(path: Path) -> list[NewsItem]:
    try:
        if not path.exists():
            return []
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
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


def _read_demo_items_from_rag(root: Path, source_name: str) -> list[NewsItem]:
    path = root / "rag_documents.jsonl"
    if not path.exists():
        return []
    items: list[NewsItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if source_name == "ptt" and row.get("source") != "PTT Stock":
            continue
        if source_name == "news" and row.get("source") == "PTT Stock":
            continue
        items.append(
            NewsItem(
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
                source=str(row.get("source", "")),
                summary=str(row.get("summary", "")) or str(row.get("text", ""))[:180],
                published_at=str(row.get("published_at", "")) or None,
                display_date=str(row.get("display_date", "")),
                sentiment=str(row.get("sentiment", "中立")),
                content=str(row.get("text", "")),
                keyword=str(row.get("stock_name", "")),
                stock_id=str(row.get("stock_id", "")),
                crawl_time=datetime.now().isoformat(timespec="seconds"),
            )
        )
    return items


def _choose_dataframe_source(
    source: str,
    new_data: pd.DataFrame,
    cache_path: Path,
    demo_path: Path,
    columns: list[str],
    validator,
    error_message: str,
    allow_demo: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if validator(new_data):
        return new_data, _source_status(source, "success", len(new_data))

    cached = _read_dataframe(cache_path, columns)
    if validator(cached):
        return cached, _source_status(source, "cache", len(cached), error_message, used_cache=True)

    demo = _read_dataframe(demo_path, columns) if allow_demo else pd.DataFrame(columns=columns)
    if validator(demo):
        return demo, _source_status(source, "demo", len(demo), error_message, used_demo=True)

    return pd.DataFrame(columns=columns), _source_status(source, "failed", 0, error_message)


def _choose_items_source(
    source: str,
    new_items: list[NewsItem],
    cache_path: Path,
    demo_root: Path,
    validator,
    error_message: str,
    allow_demo: bool,
) -> tuple[list[NewsItem], dict[str, Any]]:
    if validator(new_items):
        return new_items, _source_status(source, "success", len(new_items))

    cached = _read_news_items(cache_path)
    if validator(cached):
        return cached, _source_status(source, "cache", len(cached), error_message, used_cache=True)

    demo = _read_demo_items_from_rag(demo_root, source) if allow_demo else []
    if validator(demo):
        return demo, _source_status(source, "demo", len(demo), error_message, used_demo=True)

    return [], _source_status(source, "failed", 0, error_message)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _is_fresh_success(status_map: dict[str, dict[str, Any]], source: str) -> bool:
    if not status_map:
        return True
    status = status_map.get(source, {})
    return status.get("status") == "success" and not status.get("used_cache") and not status.get("used_demo")


def _should_write_source(status_map: dict[str, dict[str, Any]], source: str, data: Any) -> bool:
    if not _is_fresh_success(status_map, source):
        return False
    try:
        return len(data) > 0
    except Exception:
        return bool(data)


def _build_news_text_rows(items: list[NewsItem]) -> list[dict[str, str]]:
    return [
        {
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "summary": item.summary,
            "published_at": item.published_at or "",
            "stock_id": item.stock_id,
            "crawl_time": item.crawl_time,
            "content": item.content,
        }
        for item in items
    ]


def _postprocess_items(
    items: list[NewsItem],
    max_content_chars: int,
    strict_dates: bool,
    stock_id: str = "",
    crawl_time: str = "",
) -> list[NewsItem]:
    processed: list[NewsItem] = []
    for item in items:
        if strict_dates and not item.published_at:
            continue
        if max_content_chars > 0 and item.content and len(item.content) > max_content_chars:
            item.content = item.content[:max_content_chars].rstrip()
        item.stock_id = item.stock_id or stock_id
        item.crawl_time = item.crawl_time or crawl_time or datetime.now().isoformat(timespec="seconds")
        if not item.summary:
            item.summary = (item.content or item.title)[:180].rstrip()
        processed.append(item)
    return processed


def _count_undated(items: list[NewsItem]) -> int:
    return sum(1 for item in items if not item.published_at)


def _latest_error_message(errors: list[dict[str, str]], start_index: int) -> str:
    if len(errors) <= start_index:
        return ""
    latest = errors[-1]
    return latest.get("message") or latest.get("error") or str(latest)


def _fetch_news_by_source(
    stock_id: str,
    keyword: str,
    limit: int,
    days: int,
    months: int,
    source: str,
    mode: str,
    scroll_rounds: int,
    progress,
    errors: list[dict[str, str]],
) -> list[NewsItem]:
    source = source.lower().strip()
    if source not in {"yahoo", "google", "both"}:
        raise ValueError("--news-source must be one of: yahoo, google, both")

    items: list[NewsItem] = []
    if source in {"yahoo", "both"}:
        items.extend(
            _fetch_yahoo_by_mode(
                stock_id=stock_id,
                stock_name=keyword,
                limit=limit,
                months=months,
                mode=mode,
                scroll_rounds=scroll_rounds,
                progress=progress,
                errors=errors,
            )
        )

    if source in {"google", "both"} and len(items) < limit:
        items.extend(
            fetch_google_news(
                keyword,
                limit=limit - len(items),
                days=days,
                progress=progress,
                errors=errors,
            )
        )

    return items[:limit]


def _fetch_yahoo_by_mode(
    stock_id: str,
    stock_name: str,
    limit: int,
    months: int,
    mode: str,
    scroll_rounds: int,
    progress,
    errors: list[dict[str, str]],
) -> list[NewsItem]:
    mode = mode.lower().strip()
    if mode not in {"rss", "scroll", "auto"}:
        raise ValueError("--yahoo-mode must be one of: rss, scroll, auto")

    if mode == "scroll":
        return fetch_yahoo_news_scroll(
            stock_id,
            stock_name,
            limit=limit,
            months=months,
            scroll_rounds=scroll_rounds,
            progress=progress,
            errors=errors,
        )

    items = fetch_yahoo_news(stock_id, stock_name, limit=limit, months=months, errors=errors, progress=progress)
    if mode == "auto" and len(items) < limit:
        scroll_items = fetch_yahoo_news_scroll(
            stock_id,
            stock_name,
            limit=limit - len(items),
            months=months,
            scroll_rounds=scroll_rounds,
            progress=progress,
            errors=errors,
        )
        items.extend(scroll_items)
    return items


class ProgressReporter:
    def phase(self, current: int, total: int, label: str) -> None:
        self.update(current, total, label)

    def update(self, current: int, total: int, label: str) -> None:
        total = max(total, 1)
        current = min(max(current, 0), total)
        width = 24
        filled = round(width * current / total)
        bar = "#" * filled + "-" * (width - filled)
        percent = current / total * 100
        print(f"[{bar}] {percent:5.1f}% {label}")

    def done(self, label: str, detail: str = "") -> None:
        suffix = f" - {detail}" if detail else ""
        print(f"[OK] {label}{suffix}")


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taiwan stock market crawler")
    parser.add_argument("stock_id", nargs="?", default=None, help="Taiwan stock id, for example 2330")
    parser.add_argument("--stock", dest="stock_option", default=None, help="Taiwan stock id, same as positional stock_id")
    parser.add_argument("--stocks", default=None, help="Comma-separated stock ids, for example 2330,2317,2454")
    parser.add_argument("--stocks-file", default=None, help="Text file containing one stock id per line")
    parser.add_argument("--name", dest="stock_name", default=None, help="Stock name, for example 台積電")
    parser.add_argument("--news-keyword", default=None, help="News/PTT search keyword. Defaults to stock name.")
    parser.add_argument("--keyword-only", default=None, help="Collect news/PTT by keyword without stock price/chip data")
    parser.add_argument("--news-source", choices=["yahoo", "google", "both"], default="yahoo", help="News source to crawl")
    parser.add_argument("--range", dest="price_range", default="3mo", help="Yahoo price range: 1mo, 3mo, 6mo, 1y")
    parser.add_argument("--days", type=int, default=7, help="How many recent days to collect news/PTT data")
    parser.add_argument("--months", type=int, default=None, help="Compatibility option. Converted to days as months * 31.")
    parser.add_argument("--chip-days", type=int, default=10, help="TWSE institutional trading rows to fetch")
    parser.add_argument("--news-limit", type=int, default=50, help="Yahoo news item limit")
    parser.add_argument("--ptt-pages", type=int, default=80, help="PTT pages to scan")
    parser.add_argument(
        "--yahoo-mode",
        choices=["rss", "scroll", "auto"],
        default="rss",
        help="Yahoo news strategy: rss is fast, scroll simulates infinite scroll, auto tries rss then scroll",
    )
    parser.add_argument("--yahoo-scroll-rounds", type=int, default=20, help="Scroll rounds for --yahoo-mode scroll/auto")
    parser.add_argument("--max-content-chars", type=int, default=6000, help="Max article content characters kept per item")
    parser.add_argument("--strict-dates", action="store_true", help="Drop news/PTT items without a parsed published date")
    parser.add_argument("--build-rag", action="store_true", help="Build rag_index.json from rag_documents.jsonl")
    parser.add_argument("--rag-top-k", type=int, default=5, help="Number of RAG documents used in analysis")
    parser.add_argument("--analyze", action="store_true", help="Write analysis_report.txt after crawling")
    parser.add_argument("--question", default=None, help="Analysis question. Used with --analyze.")
    parser.add_argument("--no-llm", action="store_true", help="Force rule-based analysis even if API keys exist")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Run crawler without writing files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.months is not None:
        args.days = args.months * 31

    if args.keyword_only:
        result = crawl_keyword(
            args.keyword_only,
            days=args.days,
            news_source=args.news_source,
            news_limit=args.news_limit,
            ptt_pages=args.ptt_pages,
            yahoo_mode=args.yahoo_mode,
            yahoo_scroll_rounds=args.yahoo_scroll_rounds,
            max_content_chars=args.max_content_chars,
            strict_dates=args.strict_dates,
            output_dir=args.output,
            save=not args.no_save,
        )
        print_crawl_report(result)
        if args.build_rag and not args.no_save:
            index_path = write_rag_index(result, args.output)
            print(f"\nRAG index: {index_path}")
        if args.analyze:
            report_path = write_analysis_report(
                result,
                args.question,
                prefer_llm=not args.no_llm,
                output_dir=args.output,
                use_rag=True,
                rag_top_k=args.rag_top_k,
            )
            print(f"\nAnalysis report: {report_path}")
            print(f"RAG index: {Path(args.output) / str(result['stock_id']) / 'rag_index.json'}")
            print(f"RAG retrieval: {Path(args.output) / str(result['stock_id']) / 'rag_retrieval.json'}")
        return

    stocks = resolve_stock_ids(args)

    if len(stocks) > 1 and args.stock_name:
        print("[WARN] --name is ignored when crawling multiple stocks; stock names are resolved automatically.")

    for index, stock_id in enumerate(stocks, start=1):
        if len(stocks) > 1:
            print(f"\n=== Stock {index}/{len(stocks)}: {stock_id} ===")
        result = crawl_stock(
            stock_id,
            stock_name=args.stock_name if len(stocks) == 1 else None,
            news_keyword=args.news_keyword,
            price_range=args.price_range,
            chip_days=args.chip_days,
            news_limit=args.news_limit,
            ptt_pages=args.ptt_pages,
            days=args.days,
            news_source=args.news_source,
            yahoo_mode=args.yahoo_mode,
            yahoo_scroll_rounds=args.yahoo_scroll_rounds,
            max_content_chars=args.max_content_chars,
            strict_dates=args.strict_dates,
            output_dir=args.output,
            save=not args.no_save,
        )
        print_crawl_report(result)
        if args.build_rag and not args.no_save:
            index_path = write_rag_index(result, args.output)
            print(f"\nRAG index: {index_path}")
        if args.analyze:
            report_path = write_analysis_report(
                result,
                args.question,
                prefer_llm=not args.no_llm,
                output_dir=args.output,
                use_rag=True,
                rag_top_k=args.rag_top_k,
            )
            print(f"\nAnalysis report: {report_path}")
            print(f"RAG index: {Path(args.output) / str(result['stock_id']) / 'rag_index.json'}")
            print(f"RAG retrieval: {Path(args.output) / str(result['stock_id']) / 'rag_retrieval.json'}")


if __name__ == "__main__":
    main()
