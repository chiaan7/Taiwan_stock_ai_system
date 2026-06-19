from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


STOCK_NAMES = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2303": "聯電",
    "2412": "中華電",
    "2881": "富邦金",
    "2882": "國泰金",
    "1301": "台塑",
    "1303": "南亞",
    "2308": "台達電",
}


@dataclass
class StockSnapshot:
    stock_id: str
    stock_name: str
    last_price: float | None
    previous_close: float | None
    change: float | None
    change_pct: float | None
    volume: int | None
    source: str


def normalize_stock_id(raw_value: str) -> str:
    value = str(raw_value).strip().upper()
    if value.endswith(".TW") or value.endswith(".TWO"):
        value = value.split(".")[0]
    return "".join(ch for ch in value if ch.isdigit()) or value


def get_stock_name(stock_id: str) -> str:
    return STOCK_NAMES.get(normalize_stock_id(stock_id), f"{normalize_stock_id(stock_id)}")


def to_yahoo_symbol(stock_id: str) -> str:
    stock_id = normalize_stock_id(stock_id)
    return stock_id if "." in stock_id else f"{stock_id}.TW"


def _read_json(url: str, timeout: int = 12, retries: int = 3, backoff: float = 0.8) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            )
        },
    )
    last_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    if last_error:
        raise last_error
    return {}


def fetch_price_history(stock_id: str, range_value: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo's public chart endpoint.

    This avoids making the app hard-dependent on yfinance at runtime. If the
    endpoint is unavailable, an empty DataFrame is returned and the UI can still
    show the rule-based interpretation with a clear limitation.
    """
    symbol = to_yahoo_symbol(stock_id)
    params = urlencode({"range": range_value, "interval": interval})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"

    try:
        payload = _read_json(url)
        result = payload["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        quote = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [])
    except Exception:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adjclose", "volume"])

    rows = []
    for index, ts in enumerate(timestamps):
        close = _safe_get(quote.get("close", []), index)
        rows.append(
            {
                "date": datetime.fromtimestamp(ts).date(),
                "open": _safe_get(quote.get("open", []), index),
                "high": _safe_get(quote.get("high", []), index),
                "low": _safe_get(quote.get("low", []), index),
                "close": close,
                "adjclose": _safe_get(adjclose, index) or close,
                "volume": _safe_get(quote.get("volume", []), index),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["close"])
    return df.reset_index(drop=True)


def _safe_get(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def build_snapshot(stock_id: str, history: pd.DataFrame | None = None) -> StockSnapshot:
    stock_id = normalize_stock_id(stock_id)
    history = history if history is not None else fetch_price_history(stock_id)
    if history.empty:
        return StockSnapshot(stock_id, get_stock_name(stock_id), None, None, None, None, None, "no data")

    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) >= 2 else latest
    last_price = _to_float(latest.get("close"))
    previous_close = _to_float(previous.get("close"))
    change = None if last_price is None or previous_close is None else last_price - previous_close
    change_pct = None if change is None or not previous_close else change / previous_close * 100
    volume = latest.get("volume")

    return StockSnapshot(
        stock_id=stock_id,
        stock_name=get_stock_name(stock_id),
        last_price=last_price,
        previous_close=previous_close,
        change=change,
        change_pct=change_pct,
        volume=None if pd.isna(volume) else int(volume),
        source="Yahoo Finance chart API",
    )


def fetch_institutional_trading(
    stock_id: str,
    lookback_days: int = 10,
    errors: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Fetch recent three-institution trading data.

    TWSE's table endpoint occasionally rejects direct requests. The crawler
    first tries the official table API, then falls back to FinMind's public
    institutional trading dataset so the project can still produce a useful
    chip analysis during demos.
    """
    stock_id = normalize_stock_id(stock_id)
    diagnostics: list[str] = []

    twse_data = _fetch_twse_institutional_trading(stock_id, lookback_days, diagnostics)
    if not twse_data.empty:
        return twse_data

    finmind_data = _fetch_finmind_institutional_trading(stock_id, lookback_days, diagnostics)
    if not finmind_data.empty:
        return finmind_data

    if errors is not None:
        detail = "；".join(diagnostics[:4]) if diagnostics else "所有資料來源皆無回傳可用資料"
        errors.append(
            {
                "source": "institutional_trading",
                "message": f"三大法人資料抓取失敗或無資料：{detail}",
                "url": "TWSE T86 / FinMind TaiwanStockInstitutionalInvestorsBuySell",
            }
        )
    return _empty_chip_dataframe()


def _fetch_twse_institutional_trading(
    stock_id: str,
    lookback_days: int,
    diagnostics: list[str],
) -> pd.DataFrame:
    """Fetch TWSE three-institution trading data for recent trading days."""
    stock_id = normalize_stock_id(stock_id)
    rows: list[dict[str, Any]] = []
    today = datetime.now()

    for offset in range(lookback_days * 2):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        params = urlencode(
            {
                "response": "json",
                "date": day.strftime("%Y%m%d"),
                "selectType": "ALLBUT0999",
            }
        )
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?{params}"
        try:
            payload = _read_json(url)
        except Exception as exc:
            if len(diagnostics) < 4:
                diagnostics.append(f"TWSE {day.strftime('%Y-%m-%d')} {type(exc).__name__}")
            continue

        fields = payload.get("fields", [])
        data = payload.get("data", [])
        for item in data:
            record = dict(zip(fields, item))
            if str(record.get("證券代號", "")).strip() == stock_id:
                rows.append(_normalize_chip_record(day.date(), record))
                break
        if len(rows) >= lookback_days:
            break

    if not rows:
        diagnostics.append("TWSE T86 未回傳目標股票資料")
        return _empty_chip_dataframe()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _fetch_finmind_institutional_trading(
    stock_id: str,
    lookback_days: int,
    diagnostics: list[str],
) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=max(45, lookback_days * 5))).strftime("%Y-%m-%d")
    params = urlencode(
        {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": stock_id,
            "start_date": start_date,
        }
    )
    url = f"https://api.finmindtrade.com/api/v4/data?{params}"
    try:
        payload = _read_json(url, timeout=20)
    except Exception as exc:
        diagnostics.append(f"FinMind {type(exc).__name__}")
        return _empty_chip_dataframe()

    if int(payload.get("status", 0) or 0) != 200:
        diagnostics.append(f"FinMind status={payload.get('status')} msg={payload.get('msg')}")
        return _empty_chip_dataframe()

    daily: dict[str, dict[str, int]] = {}
    for item in payload.get("data", []):
        if normalize_stock_id(str(item.get("stock_id", ""))) != stock_id:
            continue
        day = str(item.get("date", "")).strip()
        name = str(item.get("name", "")).strip()
        if not day:
            continue
        entry = daily.setdefault(
            day,
            {
                "foreign_net": 0,
                "investment_trust_net": 0,
                "dealer_net": 0,
            },
        )
        net = _parse_int(item.get("buy")) - _parse_int(item.get("sell"))
        if name == "Foreign_Investor":
            entry["foreign_net"] += net
        elif name == "Investment_Trust":
            entry["investment_trust_net"] += net
        elif name in {"Dealer_self", "Dealer_Hedging"}:
            entry["dealer_net"] += net

    rows: list[dict[str, Any]] = []
    for day in sorted(daily.keys())[-lookback_days:]:
        record = daily[day]
        try:
            parsed_day = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            continue
        rows.append(
            {
                "date": parsed_day,
                "foreign_net": record["foreign_net"],
                "investment_trust_net": record["investment_trust_net"],
                "dealer_net": record["dealer_net"],
                "total_net": (
                    record["foreign_net"]
                    + record["investment_trust_net"]
                    + record["dealer_net"]
                ),
            }
        )

    if not rows:
        diagnostics.append("FinMind 未回傳目標股票資料")
        return _empty_chip_dataframe()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _empty_chip_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "foreign_net", "investment_trust_net", "dealer_net", "total_net"]
    )


def _normalize_chip_record(day: datetime.date, record: dict[str, Any]) -> dict[str, Any]:
    foreign = _parse_int(record.get("外陸資買賣超股數(不含外資自營商)"))
    trust = _parse_int(record.get("投信買賣超股數"))
    dealer = _parse_int(record.get("自營商買賣超股數"))
    total = _parse_int(record.get("三大法人買賣超股數")) or (foreign + trust + dealer)
    return {
        "date": day,
        "foreign_net": foreign,
        "investment_trust_net": trust,
        "dealer_net": dealer,
        "total_net": total,
    }


def _parse_int(value: Any) -> int:
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return 0


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def summarize_price(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {"trend": "資料不足", "return_pct": None, "volatility": None}
    start = float(history.iloc[0]["close"])
    end = float(history.iloc[-1]["close"])
    return_pct = (end - start) / start * 100 if start else None
    volatility = float(history["close"].pct_change().std() * 100) if len(history) > 2 else None
    if return_pct is None:
        trend = "資料不足"
    elif return_pct > 3:
        trend = "偏多"
    elif return_pct < -3:
        trend = "偏空"
    else:
        trend = "中立"
    return {"trend": trend, "return_pct": return_pct, "volatility": volatility}
