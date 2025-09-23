from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="Stockly Market API", version="0.1.0")


@app.get("/health", summary="Worker status check")
async def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/v7/finance/quote", summary="Batch quote lookup")
async def get_quotes(symbols: str = Query(..., description="Comma separated ticker symbols")) -> Dict[str, Any]:
    tickers = [sym.strip() for sym in symbols.split(",") if sym.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results = await run_in_threadpool(_load_quotes, tickers)
    return {"quoteResponse": {"result": results, "error": None}}


@app.get("/v1/finance/search", summary="Symbol/company search")
async def search_quotes(
    q: str = Query(..., description="Search term"),
    quotesCount: int = Query(10, ge=1, le=25, description="Max number of quote matches")
) -> Dict[str, Any]:
    if not q:
        return {"quotes": [], "news": [], "total": 0}

    payload = await run_in_threadpool(_search_quotes, q, quotesCount)
    return payload


@app.get("/v10/finance/quoteSummary/{symbol}", summary="Detailed quote summary")
async def quote_summary(symbol: str, modules: str = Query("", description="Comma separated modules")) -> Dict[str, Any]:
    result = await run_in_threadpool(_load_summary, symbol, modules)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")
    return {"quoteSummary": {"result": [result], "error": None}}


@app.get("/v8/finance/chart/{symbol}", summary="Historical chart data")
async def chart_data(
    symbol: str,
    range: Optional[str] = Query(None, alias="range"),
    interval: Optional[str] = Query(None)
) -> Dict[str, Any]:
    payload = await run_in_threadpool(_load_chart, symbol, range, interval)
    return {"chart": {"result": payload, "error": None}}


def _load_quotes(symbols: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.get_info()
        except Exception:  # pragma: no cover - yfinance failure
            continue

        results.append(
            {
                "symbol": symbol,
                "longName": info.get("longName"),
                "shortName": info.get("shortName"),
                "regularMarketPrice": info.get("regularMarketPrice"),
                "regularMarketChange": info.get("regularMarketChange"),
                "regularMarketChangePercent": info.get("regularMarketChangePercent"),
                "marketCap": info.get("marketCap"),
                "regularMarketVolume": info.get("regularMarketVolume"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "currency": info.get("currency"),
                "exchangeName": info.get("fullExchangeName") or info.get("exchange"),
                "regularMarketDayHigh": info.get("regularMarketDayHigh"),
                "regularMarketDayLow": info.get("regularMarketDayLow"),
                "trailingPE": info.get("trailingPE"),
            }
        )

    return results


def _search_quotes(query: str, limit: int) -> Dict[str, Any]:
    if not hasattr(yf, "search"):
        raise RuntimeError("yfinance.search is unavailable in this environment")

    payload = yf.search(query)
    quotes = payload.get("quotes", [])[:limit]
    news = payload.get("news", [])
    total = payload.get("total", len(quotes))
    return {
        "quotes": [
            {
                "symbol": item.get("symbol"),
                "shortname": item.get("shortname"),
                "longname": item.get("longname"),
                "exchDisp": item.get("exchDisp"),
                "typeDisp": item.get("typeDisp"),
                "quoteType": item.get("quoteType"),
            }
            for item in quotes
        ],
        "news": news,
        "total": total,
    }


def _load_summary(symbol: str, modules: str) -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()
    except Exception:  # pragma: no cover
        return None

    def wrap(value: Optional[Any]) -> Optional[Dict[str, Any]]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        fmt: str
        if isinstance(value, float):
            fmt = f"{value:,.2f}"
        elif isinstance(value, int):
            fmt = f"{value:,}"
        else:
            fmt = str(value)
        return {"raw": value, "fmt": fmt}

    sections: Dict[str, Optional[Dict[str, Any]]] = {
        "price": {
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
            "currency": info.get("currency"),
            "exchangeName": info.get("fullExchangeName") or info.get("exchange"),
            "regularMarketPrice": wrap(info.get("regularMarketPrice")),
            "regularMarketChange": wrap(info.get("regularMarketChange")),
            "regularMarketChangePercent": wrap(info.get("regularMarketChangePercent")),
        },
        "summaryProfile": {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "longBusinessSummary": info.get("longBusinessSummary"),
        },
        "summaryDetail": {
            "fiftyTwoWeekHigh": wrap(info.get("fiftyTwoWeekHigh")),
            "fiftyTwoWeekLow": wrap(info.get("fiftyTwoWeekLow")),
            "marketCap": wrap(info.get("marketCap")),
            "trailingPE": wrap(info.get("trailingPE")),
            "regularMarketVolume": wrap(info.get("regularMarketVolume")),
            "volume": wrap(info.get("volume")),
        },
        "defaultKeyStatistics": {
            "enterpriseValue": wrap(info.get("enterpriseValue")),
            "forwardPE": wrap(info.get("forwardPE")),
            "pegRatio": wrap(info.get("pegRatio")),
        },
        "financialData": {
            "targetMeanPrice": wrap(info.get("targetMeanPrice")),
            "totalCash": wrap(info.get("totalCash")),
        },
    }

    def prune(section: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if section is None:
            return None
        cleaned = {k: v for k, v in section.items() if v is not None}
        return cleaned or None

    result = {key: prune(value) for key, value in sections.items() if prune(value) is not None}

    requested_modules = {module.strip() for module in modules.split(",") if module.strip()}
    if requested_modules:
        result = {key: val for key, val in result.items() if key in requested_modules}

    return result


def _load_chart(symbol: str, range_param: Optional[str], interval: Optional[str]) -> List[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=range_param or "1mo", interval=interval or "1d", auto_adjust=False)
        info = ticker.get_info()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if hist.empty:
        return []

    hist = hist.fillna(0)
    timestamps = [int(pd.Timestamp(ts).timestamp()) for ts in hist.index]
    payload = {
        "meta": {
            "currency": info.get("currency"),
            "symbol": symbol,
            "regularMarketPrice": info.get("regularMarketPrice"),
            "chartPreviousClose": info.get("previousClose"),
            "exchangeName": info.get("fullExchangeName") or info.get("exchange"),
            "dataGranularity": interval or "1d",
        },
        "timestamp": timestamps,
        "indicators": {
            "quote": [
                {
                    "close": _series_to_list(hist.get("Close")),
                    "open": _series_to_list(hist.get("Open")),
                    "high": _series_to_list(hist.get("High")),
                    "low": _series_to_list(hist.get("Low")),
                    "volume": _series_to_list(hist.get("Volume"), cast_to_int=True),
                }
            ]
        },
    }

    return [payload]


def _series_to_list(series: Optional[pd.Series], cast_to_int: bool = False) -> List[Any]:
    if series is None:
        return []
    values: List[Any] = []
    for value in series.tolist():
        if cast_to_int:
            values.append(int(value))
        else:
            values.append(float(value))
    return values

