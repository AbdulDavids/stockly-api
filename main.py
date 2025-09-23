from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from asyncio_throttle import Throttler
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

# Configure yfinance settings as per documentation
# Enable debug mode for troubleshooting (you can disable this in production)
# yf.enable_debug_mode()

# Set custom cache location if needed (optional)
# yf.set_tz_cache_location("./yfinance_cache")

app = FastAPI(title="Stockly Market API", version="0.1.0")

# Configure rate limiting and caching
class YahooFinanceManager:
    def __init__(self):
        # Set up throttler for async operations - this is our main rate limiting mechanism
        # Yahoo Finance has rate limits, so we need to be conservative
        self.throttler = Throttler(rate_limit=1)  # 1 request per second
        
        # Cache for ticker objects to avoid recreating them
        self._ticker_cache = {}
        self._cache_expiry = {}
        
        # Cache for results to avoid redundant API calls
        self._result_cache = {}
        self._result_cache_expiry = {}
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a ticker object with caching"""
        now = time.time()
        
        # Check if we have a cached ticker that's still valid (5 minutes)
        if symbol in self._ticker_cache:
            if symbol in self._cache_expiry and now - self._cache_expiry[symbol] < 300:
                return self._ticker_cache[symbol]
        
        # Create new ticker - let yfinance handle the session
        ticker = yf.Ticker(symbol)
        self._ticker_cache[symbol] = ticker
        self._cache_expiry[symbol] = now
        
        return ticker
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        now = time.time()
        if cache_key in self._result_cache:
            if cache_key in self._result_cache_expiry and now - self._result_cache_expiry[cache_key] < 300:
                return self._result_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with timestamp"""
        self._result_cache[cache_key] = result
        self._result_cache_expiry[cache_key] = time.time()
    
    async def get_info_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Get ticker info with retry logic for rate limiting and caching"""
        cache_key = f"info_{symbol}"
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # If not in cache, fetch with rate limiting
        async with self.throttler:
            for attempt in range(max_retries):
                try:
                    ticker = self.get_ticker(symbol)
                    info = await run_in_threadpool(ticker.get_info)
                    
                    # Cache the successful result
                    self._cache_result(cache_key, info)
                    return info
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "too many requests" in error_str:
                        if attempt < max_retries - 1:
                            # Exponential backoff: 2^attempt * 2 seconds
                            wait_time = (2 ** attempt) * 2
                            print(f"Rate limited on {symbol}, waiting {wait_time}s before retry {attempt + 1}")
                            await asyncio.sleep(wait_time)
                            continue
                    elif attempt < max_retries - 1:
                        # For other errors, wait a bit and retry
                        await asyncio.sleep(1)
                        continue
                    
                    # Log the error but don't raise it
                    print(f"Failed to get info for {symbol} after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    async def get_history_with_retry(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Get ticker history with retry logic and caching"""
        # Create cache key based on symbol and parameters
        cache_key = f"history_{symbol}_{hash(str(sorted(kwargs.items())))}"
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
            
        async with self.throttler:
            for attempt in range(3):
                try:
                    ticker = self.get_ticker(symbol)
                    hist = await run_in_threadpool(ticker.history, **kwargs)
                    
                    # Cache the successful result
                    self._cache_result(cache_key, hist)
                    return hist
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "too many requests" in error_str:
                        if attempt < 2:
                            wait_time = (2 ** attempt) * 2
                            print(f"Rate limited on {symbol} history, waiting {wait_time}s before retry {attempt + 1}")
                            await asyncio.sleep(wait_time)
                            continue
                    elif attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    
                    print(f"Failed to get history for {symbol} after 3 attempts: {e}")
                    return None
        
        return None

# Global instance
yahoo_manager = YahooFinanceManager()


@app.get("/health", summary="Worker status check")
async def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/v7/finance/quote", summary="Batch quote lookup")
async def get_quotes(symbols: str = Query(..., description="Comma separated ticker symbols")) -> Dict[str, Any]:
    tickers = [sym.strip() for sym in symbols.split(",") if sym.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results = await _load_quotes_async(tickers)
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
    result = await _load_summary_async(symbol, modules)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")
    return {"quoteSummary": {"result": [result], "error": None}}


@app.get("/v8/finance/chart/{symbol}", summary="Historical chart data")
async def chart_data(
    symbol: str,
    range: Optional[str] = Query(None, alias="range"),
    interval: Optional[str] = Query(None)
) -> Dict[str, Any]:
    payload = await _load_chart_async(symbol, range, interval)
    return {"chart": {"result": payload, "error": None}}


async def _load_quotes_async(symbols: List[str]) -> List[Dict[str, Any]]:
    """Load quotes with proper rate limiting and retry logic"""
    results: List[Dict[str, Any]] = []
    
    # Process symbols sequentially to avoid overwhelming the API
    for symbol in symbols:
        info = await yahoo_manager.get_info_with_retry(symbol)
        if info is None:
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

# Keep the old function for backwards compatibility
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


async def _load_summary_async(symbol: str, modules: str) -> Optional[Dict[str, Any]]:
    """Load summary with proper rate limiting and retry logic"""
    info = await yahoo_manager.get_info_with_retry(symbol)
    if info is None:
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


def _load_summary(symbol: str, modules: str) -> Optional[Dict[str, Any]]:
    """Legacy function kept for backwards compatibility"""
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


async def _load_chart_async(symbol: str, range_param: Optional[str], interval: Optional[str]) -> List[Dict[str, Any]]:
    """Load chart data with proper rate limiting and retry logic"""
    try:
        # Get both history and info with rate limiting
        hist = await yahoo_manager.get_history_with_retry(
            symbol,
            period=range_param or "1mo",
            interval=interval or "1d",
            auto_adjust=False
        )
        info = await yahoo_manager.get_info_with_retry(symbol)
        
        if hist is None or info is None:
            raise HTTPException(status_code=404, detail=f"Unable to retrieve data for symbol '{symbol}'")
            
    except HTTPException:
        raise
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


def _load_chart(symbol: str, range_param: Optional[str], interval: Optional[str]) -> List[Dict[str, Any]]:
    """Legacy function kept for backwards compatibility"""
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

