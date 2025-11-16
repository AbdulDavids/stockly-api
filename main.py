from __future__ import annotations

import asyncio
import math
import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from asyncio_throttle import Throttler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.utils import get_openapi
from openai import AsyncOpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Configure yfinance cache locations BEFORE importing yfinance
# This prevents the read-only filesystem errors in containerized environments
cache_dir = os.environ.get("YFINANCE_CACHE_DIR", tempfile.gettempdir())

# Test if we can write to the cache directory
cache_writable = False
try:
    test_path = os.path.join(cache_dir, "test_write_permissions")
    os.makedirs(cache_dir, exist_ok=True)
    with open(test_path, "w") as f:
        f.write("test")
    os.remove(test_path)
    cache_writable = True
    print(f"✅ Cache directory is writable: {cache_dir}")
except Exception as e:
    print(
        f"⚠️  Cache directory not writable ({e}), yfinance will run without internal caching"
    )

# Set up yfinance cache paths if filesystem is writable
if cache_writable:
    # Set timezone cache location
    tz_cache_path = os.path.join(cache_dir, "yfinance_tz_cache")
    os.makedirs(tz_cache_path, exist_ok=True)

    # Set cookie cache location by setting the cache directory in environment
    # yfinance will use ~/.cache/py-yfinance, so we create that structure
    py_cache_dir = os.path.join(cache_dir, ".cache")
    py_yfinance_cache = os.path.join(py_cache_dir, "py-yfinance")
    os.makedirs(py_yfinance_cache, exist_ok=True)

    # Override HOME environment variable temporarily for yfinance
    original_home = os.environ.get("HOME")
    os.environ["HOME"] = cache_dir

# Now import yfinance after setting up cache locations
import yfinance as yf

# Configure yfinance settings after import
if cache_writable:
    try:
        yf.set_tz_cache_location(tz_cache_path)
        print(f"✅ yfinance timezone cache: {tz_cache_path}")
        print(f"✅ yfinance cookie cache: {py_yfinance_cache}")
    except Exception as e:
        print(f"⚠️  Could not configure yfinance caches: {e}")

    # Restore original HOME if it existed
    if original_home:
        os.environ["HOME"] = original_home
    elif "HOME" in os.environ:
        del os.environ["HOME"]

# Enable debug mode for troubleshooting (you can disable this in production)
# yf.enable_debug_mode()


# Response Models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    cache_stats: Dict[str, Any]
    yfinance_version: str


class QuoteData(BaseModel):
    symbol: str
    regularMarketPrice: Optional[float] = None
    regularMarketChange: Optional[float] = None
    regularMarketChangePercent: Optional[float] = None
    regularMarketVolume: Optional[int] = None
    marketCap: Optional[float] = None
    fiftyTwoWeekHigh: Optional[float] = None
    fiftyTwoWeekLow: Optional[float] = None


class QuoteResponse(BaseModel):
    quoteResponse: Dict[str, Any]


class SearchResult(BaseModel):
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    type: Optional[str] = None


class SearchResponse(BaseModel):
    quotes: List[SearchResult]
    news: List[Dict[str, Any]] = []


class TrendingStock(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    change: Optional[float] = None
    changePercent: Optional[float] = None


class TrendingResponse(BaseModel):
    trending: List[TrendingStock]
    region: str
    count: int
    error: Optional[str] = None


class MarketMoverStock(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    change: Optional[float] = None
    changePercent: Optional[float] = None
    volume: Optional[int] = None


class MarketMoversResponse(BaseModel):
    stocks: List[MarketMoverStock]
    count: int
    error: Optional[str] = None


class DividendData(BaseModel):
    date: Any  # Can be string or timestamp
    amount: Any  # Can be float or other numeric type


class DividendsResponse(BaseModel):
    symbol: str
    dividends: List[DividendData]
    count: int
    period: str


class SplitData(BaseModel):
    date: Any  # Can be string or timestamp
    ratio: Any  # Can be string or numeric


class SplitsResponse(BaseModel):
    symbol: str
    splits: List[SplitData]
    count: int
    period: str


class NewsArticle(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    publishedAt: Optional[Any] = None  # Can be int timestamp or string
    thumbnail: Optional[str] = None


class NewsResponse(BaseModel):
    symbol: str
    news: List[NewsArticle]
    count: int


class CompareResponse(BaseModel):
    symbols: List[str]
    metrics: List[str]
    comparison: List[Dict[str, Any]]


class SectorData(BaseModel):
    name: str
    performance: Optional[float] = None
    volume: Optional[int] = None


class SectorResponse(BaseModel):
    sectors: List[SectorData]
    count: int


class StockRecommendation(BaseModel):
    symbol: Optional[str] = None
    recommendation: str  # "BUY", "SELL", or "HOLD"
    priceTarget: float
    riskScore: int  # 1-10
    confidence: int  # 0-100
    justification: str
    timestamp: Optional[str] = None
    error: Optional[str] = None


tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and status endpoints",
    },
    {
        "name": "Market Data",
        "description": "Real-time and historical stock market data",
    },
    {
        "name": "Search",
        "description": "Search for stocks and companies",
    },
    {
        "name": "Market Movers",
        "description": "Trending, gainers, losers, and most active stocks",
    },
    {
        "name": "Company Data",
        "description": "Company-specific information including dividends, splits, news, earnings, and financials",
    },
    {
        "name": "Analysis",
        "description": "Stock comparison and sector performance analysis",
    },
    {
        "name": "AI Insights",
        "description": "AI-powered stock analysis and recommendations",
    },
]

app = FastAPI(
    title="Stockly Market API",
    version="0.1.0",
    description="""
## Stockly Market API

A comprehensive stock market API powered by yfinance with rate limiting and caching.

### Features:
* **Real-time stock data** - Get current stock prices and market data
* **Historical data** - Access historical stock prices and trading volumes
* **Technical indicators** - Calculate moving averages, RSI, MACD, and more
* **AI-powered analysis** - Get AI-generated stock recommendations and insights
* **Rate limiting** - Built-in rate limiting to prevent API abuse
* **Caching** - Intelligent caching for improved performance

### Rate Limits:
* Configurable rate limiting per endpoint
* Default: 1 request per second to Yahoo Finance API
    """,
    contact={
        "name": "Stockly API",
        "url": "https://github.com/your-repo/stockly-api",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
)


def custom_openapi() -> Dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    # Use FastAPI's default openapi generation
    from fastapi.openapi.utils import get_openapi as _get_openapi

    openapi_schema = _get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add tags metadata
    if app.openapi_tags:
        openapi_schema["tags"] = app.openapi_tags

    # Add custom server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "https://stockly-api.vercel.app", "description": "Production"},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Configure rate limiting and caching
class YahooFinanceManager:
    def __init__(self):
        # Configuration from environment variables
        rate_limit = float(
            os.environ.get("YFINANCE_RATE_LIMIT", "1.0")
        )  # requests per second
        cache_ttl = int(
            os.environ.get("YFINANCE_CACHE_TTL", "300")
        )  # cache TTL in seconds
        self.verbose_logging = (
            os.environ.get("YFINANCE_VERBOSE", "false").lower() == "true"
        )

        # Set up throttler for async operations - this is our main rate limiting mechanism
        self.throttler = Throttler(rate_limit=rate_limit)
        self.cache_ttl = cache_ttl

        # AI cache TTL - default to 24 hours (86400 seconds)
        self.ai_cache_ttl = int(
            os.environ.get("AI_CACHE_TTL", "86400")
        )

        print(
            f"🔧 Rate limit: {rate_limit} req/sec, Cache TTL: {cache_ttl}s, AI Cache TTL: {self.ai_cache_ttl}s, Verbose: {self.verbose_logging}"
        )

        # Cache for ticker objects to avoid recreating them
        self._ticker_cache = {}
        self._cache_expiry = {}

        # Cache for results to avoid redundant API calls
        self._result_cache = {}
        self._result_cache_expiry = {}

        # Cache for AI generations (separate cache with different TTL)
        self._ai_cache = {}
        self._ai_cache_expiry = {}

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a ticker object with caching"""
        now = time.time()

        # Check if we have a cached ticker that's still valid
        if symbol in self._ticker_cache:
            if (
                symbol in self._cache_expiry
                and now - self._cache_expiry[symbol] < self.cache_ttl
            ):
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
            if (
                cache_key in self._result_cache_expiry
                and now - self._result_cache_expiry[cache_key] < self.cache_ttl
            ):
                return self._result_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with timestamp"""
        self._result_cache[cache_key] = result
        self._result_cache_expiry[cache_key] = time.time()

    def get_cached_ai_result(self, cache_key: str) -> Optional[Any]:
        """Get cached AI result if available and not expired"""
        now = time.time()
        if cache_key in self._ai_cache:
            if (
                cache_key in self._ai_cache_expiry
                and now - self._ai_cache_expiry[cache_key] < self.ai_cache_ttl
            ):
                if self.verbose_logging:
                    print(f"💾 AI Cache HIT for {cache_key}")
                return self._ai_cache[cache_key]
        if self.verbose_logging:
            print(f"🌐 AI Cache MISS for {cache_key}")
        return None

    def cache_ai_result(self, cache_key: str, result: Any):
        """Cache an AI result with timestamp"""
        self._ai_cache[cache_key] = result
        self._ai_cache_expiry[cache_key] = time.time()
        if self.verbose_logging:
            print(f"💾 AI result cached for {cache_key}")

    async def get_info_with_retry(
        self, symbol: str, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Get ticker info with retry logic for rate limiting and caching"""
        cache_key = f"info_{symbol}"

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            if self.verbose_logging:
                print(f"💾 Cache HIT for {symbol} info")
            return cached_result

        if self.verbose_logging:
            print(f"🌐 Cache MISS for {symbol} info - fetching from API")

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
                            wait_time = (2**attempt) * 2
                            print(
                                f"Rate limited on {symbol}, waiting {wait_time}s before retry {attempt + 1}"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                    elif attempt < max_retries - 1:
                        # For other errors, wait a bit and retry
                        await asyncio.sleep(1)
                        continue

                    # Log the error but don't raise it
                    print(
                        f"Failed to get info for {symbol} after {max_retries} attempts: {e}"
                    )
                    return None

        return None

    async def get_history_with_retry(
        self, symbol: str, **kwargs
    ) -> Optional[pd.DataFrame]:
        """Get ticker history with retry logic and caching"""
        # Create cache key based on symbol and parameters
        cache_key = f"history_{symbol}_{hash(str(sorted(kwargs.items())))}"

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            if self.verbose_logging:
                print(f"💾 Cache HIT for {symbol} history")
            return cached_result

        if self.verbose_logging:
            print(f"🌐 Cache MISS for {symbol} history - fetching from API")

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
                            wait_time = (2**attempt) * 2
                            print(
                                f"Rate limited on {symbol} history, waiting {wait_time}s before retry {attempt + 1}"
                            )
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


@app.get("/health", summary="Worker status check", tags=["Health"], response_model=HealthResponse)
async def health() -> HealthResponse:
    cache_stats = {
        "ticker_cache_size": len(yahoo_manager._ticker_cache),
        "result_cache_size": len(yahoo_manager._result_cache),
        "cache_ttl_seconds": yahoo_manager.cache_ttl,
        "rate_limit_per_second": yahoo_manager.throttler.rate_limit,
    }

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": cache_stats,
        "yfinance_version": yf.__version__,
    }


@app.get(
    "/v7/finance/quote",
    summary="Batch quote lookup",
    tags=["Market Data"],
    response_model=QuoteResponse,
)
async def get_quotes(
    symbols: str = Query(..., description="Comma separated ticker symbols"),
) -> QuoteResponse:
    tickers = [sym.strip() for sym in symbols.split(",") if sym.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results = await _load_quotes_async(tickers)
    return {"quoteResponse": {"result": results, "error": None}}


@app.get(
    "/v1/finance/search",
    summary="Symbol/company search",
    tags=["Search"],
    response_model=SearchResponse,
)
async def search_quotes(
    q: str = Query(..., description="Search term"),
    quotesCount: int = Query(
        10, ge=1, le=25, description="Max number of quote matches"
    ),
) -> SearchResponse:
    if not q:
        return {"quotes": [], "news": [], "total": 0}

    payload = await run_in_threadpool(_search_quotes, q, quotesCount)
    return payload


@app.get("/v10/finance/quoteSummary/{symbol}", summary="Detailed quote summary", tags=["Market Data"])
async def quote_summary(
    symbol: str, modules: str = Query("", description="Comma separated modules")
) -> Dict[str, Any]:
    result = await _load_summary_async(symbol, modules)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")
    return {"quoteSummary": {"result": [result], "error": None}}


@app.get("/v8/finance/chart/{symbol}", summary="Historical chart data", tags=["Market Data"])
async def chart_data(
    symbol: str,
    range: Optional[str] = Query(None, alias="range"),
    interval: Optional[str] = Query(None),
) -> Dict[str, Any]:
    payload = await _load_chart_async(symbol, range, interval)
    return {"chart": {"result": payload, "error": None}}


# Additional helpful endpoints
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi():
    """Get OpenAPI JSON specification"""
    return app.openapi()


@app.get(
    "/v1/finance/trending",
    summary="Get trending stocks",
    tags=["Market Movers"],
    response_model=TrendingResponse,
)
async def get_trending(
    region: str = Query("US", description="Region code (US, GB, CA, etc.)"),
    count: int = Query(
        10, ge=1, le=50, description="Number of trending stocks to return"
    ),
) -> TrendingResponse:
    """Get trending stocks for a specific region"""
    try:
        # Convert Query objects to actual values if needed (for direct function calls)
        actual_region = region.default if hasattr(region, "default") else region
        actual_count = count.default if hasattr(count, "default") else count

        trending = await _get_trending_symbols(actual_region, actual_count)
        return {"trending": trending, "region": actual_region, "count": len(trending)}
    except Exception as e:
        error_region = region.default if hasattr(region, "default") else region
        return {"trending": [], "region": error_region, "count": 0, "error": str(e)}


@app.get(
    "/v1/finance/gainers",
    summary="Top gaining stocks",
    tags=["Market Movers"],
    response_model=MarketMoversResponse,
)
async def get_gainers(
    count: int = Query(10, ge=1, le=50, description="Number of top gainers to return"),
) -> MarketMoversResponse:
    """Get top gaining stocks"""
    try:
        gainers = await _get_market_movers("gainers", count)
        return {"stocks": gainers, "count": len(gainers)}
    except Exception as e:
        return {"stocks": [], "count": 0, "error": str(e)}


@app.get(
    "/v1/finance/losers",
    summary="Top losing stocks",
    tags=["Market Movers"],
    response_model=MarketMoversResponse,
)
async def get_losers(
    count: int = Query(10, ge=1, le=50, description="Number of top losers to return"),
) -> MarketMoversResponse:
    """Get top losing stocks"""
    try:
        losers = await _get_market_movers("losers", count)
        return {"stocks": losers, "count": len(losers)}
    except Exception as e:
        return {"stocks": [], "count": 0, "error": str(e)}


@app.get(
    "/v1/finance/most-active",
    summary="Most active stocks",
    tags=["Market Movers"],
    response_model=MarketMoversResponse,
)
async def get_most_active(
    count: int = Query(
        10, ge=1, le=50, description="Number of most active stocks to return"
    ),
) -> MarketMoversResponse:
    """Get most active stocks by volume"""
    try:
        active = await _get_market_movers("most_active", count)
        return {"stocks": active, "count": len(active)}
    except Exception as e:
        return {"stocks": [], "count": 0, "error": str(e)}


@app.get(
    "/v1/finance/dividends/{symbol}",
    summary="Dividend history",
    tags=["Company Data"],
    response_model=DividendsResponse,
)
async def get_dividends(
    symbol: str,
    period: str = Query(
        "1y",
        description="Period for dividend history (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
    ),
) -> DividendsResponse:
    """Get dividend history for a symbol"""
    try:
        dividends = await _load_dividends_async(symbol, period)
        return {"symbol": symbol, "dividends": dividends, "count": len(dividends), "period": period}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/finance/splits/{symbol}",
    summary="Stock split history",
    tags=["Company Data"],
    response_model=SplitsResponse,
)
async def get_splits(
    symbol: str,
    period: str = Query(
        "5y",
        description="Period for split history (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
    ),
) -> SplitsResponse:
    """Get stock split history for a symbol"""
    try:
        splits = await _load_splits_async(symbol, period)
        return {"symbol": symbol, "splits": splits, "count": len(splits), "period": period}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/finance/news/{symbol}",
    summary="Company news",
    tags=["Company Data"],
    response_model=NewsResponse,
)
async def get_company_news(
    symbol: str,
    count: int = Query(
        10, ge=1, le=100, description="Number of news articles to return"
    ),
) -> NewsResponse:
    """Get recent news for a company"""
    try:
        news = await _load_news_async(symbol, count)
        return {"symbol": symbol, "news": news, "count": len(news)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/finance/earnings/{symbol}", summary="Earnings data", tags=["Company Data"])
async def get_earnings(
    symbol: str,
    quarterly: bool = Query(
        False, description="Get quarterly earnings instead of annual"
    ),
) -> Dict[str, Any]:
    """Get earnings data for a symbol"""
    try:
        earnings = await _load_earnings_async(symbol, quarterly)
        return {"symbol": symbol, "earnings": earnings, "quarterly": quarterly}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/finance/financials/{symbol}", summary="Financial statements", tags=["Company Data"])
async def get_financials(
    symbol: str,
    statement: str = Query(
        "income", description="Type of financial statement (income, balance, cashflow)"
    ),
    quarterly: bool = Query(False, description="Get quarterly data instead of annual"),
) -> Dict[str, Any]:
    """Get financial statements for a symbol"""
    try:
        financials = await _load_financials_async(symbol, statement, quarterly)
        return {
            "symbol": symbol,
            "statement": statement,
            "quarterly": quarterly,
            "data": financials,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/finance/holders/{symbol}", summary="Institutional holders", tags=["Company Data"])
async def get_holders(
    symbol: str,
    holder_type: str = Query(
        "institutional",
        description="Type of holders (institutional, mutual_fund, major)",
    ),
) -> Dict[str, Any]:
    """Get holder information for a symbol"""
    try:
        holders = await _load_holders_async(symbol, holder_type)
        return {"symbol": symbol, "holder_type": holder_type, "holders": holders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/finance/compare",
    summary="Compare multiple stocks",
    tags=["Analysis"],
    response_model=CompareResponse,
)
async def compare_stocks(
    symbols: str = Query(..., description="Comma separated ticker symbols to compare"),
    metrics: str = Query(
        "price,change,volume,pe", description="Comma separated metrics to compare"
    ),
) -> CompareResponse:
    """Compare key metrics across multiple stocks"""
    try:
        tickers = [sym.strip() for sym in symbols.split(",") if sym.strip()]
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

        comparison = await _compare_stocks_async(tickers, metric_list)
        return {"symbols": tickers, "metrics": metric_list, "comparison": comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/finance/sectors",
    summary="Sector performance",
    tags=["Analysis"],
    response_model=SectorResponse,
)
async def get_sector_performance() -> SectorResponse:
    """Get sector performance data"""
    try:
        sectors = await run_in_threadpool(_get_sector_performance)
        return {"sectors": sectors, "count": len(sectors)}
    except Exception as e:
        return {"sectors": [], "count": 0, "error": str(e)}


# Helper functions for new endpoints
async def _get_trending_symbols(region: str, count: int) -> List[Dict[str, Any]]:
    """Get trending symbols - using popular symbols since yfinance trending is not available"""
    try:
        # Use popular symbols by region since yfinance.search doesn't work as expected
        trending_symbols = {
            "US": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"],
            "GB": ["LLOY.L", "BP.L", "SHEL.L", "AZN.L"],
            "CA": ["SHOP.TO", "CNR.TO", "RY.TO", "TD.TO"],
        }

        available_symbols = trending_symbols.get(region, trending_symbols["US"])
        symbols = (
            available_symbols[:count]
            if count <= len(available_symbols)
            else available_symbols
        )
        result = []

        for symbol in symbols:
            info = await yahoo_manager.get_info_with_retry(symbol)
            if info and info.get("regularMarketPrice"):
                result.append(
                    {
                        "symbol": symbol,
                        "name": info.get("shortName", symbol),
                        "price": info.get("regularMarketPrice"),
                        "change": info.get("regularMarketChange"),
                        "changePercent": info.get("regularMarketChangePercent"),
                    }
                )

        return result
    except Exception as e:
        print(f"Error getting trending symbols: {e}")
        return []


async def _get_market_movers(mover_type: str, count: int) -> List[Dict[str, Any]]:
    """Get market movers - real data from yfinance, categorized by typical performance"""
    try:
        # Use real symbols and get actual data - let real market data determine gainers/losers
        popular_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "NFLX",
            "AMD",
            "CRM",
        ]

        result = []
        for symbol in popular_symbols:
            info = await yahoo_manager.get_info_with_retry(symbol)
            if info and info.get("regularMarketPrice"):
                result.append(
                    {
                        "symbol": symbol,
                        "name": info.get("shortName", symbol),
                        "price": info.get("regularMarketPrice"),
                        "change": info.get("regularMarketChange"),
                        "changePercent": info.get("regularMarketChangePercent"),
                        "volume": info.get("regularMarketVolume"),
                    }
                )

        # Sort by actual performance
        if mover_type == "gainers":
            result.sort(key=lambda x: x.get("changePercent", -999), reverse=True)
        elif mover_type == "losers":
            result.sort(key=lambda x: x.get("changePercent", 999))
        elif mover_type == "most_active":
            result.sort(key=lambda x: x.get("volume", 0), reverse=True)

        return result[:count]
    except Exception as e:
        print(f"Error getting market movers: {e}")
        return []


async def _load_dividends_async(symbol: str, period: str) -> List[Dict[str, Any]]:
    """Load dividend history for a symbol"""
    cache_key = f"dividends_{symbol}_{period}"

    # Check cache first
    cached_result = yahoo_manager._get_cached_result(cache_key)
    if cached_result is not None:
        if yahoo_manager.verbose_logging:
            print(f"💾 Cache HIT for {symbol} dividends")
        return cached_result

    if yahoo_manager.verbose_logging:
        print(f"🌐 Cache MISS for {symbol} dividends - fetching from API")

    async with yahoo_manager.throttler:
        try:
            ticker = yahoo_manager.get_ticker(symbol)
            dividends = ticker.dividends

            if not dividends.empty:
                # Filter by period
                if period != "max":
                    dividends = dividends.tail(50)  # Limit for performance

                result = []
                for date, amount in dividends.items():
                    result.append(
                        {"date": date.strftime("%Y-%m-%d"), "amount": float(amount)}
                    )

                yahoo_manager._cache_result(cache_key, result)
                return result
            else:
                yahoo_manager._cache_result(cache_key, [])
                return []

        except Exception as e:
            print(f"Error loading dividends for {symbol}: {e}")
            return []


async def _load_splits_async(symbol: str, period: str) -> List[Dict[str, Any]]:
    """Load stock split history for a symbol"""
    cache_key = f"splits_{symbol}_{period}"

    # Check cache first
    cached_result = yahoo_manager._get_cached_result(cache_key)
    if cached_result is not None:
        if yahoo_manager.verbose_logging:
            print(f"💾 Cache HIT for {symbol} splits")
        return cached_result

    if yahoo_manager.verbose_logging:
        print(f"🌐 Cache MISS for {symbol} splits - fetching from API")

    async with yahoo_manager.throttler:
        try:
            ticker = yahoo_manager.get_ticker(symbol)
            splits = ticker.splits

            if not splits.empty:
                result = []
                for date, ratio in splits.items():
                    result.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "ratio": f"{int(ratio)}:1"
                            if ratio > 1
                            else f"1:{int(1 / ratio)}",
                        }
                    )

                yahoo_manager._cache_result(cache_key, result)
                return result
            else:
                yahoo_manager._cache_result(cache_key, [])
                return []

        except Exception as e:
            print(f"Error loading splits for {symbol}: {e}")
            return []


async def _load_news_async(symbol: str, count: int) -> List[Dict[str, Any]]:
    """Load recent news for a symbol"""
    cache_key = f"news_{symbol}_{count}"

    # Check cache first
    cached_result = yahoo_manager._get_cached_result(cache_key)
    if cached_result is not None:
        if yahoo_manager.verbose_logging:
            print(f"💾 Cache HIT for {symbol} news")
        return cached_result

    if yahoo_manager.verbose_logging:
        print(f"🌐 Cache MISS for {symbol} news - fetching from API")

    async with yahoo_manager.throttler:
        try:
            ticker = yahoo_manager.get_ticker(symbol)
            news = ticker.news

            result = []
            for article in news[:count]:
                result.append(
                    {
                        "title": article.get("title", ""),
                        "link": article.get("link", ""),
                        "publisher": article.get("publisher", ""),
                        "publishedAt": article.get("providerPublishTime", 0),
                        "summary": article.get("summary", ""),
                    }
                )

            yahoo_manager._cache_result(cache_key, result)
            return result

        except Exception as e:
            print(f"Error loading news for {symbol}: {e}")
            return []


async def _load_earnings_async(symbol: str, quarterly: bool) -> Dict[str, Any]:
    """Load earnings data for a symbol"""
    cache_key = f"earnings_{symbol}_{quarterly}"

    # Check cache first
    cached_result = yahoo_manager._get_cached_result(cache_key)
    if cached_result is not None:
        if yahoo_manager.verbose_logging:
            print(f"💾 Cache HIT for {symbol} earnings")
        return cached_result

    if yahoo_manager.verbose_logging:
        print(f"🌐 Cache MISS for {symbol} earnings - fetching from API")

    async with yahoo_manager.throttler:
        try:
            ticker = yahoo_manager.get_ticker(symbol)

            if quarterly:
                # Use quarterly income statement and extract net income
                income_stmt = ticker.quarterly_income_stmt
            else:
                # Use annual income statement and extract net income
                income_stmt = ticker.income_stmt

            if income_stmt is not None and not income_stmt.empty:
                # Extract net income as earnings proxy
                result = {}
                if "Net Income" in income_stmt.index:
                    net_income_data = income_stmt.loc["Net Income"]
                    result = {
                        "net_income": net_income_data.to_dict(),
                        "periods": list(
                            net_income_data.index.strftime("%Y-%m-%d")
                            if hasattr(net_income_data.index, "strftime")
                            else net_income_data.index
                        ),
                    }
                else:
                    # Fallback: return the entire income statement
                    result = income_stmt.to_dict()

                yahoo_manager._cache_result(cache_key, result)
                return result
            else:
                yahoo_manager._cache_result(cache_key, {})
                return {}

        except Exception as e:
            print(f"Error loading earnings for {symbol}: {e}")
            return {}


async def _load_financials_async(
    symbol: str, statement: str, quarterly: bool
) -> Dict[str, Any]:
    """Load financial statements for a symbol"""
    cache_key = f"financials_{symbol}_{statement}_{quarterly}"

    # Check cache first
    cached_result = yahoo_manager._get_cached_result(cache_key)
    if cached_result is not None:
        if yahoo_manager.verbose_logging:
            print(f"💾 Cache HIT for {symbol} {statement}")
        return cached_result

    if yahoo_manager.verbose_logging:
        print(f"🌐 Cache MISS for {symbol} {statement} - fetching from API")

    async with yahoo_manager.throttler:
        try:
            ticker = yahoo_manager.get_ticker(symbol)

            # Get the appropriate financial statement
            if statement == "income":
                data = ticker.quarterly_income_stmt if quarterly else ticker.income_stmt
            elif statement == "balance":
                data = (
                    ticker.quarterly_balance_sheet
                    if quarterly
                    else ticker.balance_sheet
                )
            elif statement == "cashflow":
                data = ticker.quarterly_cashflow if quarterly else ticker.cashflow
            else:
                data = None

            if data is not None and not data.empty:
                result = data.to_dict()
                yahoo_manager._cache_result(cache_key, result)
                return result
            else:
                yahoo_manager._cache_result(cache_key, {})
                return {}

        except Exception as e:
            print(f"Error loading {statement} for {symbol}: {e}")
            return {}


async def _load_holders_async(symbol: str, holder_type: str) -> List[Dict[str, Any]]:
    """Load holder information for a symbol"""
    cache_key = f"holders_{symbol}_{holder_type}"

    # Check cache first
    cached_result = yahoo_manager._get_cached_result(cache_key)
    if cached_result is not None:
        if yahoo_manager.verbose_logging:
            print(f"💾 Cache HIT for {symbol} {holder_type} holders")
        return cached_result

    if yahoo_manager.verbose_logging:
        print(f"🌐 Cache MISS for {symbol} {holder_type} holders - fetching from API")

    async with yahoo_manager.throttler:
        try:
            ticker = yahoo_manager.get_ticker(symbol)

            if holder_type == "institutional":
                holders = ticker.institutional_holders
            elif holder_type == "mutual_fund":
                holders = ticker.mutualfund_holders
            elif holder_type == "major":
                holders = ticker.major_holders
            else:
                holders = None

            if holders is not None and not holders.empty:
                result = holders.to_dict("records")
                yahoo_manager._cache_result(cache_key, result)
                return result
            else:
                yahoo_manager._cache_result(cache_key, [])
                return []

        except Exception as e:
            print(f"Error loading {holder_type} holders for {symbol}: {e}")
            return []


async def _compare_stocks_async(
    symbols: List[str], metrics: List[str]
) -> List[Dict[str, Any]]:
    """Compare metrics across multiple stocks"""
    results = []

    for symbol in symbols:
        try:
            info = await yahoo_manager.get_info_with_retry(symbol)
            if info:
                comparison = {"symbol": symbol}

                for metric in metrics:
                    if metric == "price":
                        comparison["price"] = info.get("regularMarketPrice")
                    elif metric == "change":
                        comparison["change"] = info.get("regularMarketChange")
                        comparison["changePercent"] = info.get(
                            "regularMarketChangePercent"
                        )
                    elif metric == "volume":
                        comparison["volume"] = info.get("regularMarketVolume")
                    elif metric == "pe":
                        comparison["pe"] = info.get("trailingPE")
                    elif metric == "marketcap":
                        comparison["marketCap"] = info.get("marketCap")
                    elif metric == "dividend":
                        comparison["dividendYield"] = info.get("dividendYield")

                results.append(comparison)
        except Exception as e:
            print(f"Error comparing {symbol}: {e}")
            continue

    return results


def _get_sector_performance() -> List[Dict[str, Any]]:
    """Get sector performance data - currently not available through yfinance"""
    # yfinance doesn't provide sector data directly
    # This would need to be implemented with another data source
    return []


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
    """Search for quotes - simplified since yfinance search is not reliable"""
    try:
        # Simple mapping for common searches since yf.search doesn't work reliably
        search_mappings = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "facebook": "META",
            "netflix": "NFLX",
            "nvidia": "NVDA",
        }

        query_lower = query.lower()
        matches = []

        # Check if query matches any known companies
        for company, symbol in search_mappings.items():
            if company in query_lower or query_lower in company:
                matches.append(symbol)

        # Also check if query is already a symbol
        if query.upper() not in matches:
            matches.append(query.upper())

        quotes = []
        for symbol in matches[:limit]:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and info.get("symbol"):
                    quotes.append(
                        {
                            "symbol": info.get("symbol", symbol),
                            "shortname": info.get("shortName", ""),
                            "longname": info.get("longName", ""),
                            "exchDisp": info.get("exchange", ""),
                            "typeDisp": "Equity",
                            "quoteType": "EQUITY",
                        }
                    )
            except:
                continue

        return {"quotes": quotes, "news": [], "total": len(quotes)}
    except Exception as e:
        print(f"Search error: {e}")
        return {"quotes": [], "news": [], "total": 0}


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

    result = {
        key: prune(value) for key, value in sections.items() if prune(value) is not None
    }

    requested_modules = {
        module.strip() for module in modules.split(",") if module.strip()
    }
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

    result = {
        key: prune(value) for key, value in sections.items() if prune(value) is not None
    }

    requested_modules = {
        module.strip() for module in modules.split(",") if module.strip()
    }
    if requested_modules:
        result = {key: val for key, val in result.items() if key in requested_modules}

    return result


async def _load_chart_async(
    symbol: str, range_param: Optional[str], interval: Optional[str]
) -> List[Dict[str, Any]]:
    """Load chart data with proper rate limiting and retry logic"""
    try:
        # Get both history and info with rate limiting
        hist = await yahoo_manager.get_history_with_retry(
            symbol,
            period=range_param or "1mo",
            interval=interval or "1d",
            auto_adjust=False,
        )
        info = await yahoo_manager.get_info_with_retry(symbol)

        if hist is None or info is None:
            raise HTTPException(
                status_code=404, detail=f"Unable to retrieve data for symbol '{symbol}'"
            )

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


def _load_chart(
    symbol: str, range_param: Optional[str], interval: Optional[str]
) -> List[Dict[str, Any]]:
    """Legacy function kept for backwards compatibility"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            period=range_param or "1mo", interval=interval or "1d", auto_adjust=False
        )
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


def _series_to_list(
    series: Optional[pd.Series], cast_to_int: bool = False
) -> List[Any]:
    if series is None:
        return []
    values: List[Any] = []
    for value in series.tolist():
        if cast_to_int:
            values.append(int(value))
        else:
            values.append(float(value))
    return values


# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.get(
    "/generate-stock-insights",
    summary="AI-powered stock analysis",
    tags=["AI Insights"],
    response_model=StockRecommendation,
)
async def generate_stock_insights(
    symbol: str = Query(..., description="Stock symbol")
) -> StockRecommendation:
    """
    Generate AI-powered stock insights using GPT-4o-mini with structured output.
    Returns BUY/SELL/HOLD recommendation, price target, risk score, confidence, and justification.

    Results are cached daily - the same symbol will return cached results for 24 hours.
    """
    try:
        # Create cache key based on symbol and current date
        from datetime import date
        cache_key = f"ai_insights_{symbol.upper()}_{date.today().isoformat()}"

        # Check cache first
        cached_result = yahoo_manager.get_cached_ai_result(cache_key)
        if cached_result is not None:
            # Ensure symbol is always present in cached results
            if "symbol" not in cached_result or cached_result["symbol"] is None:
                cached_result["symbol"] = symbol
            return cached_result

        # Fetch stock data using existing infrastructure
        ticker = yf.Ticker(symbol)
        info = await run_in_threadpool(ticker.get_info)
        hist = await run_in_threadpool(
            ticker.history, period="1y", interval="1d", auto_adjust=False
        )

        if not info or hist.empty:
            raise HTTPException(
                status_code=404, detail=f"Stock data not found for {symbol}"
            )

        # Calculate performance metrics
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")
        if not current_price:
            raise HTTPException(status_code=400, detail="Unable to fetch current price")

        # Calculate 1-year performance
        year_ago_price = hist["Close"].iloc[0] if len(hist) > 0 else current_price
        year_performance = (
            ((current_price - year_ago_price) / year_ago_price * 100)
            if year_ago_price
            else 0
        )

        # Calculate volatility (standard deviation of daily returns)
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = daily_returns.std() * 100 if len(daily_returns) > 0 else 0

        # Prepare prompt for AI
        prompt = f"""Analyze the following stock and provide investment insights:

Stock: {symbol}
Company: {info.get("longName", "N/A")}
Sector: {info.get("sector", "N/A")}
Industry: {info.get("industry", "N/A")}
Current Price: ${current_price:.2f}
52-Week High: ${info.get("fiftyTwoWeekHigh", "N/A")}
52-Week Low: ${info.get("fiftyTwoWeekLow", "N/A")}
Market Cap: {info.get("marketCap", "N/A")}
P/E Ratio: {info.get("trailingPE", "N/A")}
Forward P/E: {info.get("forwardPE", "N/A")}
PEG Ratio: {info.get("pegRatio", "N/A")}
1-Year Performance: {year_performance:.2f}%
Volatility (std dev): {volatility:.2f}%
Analyst Target Price: ${info.get("targetMeanPrice", "N/A")}

Provide a recommendation (BUY, SELL, or HOLD), a 6-12 month price target, risk score (1-10), confidence level (0-100), and a 2-3 sentence justification."""

        # Call OpenAI with structured output
        completion = await openai_client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst AI providing stock investment insights.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=StockRecommendation,
        )

        recommendation = completion.choices[0].message.parsed

        # Validate and prepare response
        result = {
            "symbol": symbol,
            "recommendation": recommendation.recommendation.upper(),
            "priceTarget": recommendation.priceTarget,
            "riskScore": max(1, min(10, recommendation.riskScore)),
            "confidence": max(0, min(100, recommendation.confidence)),
            "justification": recommendation.justification[:500],
            "timestamp": datetime.now().isoformat(),
            "error": None,
        }

        # Cache the result for 24 hours
        yahoo_manager.cache_ai_result(cache_key, result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate AI insights: {str(e)}"
        )
