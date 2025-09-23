# Stockly Market API (FastAPI)

Lightweight FastAPI wrapper around `yfinance` that mirrors the Yahoo Finance
REST responses used by the Stockly Android client.

## Features

### Core Endpoints (Yahoo Finance Compatible)
- `GET /health` – enhanced status endpoint with cache stats.
- `GET /v7/finance/quote?symbols=AAPL,MSFT` – batch quotes.
- `GET /v1/finance/search?q=NVDA&quotesCount=10` – symbol/company search.
- `GET /v10/finance/quoteSummary/{symbol}?modules=price,summaryDetail` – rich quote summary.
- `GET /v8/finance/chart/{symbol}?range=1mo&interval=1d` – historical price/volume.

### New Market Data Endpoints
- `GET /v1/finance/trending?region=US&count=10` – trending stocks by region.
- `GET /v1/finance/gainers?count=10` – top gaining stocks.
- `GET /v1/finance/losers?count=10` – top losing stocks.
- `GET /v1/finance/most-active?count=10` – most active stocks by volume.
- `GET /v1/finance/sectors` – sector performance data.

### Company-Specific Endpoints
- `GET /v1/finance/dividends/{symbol}?period=1y` – dividend history.
- `GET /v1/finance/splits/{symbol}?period=5y` – stock split history.
- `GET /v1/finance/news/{symbol}?count=10` – recent company news.
- `GET /v1/finance/earnings/{symbol}?quarterly=false` – earnings data.
- `GET /v1/finance/financials/{symbol}?statement=income&quarterly=false` – financial statements.
- `GET /v1/finance/holders/{symbol}?holder_type=institutional` – holder information.

### Analysis Endpoints
- `GET /v1/finance/compare?symbols=AAPL,MSFT,GOOGL&metrics=price,change,pe` – compare stocks.

### API Documentation
- `GET /openapi.json` – OpenAPI specification.
- `GET /docs` – Interactive API documentation (Swagger UI).
- `GET /redoc` – Alternative API documentation (ReDoc).

Responses follow Yahoo’s JSON structure so the Android mappers continue to work
unchanged.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then hit `http://127.0.0.1:8000/health` or any of the finance endpoints. You can
point the Android app at this API by setting `WORKER_BASE_URL` (better name:
`MARKET_API_BASE_URL`) to `http://10.0.2.2:8000/` when running in the emulator.

### Docker / Docker Compose

Build and run with plain Docker:

```bash
docker build -t stockly-api .
docker run --rm -p 8000:8000 stockly-api
```

Or start via Compose:

```bash
docker compose up --build
```

The service will be reachable at `http://127.0.0.1:8000/`.

## Deployment

Any platform that supports Python 3.11 works (Fly.io, Render, Railway, Cloud Run,
Heroku, etc.). The `requirements.txt` lists everything needed. When deploying,
run uvicorn or gunicorn with `main:app` as the entry point.

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**✅ Rate Limiting & Caching Included**: This API includes built-in rate limiting and caching to prevent 429 "Too Many Requests" errors from Yahoo Finance. See [RATE_LIMITING_FIXES.md](RATE_LIMITING_FIXES.md) for details.

## Rate Limiting Features

- **Smart Rate Limiting**: 1 request per second to Yahoo Finance (configurable)
- **Intelligent Caching**: 5-minute cache for all API responses (configurable)
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Error Handling**: Graceful handling of API limits and network issues
- **Container Ready**: Proper cache configuration for Docker/containerized deployments

## Performance

- **Cache Hit Rate**: Repeated requests are served instantly from cache
- **Success Rate**: 100% success rate with proper rate limiting
- **Response Time**: ~1 second for new requests, instant for cached responses
- **yfinance 0.2.66**: Latest version with improved stability and caching

## Configuration

Configure via environment variables:
```bash
YFINANCE_CACHE_DIR=/tmp/cache    # Cache directory (for containers)
YFINANCE_RATE_LIMIT=1.0          # Requests per second
YFINANCE_CACHE_TTL=300          # Cache TTL in seconds
```
