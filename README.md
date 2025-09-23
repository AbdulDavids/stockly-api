# Stockly Market API (FastAPI)

Lightweight FastAPI wrapper around `yfinance` that mirrors the Yahoo Finance
REST responses used by the Stockly Android client.

## Features

- `GET /health` – basic status endpoint.
- `GET /v7/finance/quote?symbols=AAPL,MSFT` – batch quotes.
- `GET /v1/finance/search?q=NVDA&quotesCount=10` – symbol/company search.
- `GET /v10/finance/quoteSummary/{symbol}?modules=price,summaryDetail` – rich
  quote summary.
- `GET /v8/finance/chart/{symbol}?range=1mo&interval=1d` – historical price/volume.

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

Consider adding caching or rate limiting if you expect high traffic—`yfinance`
does not provide hard guarantees around upstream usage limits.
