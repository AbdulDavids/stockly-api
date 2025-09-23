# API Endpoints Reference

This document provides a comprehensive reference for all available endpoints in the Stockly Market API.

## Core Endpoints (Yahoo Finance Compatible)

### Health Check
```http
GET /health
```
Returns API status, cache statistics, and version information.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-09-23T21:56:59.138613",
  "cache_stats": {
    "ticker_cache_size": 5,
    "result_cache_size": 12,
    "cache_ttl_seconds": 300,
    "rate_limit_per_second": 1.0
  },
  "yfinance_version": "0.2.66"
}
```

### Batch Quotes
```http
GET /v7/finance/quote?symbols=AAPL,MSFT,GOOGL
```
Get current market data for multiple symbols.

### Symbol Search
```http
GET /v1/finance/search?q=Apple&quotesCount=10
```
Search for stocks by company name or symbol.

### Quote Summary
```http
GET /v10/finance/quoteSummary/AAPL?modules=price,summaryDetail
```
Get detailed information for a specific symbol.

### Historical Chart Data
```http
GET /v8/finance/chart/AAPL?range=1mo&interval=1d
```
Get historical price and volume data.

## Market Data Endpoints

### Trending Stocks
```http
GET /v1/finance/trending?region=US&count=10
```
Get trending stocks by region.

**Parameters:**
- `region` (string): Region code (US, GB, CA, etc.) - Default: "US"
- `count` (integer): Number of stocks to return (1-50) - Default: 10

**Response:**
```json
{
  "trending": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "price": 254.43,
      "change": -1.64,
      "changePercent": -0.6443276
    }
  ],
  "region": "US",
  "count": 10
}
```

### Top Gainers
```http
GET /v1/finance/gainers?count=10
```
Get top gaining stocks by percentage change.

### Top Losers
```http
GET /v1/finance/losers?count=10
```
Get top losing stocks by percentage change.

### Most Active
```http
GET /v1/finance/most-active?count=10
```
Get most active stocks by trading volume.

### Sector Performance
```http
GET /v1/finance/sectors
```
Get performance data for all market sectors.

**Response:**
```json
{
  "sectors": [
    {
      "name": "Technology",
      "change": 2.5,
      "volume": 1500000000
    }
  ],
  "count": 11
}
```

## Company-Specific Endpoints

### Dividend History
```http
GET /v1/finance/dividends/AAPL?period=1y
```
Get dividend payment history for a symbol.

**Parameters:**
- `period` (string): Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max) - Default: "1y"

**Response:**
```json
{
  "symbol": "AAPL",
  "dividends": [
    {
      "date": "2025-08-11",
      "amount": 0.26
    }
  ],
  "period": "1y"
}
```

### Stock Split History
```http
GET /v1/finance/splits/AAPL?period=5y
```
Get stock split history for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "splits": [
    {
      "date": "2020-08-31",
      "ratio": "4:1"
    }
  ],
  "period": "5y"
}
```

### Company News
```http
GET /v1/finance/news/AAPL?count=10
```
Get recent news articles for a company.

**Response:**
```json
{
  "symbol": "AAPL",
  "news": [
    {
      "title": "Apple Reports Q4 Results",
      "link": "https://...",
      "publisher": "Reuters",
      "publishedAt": 1695472800,
      "summary": "Apple reported strong Q4 results..."
    }
  ],
  "count": 10
}
```

### Earnings Data
```http
GET /v1/finance/earnings/AAPL?quarterly=false
```
Get earnings data (net income) for a symbol.

**Parameters:**
- `quarterly` (boolean): Get quarterly data instead of annual - Default: false

### Financial Statements
```http
GET /v1/finance/financials/AAPL?statement=income&quarterly=false
```
Get financial statements for a symbol.

**Parameters:**
- `statement` (string): Type of statement (income, balance, cashflow) - Default: "income"
- `quarterly` (boolean): Get quarterly data instead of annual - Default: false

### Institutional Holders
```http
GET /v1/finance/holders/AAPL?holder_type=institutional
```
Get holder information for a symbol.

**Parameters:**
- `holder_type` (string): Type of holders (institutional, mutual_fund, major) - Default: "institutional"

## Analysis Endpoints

### Stock Comparison
```http
GET /v1/finance/compare?symbols=AAPL,MSFT,GOOGL&metrics=price,change,pe
```
Compare key metrics across multiple stocks.

**Parameters:**
- `symbols` (string): Comma-separated ticker symbols
- `metrics` (string): Comma-separated metrics (price, change, volume, pe, marketcap, dividend)

**Response:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "metrics": ["price", "change", "pe"],
  "comparison": [
    {
      "symbol": "AAPL",
      "price": 254.43,
      "change": -1.64,
      "changePercent": -0.6443276,
      "pe": 38.66717
    }
  ]
}
```

## API Documentation Endpoints

### OpenAPI Specification
```http
GET /openapi.json
```
Get the complete OpenAPI specification in JSON format.

### Interactive Documentation
```http
GET /docs
```
Access Swagger UI for interactive API documentation.

### Alternative Documentation
```http
GET /redoc
```
Access ReDoc for alternative API documentation.

## Rate Limiting and Caching

All endpoints are subject to rate limiting (1 request per second by default) and include intelligent caching:

- **Rate Limiting**: Configurable via `YFINANCE_RATE_LIMIT` environment variable
- **Caching**: 5-minute TTL by default, configurable via `YFINANCE_CACHE_TTL`
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Error Handling**: Graceful handling of API limits and network issues

## Environment Configuration

Configure the API behavior using environment variables:

```bash
# Rate limiting (requests per second)
YFINANCE_RATE_LIMIT=1.0

# Cache TTL (seconds)
YFINANCE_CACHE_TTL=300

# Cache directory
YFINANCE_CACHE_DIR=/tmp

# Verbose logging
YFINANCE_VERBOSE=false
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "detail": "Detailed error information"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (symbol not found)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error