import yfinance as yf

def download_stock_data(ticker, start, end):
    """
    Download historical stock price data for a given ticker and date range.
    Returns a pandas DataFrame.
    """
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

def fetch_company_news(ticker, max_headlines=10):
    """
    Fetch recent news headlines for a specific stock ticker using yfinance.
    Returns a list of headline strings.
    Handles missing keys gracefully.
    """
    stock = yf.Ticker(ticker)
    news_items = getattr(stock, 'news', [])
    headlines = []
    for item in news_items[:max_headlines]:
        # Try 'title', then 'headline', else skip
        headline = item.get('title') or item.get('headline')
        if headline:
            headlines.append(headline)
    return headlines
