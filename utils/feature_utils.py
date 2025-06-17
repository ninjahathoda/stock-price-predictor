def add_technical_indicators(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()
    return df
