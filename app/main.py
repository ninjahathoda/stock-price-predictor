import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import download_stock_data, fetch_company_news
from utils.feature_utils import add_technical_indicators
from utils.nlp_utils import get_sentiment_scores
from models.numerical.lstm_model import create_lstm_model, prepare_lstm_data, scale_data, inverse_scale
from models.textual.sentiment_model import aggregate_sentiment
from models.hybrid.hybrid_model import combine_features
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Stock Price Predictor", layout="wide")
st.title("📈 Advanced Stock Price Prediction Dashboard")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL")
start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end = st.date_input("End Date", pd.to_datetime("2023-01-01"))
forecast_days = st.number_input("Number of days to predict", min_value=1, max_value=30, value=7)

if st.button("Predict"):
    # 1. Load and preprocess stock data
    df = download_stock_data(ticker, str(start), str(end))
    df = add_technical_indicators(df)
    st.subheader("Historical Data (last 10 rows)")
    st.dataframe(df.tail(10))

    # 2. Prepare LSTM data
    window = 30
    X, y, scaler = prepare_lstm_data(df, 'Close', window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 3. Train LSTM model (for demo, train on the spot)
    model = create_lstm_model((window, 1))
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # 4. Fetch and analyze stock-specific news (no API key needed)
    headlines = fetch_company_news(ticker)
    if not headlines:
        st.warning("No news headlines found for this ticker.")
        sentiment_scores = []
        sentiment_score = 0
    else:
        sentiment_scores = get_sentiment_scores(headlines)
        sentiment_score = aggregate_sentiment(sentiment_scores)
        st.subheader(f"Latest News Headlines for {ticker} & Sentiment")
        for h, s in zip(headlines, sentiment_scores):
            st.write(f"{h} (Sentiment: {s:.2f})")

    # 5. Multi-day prediction loop
    future_preds_scaled = []
    last_sequence = X[-1]  # shape: (window, 1)

    for _ in range(forecast_days):
        pred_scaled = model.predict(last_sequence.reshape(1, window, 1))[0][0]
        future_preds_scaled.append(pred_scaled)
        last_sequence = np.append(last_sequence[1:], [[pred_scaled]], axis=0)

    future_preds = [inverse_scale(pred, scaler) for pred in future_preds_scaled]
    hybrid_preds = [combine_features(pred, sentiment_score) for pred in future_preds]

    # 6. Display results
    st.subheader(f"Predicted Next {forecast_days} Days (Hybrid Model)")
    for i, price in enumerate(hybrid_preds, 1):
        st.write(f"Day {i}: {price:.2f}")

    # 7. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    ax.plot(future_dates, hybrid_preds, 'ro--', label='Hybrid Forecast')
    ax.set_title(f"{ticker} Close Price & {forecast_days}-Day Hybrid Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
