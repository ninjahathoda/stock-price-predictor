# 📈 Stock Price Predictor

An interactive dashboard for predicting stock prices using machine learning and technical indicators, built with Streamlit.

---

## Features

- Fetches historical stock data from Yahoo Finance
- Calculates technical indicators (Simple Moving Averages, RSI)
- Trains a Linear Regression model for price prediction
- Evaluates model performance (MAE, MSE, R²)
- Visualizes actual vs. predicted prices
- Modular and easy to extend with more models or features

---

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

3. **Run the Streamlit app:**

---

## Usage

- Enter a stock ticker (e.g., `AAPL`)
- Select a date range
- Click **Predict**
- View predictions, evaluation metrics, and comparison plots

---

## How It Works

1. **Downloads stock data** using yfinance
2. **Calculates technical indicators** (SMA, RSI)
3. **Splits data** into training and test sets
4. **Trains a Linear Regression model**
5. **Predicts and evaluates** closing prices
6. **Plots actual vs. predicted prices**

---

## Extending the Project

- Add more technical indicators (MACD, Bollinger Bands, etc.)
- Integrate advanced models (Random Forest, XGBoost, LSTM)
- Add news sentiment analysis
- Backtest simple trading strategies
- Deploy online with Streamlit Community Cloud or Heroku

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

---

## Disclaimer

This project is for educational purposes only. Stock price prediction is inherently uncertain and should not be used for investment advice.

---

## Credits

- [yfinance](https://github.com/ranaroussi/yfinance)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
