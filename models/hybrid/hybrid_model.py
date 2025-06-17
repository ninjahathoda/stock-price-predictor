def combine_features(lstm_pred, sentiment_score):
    # Weighted sum: tune weights as needed
    return 0.8 * lstm_pred + 0.2 * (lstm_pred * (1 + sentiment_score / 10))
