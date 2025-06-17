def aggregate_sentiment(sentiment_scores):
    # Weighted average or other advanced aggregation can be used
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
