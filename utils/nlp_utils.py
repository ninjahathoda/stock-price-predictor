from textblob import TextBlob

def get_sentiment_scores(headlines):
    """
    Returns a list of polarity scores (from -1 to 1) for each headline.
    """
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return scores
