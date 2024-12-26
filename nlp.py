from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import nltk
import pandas as pd

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load Spacy model for feature extraction
nlp = spacy.load("en_core_web_sm")

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Example reviews
reviews = [
    "The camera quality of this phone is amazing! I love the battery life too.",
    "Terrible service! The staff was rude, and the checkout process was too slow.",
    "Absolutely fantastic design and performance. The build quality is top-notch!",
    "The app crashes often but has great features when it works.",
    "The delivery was quick, but the packaging was terrible.",
    "poor quality sound.",
    "meesho is a good platform,but it is not better than amazon"
]

# Analyze Sentiment and Extract Features
results = []
feature_sentiment_summary = []  # To store features and their overall sentiment

# List of personal pronouns to exclude
personal_pronouns = {"i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his",
                     "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}

for review in reviews:
    # Sentiment analysis of the full review
    sentiment_scores = sia.polarity_scores(review)
    sentiment = "Positive" if sentiment_scores['compound'] > 0 else "Negative" 

    # Feature extraction using SpaCy
    doc = nlp(review)
    features = [chunk.text for chunk in doc.noun_chunks if chunk.root.text.lower() not in personal_pronouns]

    # Analyze each feature's sentiment
    for feature in features:
        feature_score = sia.polarity_scores(feature)['compound']
        feature_sentiment = "Good" if feature_score > 0 else "Bad" 
        feature_sentiment_summary.append({
            "Feature": feature,
            "Sentiment": feature_sentiment
        })

    # Append review-level results
    results.append({
        "Review": review,
        "Features": ", ".join(features),
        "Overall Sentiment": sentiment
    })

# Convert results to DataFrames
df_reviews = pd.DataFrame(results)
df_features = pd.DataFrame(feature_sentiment_summary)

# Print the DataFrames
print("Review-Level Sentiment Analysis:")
print(df_reviews)
print("\nFeature-Level Sentiment Analysis:")
print(df_features)

# Save the results to CSV files
df_reviews.to_csv("review_sentiment_analysis.csv", index=False)
df_features.to_csv("feature_sentiment_analysis.csv", index=False)
