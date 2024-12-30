from flask import Flask, request, jsonify
from transformers import pipeline
import spacy
import pandas as pd
from textblob import TextBlob
from multiprocessing import freeze_support

# Initialize Flask app
app = Flask(__name__)

# Load SpaCy model for feature extraction
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
def init_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",  # Using a simpler model
        device=-1  # Run on CPU
    )

# Global sentiment analyzer variable
sentiment_analyzer = None

def extract_features_with_sentiment(text):
    doc = nlp(text)
    features = []
    
    # Process each sentence to maintain context
    for sent in doc.sents:
        sent_doc = nlp(sent.text)
        
        # Extract noun phrases that might be features
        for chunk in sent_doc.noun_chunks:
            # Skip if the root word is a pronoun
            if chunk.root.pos_ == 'PRON':
                continue
                
            # Skip if chunk contains any pronouns
            if any(token.pos_ == 'PRON' for token in chunk):
                continue
                
            # Look for noun phrases that are likely product features
            if chunk.root.dep_ in ('nsubj', 'dobj', 'pobj'):
                # Get the surrounding context
                start = max(0, chunk.start - 2)
                end = min(len(sent_doc), chunk.end + 2)
                context = sent_doc[start:end].text
                
                # Get the associated sentiment words
                sentiment_words = []
                for token in sent_doc:
                    if token.dep_ in ('amod', 'advmod') and token.head == chunk.root:
                        sentiment_words.append(token.text)
                    elif token.dep_ == 'acomp' and any(w.text == chunk.root.text for w in token.head.children):
                        sentiment_words.append(token.text)
                
                # Use TextBlob for sentiment analysis as a fallback
                blob_sentiment = TextBlob(context).sentiment.polarity
                
                # Use transformer model if available
                if sentiment_analyzer is not None:
                    try:
                        sentiment_result = sentiment_analyzer(context)[0]
                        sentiment_score = 1.0 if sentiment_result['label'] == 'POSITIVE' else 0.0
                    except Exception as e:
                        print(f"Error in transformer sentiment analysis: {e}")
                        sentiment_score = (blob_sentiment + 1) / 2  # Convert -1,1 to 0,1 scale
                else:
                    sentiment_score = (blob_sentiment + 1) / 2
                
                # Determine final sentiment
                final_sentiment = "Positive" if sentiment_score > 0.6 else \
                                "Negative" if sentiment_score < 0.4 else "Neutral"
                
                features.append({
                    "feature": chunk.text.lower(),
                    "sentiment": final_sentiment,
                    "context": context,
                    "confidence": sentiment_score,
                    "sentiment_words": sentiment_words
                })
    
    return features

@app.route('/api/analyze', methods=['GET','POST'])
def analyze_review():
    data = request.get_json()
    review_text = data.get('review', '')

    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    # Extract features and their sentiments
    features = extract_features_with_sentiment(review_text)
    
    # Calculate overall sentiment using TextBlob as fallback
    overall_sentiment = TextBlob(review_text).sentiment.polarity
    overall_score = (overall_sentiment + 1) / 2  # Convert to 0-1 scale
    
    # Try transformer model for overall sentiment if available
    if sentiment_analyzer is not None:
        try:
            transformer_sentiment = sentiment_analyzer(review_text[:512])[0]
            overall_score = 1.0 if transformer_sentiment['label'] == 'POSITIVE' else 0.0
        except Exception as e:
            print(f"Error in transformer overall sentiment analysis: {e}")

    # Prepare structured response
    feature_summary = []
    for feature in features:
        feature_summary.append({
            "feature": feature["feature"],
            "sentiment": feature["sentiment"],
            "context": feature["context"],
            "confidence": round(feature["confidence"], 2),
            "supporting_words": feature["sentiment_words"]
        })

    response = {
        "review": review_text,
        "overall_sentiment": {
            "score": round(overall_score, 2),
            "label": "Positive" if overall_score > 0.6 else "Negative" if overall_score < 0.4 else "Neutral"
        },
        "features": feature_summary
    }
    
    return jsonify(response)

if __name__ == '__main__':
    # Initialize multiprocessing support
    freeze_support()
    
    # Initialize the sentiment analyzer
    try:
        sentiment_analyzer = init_sentiment_analyzer()
    except Exception as e:
        print(f"Warning: Could not initialize transformer model: {e}")
        print("Falling back to TextBlob for sentiment analysis")
    
    # Run the Flask app
    app.run(host='localhost', port=8080, debug=True)