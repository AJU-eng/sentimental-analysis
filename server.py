from flask import Flask, request, jsonify
from transformers import pipeline
import spacy
from textblob import TextBlob
from multiprocessing import freeze_support
import psycopg2
from datetime import datetime
from urllib.parse import urlparse

# Initialize Flask app
app = Flask(__name__)

# Database connection string
DB_URL = "postgresql://productalyze-db_owner:DjKP6ewhN8vT@ep-broad-darkness-a5bd5gtf.us-east-2.aws.neon.tech/productalyze-db?sslmode=require"

# Parse DB URL for connection parameters
parsed_url = urlparse(DB_URL)

# Database connection function
def get_db_connection():
    return psycopg2.connect(
        dbname=parsed_url.path[1:],
        user=parsed_url.username,
        password=parsed_url.password,
        host=parsed_url.hostname,
        port=parsed_url.port,
        sslmode='require'
    )

# Initialize database tables
# Initialize database tables
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create reviews table
    cur.execute(""" 
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL PRIMARY KEY,
            product_id INTEGER NOT NULL,               
            review_text TEXT NOT NULL,
            overall_sentiment FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create features table
    cur.execute(""" 
        CREATE TABLE IF NOT EXISTS features (
            id SERIAL PRIMARY KEY,
            review_id INTEGER REFERENCES reviews(id),
            feature_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            context TEXT,
            confidence FLOAT,
            sentiment_words TEXT[]
        )
    """)

    # Check if 'product_id' column exists in the 'reviews' table
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'reviews' AND column_name = 'product_id'
    """)
    
    if not cur.fetchone():
        print("Adding 'product_id' column to 'reviews' table")
        cur.execute("ALTER TABLE reviews ADD COLUMN product_id INTEGER NOT NULL")
    
    conn.commit()
    cur.close()
    conn.close()


# Load SpaCy model for feature extraction
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
def init_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
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

@app.route('/api/analyze', methods=['POST'])
def analyze_reviews():
    try:
        # Get the list of products from the request body
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        response_data = []

        for product_data in data:
            product_id = product_data.get('product_id')
            reviews = product_data.get('reviews', [])
            
            if not product_id or not reviews:
                response_data.append({
                    "error": "Missing product_id or reviews for a product"
                })
                continue

            # Process each review for the current product
            for review_text in reviews:
                if not review_text:
                    response_data.append({
                        "error": "Review text is missing"
                    })
                    continue

                # Extract features and their sentiments for the review
                features = extract_features_with_sentiment(review_text)

                if not features:
                    response_data.append({
                        "review": review_text,
                        "error": "No features could be extracted"
                    })
                    continue

                # Calculate overall sentiment
                overall_sentiment = TextBlob(review_text).sentiment.polarity
                overall_score = (overall_sentiment + 1) / 2

                if sentiment_analyzer is not None:
                    try:
                        transformer_sentiment = sentiment_analyzer(review_text[:512])[0]
                        overall_score = 1.0 if transformer_sentiment['label'] == 'POSITIVE' else 0.0
                    except Exception as e:
                        print(f"Error in transformer overall sentiment analysis: {e}")

                # Store in database
                conn = get_db_connection()
                cur = conn.cursor()

                # Insert review with product_id
                cur.execute(
                    "INSERT INTO reviews (review_text, overall_sentiment, product_id) VALUES (%s, %s, %s) RETURNING id",
                    (review_text, overall_score, product_id)  # Include product_id in insert
                )
                review_id = cur.fetchone()[0]

                # Insert features
                for feature in features:
                    cur.execute(""" 
                        INSERT INTO features 
                        (review_id, feature_text, sentiment, context, confidence, sentiment_words)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        review_id,
                        feature["feature"],
                        feature["sentiment"],
                        feature["context"],
                        feature["confidence"],
                        feature["sentiment_words"]
                    ))

                conn.commit()
                cur.close()
                conn.close()

                # Prepare the feature summary for the response
                feature_summary = [{"feature": f["feature"], "sentiment": f["sentiment"]} for f in features]

                # Add review result to the response
                response_data.append({
                    "review_id": review_id,
                    "features": feature_summary,
                    "stored": True
                })

        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing reviews: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize multiprocessing support
    freeze_support()
    
    # Initialize the sentiment analyzer
    try:
        sentiment_analyzer = init_sentiment_analyzer()
    except Exception as e:
        print(f"Warning: Could not initialize transformer model: {e}")
        print("Falling back to TextBlob for sentiment analysis")
    
    # Initialize database
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        exit(1)
    
    # Run the Flask app
    app.run(host='localhost', port=8080, debug=True)
