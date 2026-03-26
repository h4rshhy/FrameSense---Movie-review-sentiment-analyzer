from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import pickle
import os
from transformers import pipeline

app = Flask(__name__)
CORS(app)

class MovieSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if isinstance(text, str):
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Convert to lowercase
            text = text.lower()
            # Remove extra whitespace   
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""

    def train(self, df, text_column='review', sentiment_column='sentiment'):
        """Train the sentiment analysis model"""
        # Preprocess the text data
        df['cleaned_review'] = df[text_column].apply(self.preprocess_text)
        
        # Prepare features and labels
        X = df['cleaned_review']
        y = df[sentiment_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize the text data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'test_size': len(X_test),
            'training_size': len(X_train)
        }

    def predict_sentiment(self, text):
        """Predict sentiment for a single review"""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
            
        cleaned_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return {
            'sentiment': prediction,
            'confidence': max(probability),
            'probabilities': {
                'positive': probability[1] if len(probability) > 1 else probability[0],
                'negative': probability[0] if len(probability) > 1 else probability[1]
            }
        }

    def analyze_movie_reviews(self, reviews):
        """Analyze multiple reviews and calculate movie rating"""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
            
        predictions = []
        positive_count = 0
        negative_count = 0
        total_confidence = 0
        
        for review in reviews:
            result = self.predict_sentiment(review)
            predictions.append(result)
            
            if result['sentiment'] == 'positive':
                positive_count += 1
            else:
                negative_count += 1
                
            total_confidence += result['confidence']
        
        total_reviews = len(reviews)
        positive_percentage = (positive_count / total_reviews) * 100
        negative_percentage = (negative_count / total_reviews) * 100
        avg_confidence = total_confidence / total_reviews
        
        # Calculate movie rating (1-10 scale based on positive percentage)
        movie_rating = round((positive_percentage / 100) * 10, 1)
        
        return {
            'total_reviews': total_reviews,
            'positive_reviews': positive_count,
            'negative_reviews': negative_count,
            'positive_percentage': round(positive_percentage, 2),
            'negative_percentage': round(negative_percentage, 2),
            'average_confidence': round(avg_confidence, 2),
            'movie_rating': movie_rating,
            'individual_predictions': predictions
        }

    def save_model(self, filename='sentiment_model.pkl'):
        """Save the trained model and vectorizer"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'is_trained': self.is_trained
            }, f)

    def load_model(self, filename='sentiment_model.pkl'):
        """Load a trained model and vectorizer"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.is_trained = data['is_trained']

# Initialize the analyzer
analyzer = MovieSentimentAnalyzer()

emotion_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

MOVIE_CATALOG = [
    {
        "title": "Inside Out",
        "year": 2015,
        "poster_url": "https://image.tmdb.org/t/p/w500/lRHE0vzf3oYJrhbsHXjIkF4Tl5A.jpg",
        "overview": "A heartfelt animated story about a young girl's emotions learning to work together through change, memory, sadness, joy, and growing up.",
        "why_it_hits": "emotionally reflective and character-driven"
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "poster_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "overview": "A tense crime epic where a heroic vigilante faces chaos, fear, sacrifice, and moral conflict in a city pushed to the edge.",
        "why_it_hits": "morally complex and intense"
    },
    {
        "title": "La La Land",
        "year": 2016,
        "poster_url": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
        "overview": "A bittersweet romantic musical about ambition, love, hope, disappointment, and the cost of chasing dreams.",
        "why_it_hits": "bittersweet and aspirational"
    },
    {
        "title": "Parasite",
        "year": 2019,
        "poster_url": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
        "overview": "A sharp social thriller blending tension, class anger, dark humor, surprise, and shocking emotional turns.",
        "why_it_hits": "tense, surprising, and socially sharp"
    },
    {
        "title": "The Shawshank Redemption",
        "year": 1994,
        "poster_url": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
        "overview": "A moving prison drama about injustice, endurance, friendship, sadness, hope, and eventual emotional release.",
        "why_it_hits": "hopeful despite emotional weight"
    },
    {
        "title": "Whiplash",
        "year": 2014,
        "poster_url": "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
        "overview": "An intense character study driven by obsession, pressure, anger, fear, and the thrill of artistic excellence.",
        "why_it_hits": "high-pressure and performance-focused"
    },
    {
        "title": "Little Miss Sunshine",
        "year": 2006,
        "poster_url": None,
        "overview": "An offbeat family road movie mixing warmth, awkward humor, vulnerability, disappointment, and joy.",
        "why_it_hits": "messy, human, and uplifting"
    },
    {
        "title": "Gone Girl",
        "year": 2014,
        "poster_url": "https://image.tmdb.org/t/p/w500/ts996lKsxvjkO2yiYG0ht4qAicO.jpg",
        "overview": "A dark psychological mystery full of distrust, manipulation, fear, disgust, surprise, and emotional unease.",
        "why_it_hits": "dark and psychologically unsettling"
    }
]

movie_fingerprint_cache = {}

def get_emotions(text):
    results = emotion_pipe(text[:512], top_k=None, truncation=True)
    if results and isinstance(results[0], dict):
        emotion_results = results
    else:
        emotion_results = results[0]
    emotion_results = sorted(emotion_results, key=lambda r: r['score'], reverse=True)
    return {r['label']: round(r['score'], 3) for r in emotion_results}

def get_top_emotions(emotions, top_n=3):
    return sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:top_n]

def classify_reviewer_style(review):
    sentences = [s.strip() for s in re.split(r'[.!?]+', review) if s.strip()]
    words = re.findall(r'\b\w+\b', review.lower())
    word_count = len(words)
    sentence_count = len(sentences) if sentences else 1
    avg_sentence_length = word_count / sentence_count if word_count else 0
    avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0
    unique_ratio = len(set(words)) / word_count if word_count else 0
    first_person_count = sum(1 for word in words if word in {"i", "me", "my", "mine"})
    exclamation_count = review.count('!')
    film_terms = [
        "cinematography", "narrative", "protagonist", "antagonist", "motif",
        "pacing", "screenplay", "direction", "performance", "performances",
        "arc", "editing", "score", "character", "thematic", "tone"
    ]
    film_term_count = sum(1 for term in film_terms if term in review.lower())

    critic_score = (
        film_term_count * 0.45
        + avg_sentence_length * 0.08
        + avg_word_length * 0.35
        + unique_ratio * 1.8
        - exclamation_count * 0.2
        - first_person_count * 0.08
    )

    style = "critic-like" if critic_score >= 3 else "audience-like"
    tone = "analytical" if film_term_count >= 2 or avg_sentence_length >= 18 else "emotional"
    focus = "performance-focused" if any(term in review.lower() for term in ["acting", "actor", "actress", "performance", "performances", "cast"]) else "story-focused"

    explanation_parts = []
    if film_term_count:
        explanation_parts.append(f"uses {film_term_count} film-analysis term(s)")
    if avg_sentence_length >= 18:
        explanation_parts.append("has longer, more developed sentences")
    if first_person_count >= 3 or exclamation_count >= 1:
        explanation_parts.append("sounds personal and immediate")
    if not explanation_parts:
        explanation_parts.append("uses straightforward review language")

    return {
        "style": style,
        "critic_score": round(critic_score, 2),
        "tone": tone,
        "focus": focus,
        "metrics": {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "vocabulary_richness": round(unique_ratio, 2),
            "film_term_count": film_term_count
        },
        "explanation": ", ".join(explanation_parts)
    }

def build_mismatch_insight(sentiment_result, emotions):
    positive_emotion_score = emotions.get("joy", 0) + (emotions.get("surprise", 0) * 0.35)
    heavy_emotion_score = (
        emotions.get("sadness", 0)
        + emotions.get("fear", 0)
        + emotions.get("anger", 0)
        + emotions.get("disgust", 0)
    )
    dominant_emotions = [label for label, _ in get_top_emotions(emotions, 2)]
    sentiment = sentiment_result["sentiment"]

    mismatch = False
    mismatch_type = "aligned"
    if sentiment == "positive" and heavy_emotion_score > positive_emotion_score + 0.2:
        mismatch = True
        mismatch_type = "positive sentiment, heavy emotional undertone"
        summary = "The review is positive overall, but its strongest emotional signals are darker or heavier."
    elif sentiment == "negative" and positive_emotion_score > heavy_emotion_score + 0.2:
        mismatch = True
        mismatch_type = "negative sentiment, warm emotional undertone"
        summary = "The review reads negative overall, but it still contains warmth or admiration in its emotional tone."
    else:
        summary = "The dominant emotions are broadly aligned with the overall sentiment."

    if mismatch:
        detail = f"Top emotions: {', '.join(dominant_emotions)}."
    else:
        detail = f"Top emotions support the {sentiment} verdict: {', '.join(dominant_emotions)}."

    return {
        "mismatch": mismatch,
        "type": mismatch_type,
        "summary": summary,
        "detail": detail,
        "heavy_emotion_score": round(heavy_emotion_score, 3),
        "positive_emotion_score": round(positive_emotion_score, 3)
    }

def get_movie_fingerprint(movie):
    cache_key = movie["title"]
    if cache_key not in movie_fingerprint_cache:
        movie_fingerprint_cache[cache_key] = get_emotions(movie["overview"])
    return movie_fingerprint_cache[cache_key]

def build_emotion_vector(emotions):
    return np.array([emotions.get(label, 0) for label in EMOTION_LABELS], dtype=float)

def recommend_movies(review, emotions, sentiment_result, top_n=3):
    review_vector = build_emotion_vector(emotions)
    review_norm = np.linalg.norm(review_vector)
    recommendations = []
    top_review_emotions = [label for label, _ in get_top_emotions(emotions, 2)]

    for movie in MOVIE_CATALOG:
        movie_emotions = get_movie_fingerprint(movie)
        movie_vector = build_emotion_vector(movie_emotions)
        movie_norm = np.linalg.norm(movie_vector)
        similarity = 0.0
        if review_norm and movie_norm:
            similarity = float(np.dot(review_vector, movie_vector) / (review_norm * movie_norm))

        shared_emotions = [
            label for label, _ in get_top_emotions(movie_emotions, 3)
            if label in top_review_emotions
        ]
        reason = f"Matches the review's {', '.join(top_review_emotions)} profile and feels {movie['why_it_hits']}."
        if shared_emotions:
            reason = f"Shared emotional signals in {', '.join(shared_emotions)} make this a strong match; it also feels {movie['why_it_hits']}."

        recommendations.append({
            "title": movie["title"],
            "year": movie["year"],
            "poster_url": movie.get("poster_url"),
            "overview": movie["overview"],
            "match_score": round(similarity, 3),
            "reason": reason,
            "sentiment_fit": sentiment_result["sentiment"]
        })

    recommendations = sorted(recommendations, key=lambda item: item["match_score"], reverse=True)
    return recommendations[:top_n]

# Try to load existing model
try:
    analyzer.load_model()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"ℹ️ No pre-trained model found. Please train the model first. Error: {e}")

@app.route('/')
def home():
    return jsonify({"message": "Movie Sentiment Analyzer API", "status": "running"})

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        print("Training endpoint called")  # Debug log
        
        if 'file' not in request.files:
            print("No file in request.files")  # Debug log
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")  # Debug log
        
        if file.filename == '':
            print("Empty filename")  # Debug log
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            print("Reading CSV file...")  # Debug log
            df = pd.read_csv(file)
            print(f"CSV read successfully. Shape: {df.shape}")  # Debug log
            
            # Check required columns
            if 'review' not in df.columns or 'sentiment' not in df.columns:
                print(f"Missing required columns. Found: {df.columns.tolist()}")  # Debug log
                return jsonify({'error': 'CSV must contain "review" and "sentiment" columns'}), 400
            
            print("Starting model training...")  # Debug log
            # Train the model
            result = analyzer.train(df)
            analyzer.save_model()
            
            print("Model trained successfully!")  # Debug log
            return jsonify({
                'message': 'Model trained successfully',
                'accuracy': result['accuracy'],
                'training_size': result['training_size'],
                'test_size': result['test_size'],
                'sentiment_distribution': df['sentiment'].value_counts().to_dict()
            })
        else:
            print(f"Invalid file type: {file.filename}")  # Debug log
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        print(f"Training error: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        if not data or 'review' not in data:
            return jsonify({'error': 'No review provided'}), 400
        
        review = data['review']
        result = analyzer.predict_sentiment(review)
        emotions = get_emotions(review)
        result['emotions'] = emotions
        result['reviewer_style'] = classify_reviewer_style(review)
        result['mismatch_insight'] = build_mismatch_insight(result, emotions)
        result['recommendations'] = recommend_movies(review, emotions, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-movie', methods=['POST'])
def analyze_movie():
    try:
        data = request.get_json()
        if not data or 'reviews' not in data:
            return jsonify({'error': 'No reviews provided'}), 400
        
        reviews = data['reviews']
        result = analyzer.analyze_movie_reviews(reviews)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_trained': analyzer.is_trained,
        'model_ready': analyzer.is_trained
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
