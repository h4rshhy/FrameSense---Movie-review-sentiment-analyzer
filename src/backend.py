from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import pickle
import os
import json
import random
from collections import defaultdict
from transformers import pipeline

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")

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

    def save_model(self, filename=MODEL_PATH):
        """Save the trained model and vectorizer"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'is_trained': self.is_trained
            }, f)

    def load_model(self, filename=MODEL_PATH):
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
MOVIE_PROFILE_PATH = os.path.join(BASE_DIR, "movie_emotion_profiles.json")

PRESET_MOOD_VECTORS = {
    "happy": {"joy": 0.72, "surprise": 0.12, "neutral": 0.10, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "disgust": 0.02},
    "sad": {"sadness": 0.68, "neutral": 0.12, "joy": 0.07, "fear": 0.07, "surprise": 0.03, "anger": 0.02, "disgust": 0.01},
    "tense": {"fear": 0.48, "surprise": 0.18, "anger": 0.14, "neutral": 0.10, "sadness": 0.05, "disgust": 0.03, "joy": 0.02},
    "angry": {"anger": 0.65, "disgust": 0.13, "fear": 0.07, "sadness": 0.06, "neutral": 0.05, "surprise": 0.03, "joy": 0.01},
    "excited": {"joy": 0.50, "surprise": 0.28, "neutral": 0.10, "fear": 0.04, "sadness": 0.03, "anger": 0.03, "disgust": 0.02},
    "nostalgic": {"sadness": 0.34, "joy": 0.30, "neutral": 0.18, "surprise": 0.08, "fear": 0.04, "anger": 0.03, "disgust": 0.03},
    "adventurous": {"surprise": 0.34, "joy": 0.28, "fear": 0.14, "neutral": 0.12, "anger": 0.05, "sadness": 0.04, "disgust": 0.03},
    "relaxed": {"neutral": 0.52, "joy": 0.24, "sadness": 0.08, "surprise": 0.06, "fear": 0.04, "anger": 0.03, "disgust": 0.03}
}

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
movie_profiles = []

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "movie", "film", "my", "of", "on", "or", "our", "so", "that", "the",
    "their", "them", "they", "this", "to", "too", "was", "were", "with", "you", "your"
}

ASPECT_KEYWORDS = {
    "story": {"story", "plot", "narrative", "screenplay", "script", "ending", "twist", "writing"},
    "characters": {"character", "characters", "arc", "relationship", "relationships", "chemistry"},
    "acting": {"acting", "actor", "actors", "actress", "performances", "performance", "cast"},
    "visuals": {"visual", "visuals", "cinematography", "shot", "shots", "animated", "animation", "spectacle"},
    "pacing": {"pacing", "slow", "dragging", "dragged", "rushed", "tight", "boring"},
    "emotion": {"emotional", "heart", "heartfelt", "moving", "sad", "joyful", "hopeful", "touching"},
    "humor": {"funny", "humor", "humorous", "hilarious", "comedy", "jokes", "witty"},
    "tension": {"tense", "tension", "thrilling", "thriller", "suspense", "suspenseful", "mystery"},
    "action": {"action", "exciting", "battle", "fight", "adventure", "adrenaline"},
    "music": {"music", "score", "soundtrack", "songs", "musical"},
    "originality": {"original", "unique", "fresh", "inventive", "creative", "different"}
}

POSITIVE_HINTS = {
    "love", "loved", "like", "liked", "great", "amazing", "excellent", "fantastic", "strong",
    "beautiful", "brilliant", "smart", "sharp", "fun", "funny", "moving", "emotional",
    "tense", "thrilling", "engaging", "gripping", "powerful", "touching", "hopeful"
}

NEGATIVE_HINTS = {
    "hate", "hated", "dislike", "disliked", "bad", "weak", "boring", "slow", "messy",
    "flat", "predictable", "confusing", "dull", "annoying", "rushed", "dragging",
    "dragged", "underwhelming", "generic", "bland"
}

def normalize_emotion_scores(emotions):
    normalized = {label: float(emotions.get(label, 0.0)) for label in EMOTION_LABELS}
    total = sum(normalized.values())
    if total > 0:
        normalized = {label: value / total for label, value in normalized.items()}
    return normalized

def load_movie_profiles():
    if not os.path.exists(MOVIE_PROFILE_PATH):
        return []

    try:
        with open(MOVIE_PROFILE_PATH, 'r', encoding='utf-8') as f:
            loaded_profiles = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠️ Could not load movie profiles: {exc}")
        return []

    prepared_profiles = []
    for profile in loaded_profiles:
        emotions = normalize_emotion_scores(profile.get("emotion_profile", {}))
        prepared_profile = {
            "title": profile.get("title"),
            "poster_url": profile.get("poster_url"),
            "rating": float(profile.get("rating") or 0.0),
            "vote_count": int(profile.get("vote_count") or 0),
            "overview": profile.get("overview") or "Overview unavailable.",
            "release_year": profile.get("release_year"),
            "dominant_emotion": profile.get("dominant_emotion") or max(emotions, key=emotions.get),
            "emotion_profile": emotions
        }
        prepared_profiles.append(prepared_profile)

    return prepared_profiles

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


def cosine_similarity_score(vector_a, vector_b):
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if not norm_a or not norm_b:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def tokenize_text(text):
    return [
        token for token in re.findall(r"\b[a-zA-Z']+\b", (text or "").lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def extract_review_preferences(review):
    lower_review = (review or "").lower()
    tokens = tokenize_text(lower_review)
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1

    positive_aspects = set()
    negative_aspects = set()

    for aspect, keywords in ASPECT_KEYWORDS.items():
        keyword_hits = keywords.intersection(token_counts.keys())
        if not keyword_hits:
            continue

        positive_signal = any(word in lower_review for word in POSITIVE_HINTS)
        negative_signal = any(word in lower_review for word in NEGATIVE_HINTS)

        if positive_signal and not negative_signal:
            positive_aspects.add(aspect)
        elif negative_signal and not positive_signal:
            negative_aspects.add(aspect)
        else:
            nearest_positive = 0
            nearest_negative = 0
            for sentence in re.split(r"[.!?]+", lower_review):
                if not sentence.strip():
                    continue
                has_aspect = any(keyword in sentence for keyword in keywords)
                if not has_aspect:
                    continue
                if any(word in sentence for word in POSITIVE_HINTS):
                    nearest_positive += 1
                if any(word in sentence for word in NEGATIVE_HINTS):
                    nearest_negative += 1
            if nearest_positive > nearest_negative:
                positive_aspects.add(aspect)
            elif nearest_negative > nearest_positive:
                negative_aspects.add(aspect)

    return {
        "positive_aspects": sorted(positive_aspects),
        "negative_aspects": sorted(negative_aspects),
        "keywords": list(dict.fromkeys(tokens))[:12]
    }


def score_movie_aspects(movie):
    text = " ".join([
        movie.get("title", ""),
        movie.get("overview", ""),
        movie.get("dominant_emotion", "")
    ]).lower()

    aspect_scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword in text)
        aspect_scores[aspect] = min(1.0, matches / 2) if matches else 0.0

    dominant_emotion = movie.get("dominant_emotion")
    if dominant_emotion == "joy":
        aspect_scores["humor"] = max(aspect_scores.get("humor", 0.0), 0.45)
        aspect_scores["emotion"] = max(aspect_scores.get("emotion", 0.0), 0.35)
    if dominant_emotion == "fear":
        aspect_scores["tension"] = max(aspect_scores.get("tension", 0.0), 0.55)
    if dominant_emotion == "surprise":
        aspect_scores["story"] = max(aspect_scores.get("story", 0.0), 0.45)
        aspect_scores["originality"] = max(aspect_scores.get("originality", 0.0), 0.4)

    return aspect_scores


def overlap_score(review_keywords, movie):
    if not review_keywords:
        return 0.0
    movie_tokens = set(tokenize_text(f"{movie.get('title', '')} {movie.get('overview', '')}"))
    if not movie_tokens:
        return 0.0
    overlap = len(set(review_keywords).intersection(movie_tokens))
    return overlap / max(1, min(8, len(set(review_keywords))))


def build_recommendation_reason(movie, shared_emotions, positive_aspects, improved_aspects):
    parts = []
    if shared_emotions:
        parts.append(f"keeps the same {', '.join(shared_emotions)} energy")
    if positive_aspects:
        parts.append(f"leans into what you liked about the {', '.join(positive_aspects[:2])}")
    if improved_aspects:
        parts.append(f"should land better on {', '.join(improved_aspects[:2])}")
    if not parts:
        parts.append("is emotionally close to the tone of your review")

    return f"This pick {' and '.join(parts)}."

def build_mood_vector(preset_mood=None, free_text=None):
    vectors = []

    if preset_mood:
        preset_scores = PRESET_MOOD_VECTORS.get(preset_mood)
        if not preset_scores:
            raise ValueError("Invalid preset mood.")
        vectors.append(build_emotion_vector(normalize_emotion_scores(preset_scores)))

    if free_text:
        free_text_scores = normalize_emotion_scores(get_emotions(free_text))
        vectors.append(build_emotion_vector(free_text_scores))

    if not vectors:
        raise ValueError("At least one mood input is required.")

    return np.mean(vectors, axis=0)

def recommend_movies(review, emotions, sentiment_result, top_n=3):
    candidate_movies = movie_profiles if movie_profiles else MOVIE_CATALOG
    review_vector = build_emotion_vector(emotions)
    recommendations = []
    top_review_emotions = [label for label, _ in get_top_emotions(emotions, 2)]
    review_preferences = extract_review_preferences(review)
    positive_aspects = review_preferences["positive_aspects"]
    negative_aspects = review_preferences["negative_aspects"]
    review_keywords = review_preferences["keywords"]

    for movie in candidate_movies:
        movie_emotions = movie.get("emotion_profile") or get_movie_fingerprint(movie)
        movie_vector = build_emotion_vector(movie_emotions)
        similarity = cosine_similarity_score(review_vector, movie_vector)
        aspect_scores = score_movie_aspects(movie)
        liked_alignment = (
            sum(aspect_scores.get(aspect, 0.0) for aspect in positive_aspects) / len(positive_aspects)
            if positive_aspects else 0.0
        )
        improvement_alignment = (
            sum((1 - aspect_scores.get(aspect, 0.0)) for aspect in negative_aspects) / len(negative_aspects)
            if negative_aspects else 0.0
        )
        keyword_overlap = overlap_score(review_keywords, movie)
        rating_quality = max(0.0, min(1.0, float(movie.get("rating", 0.0)) / 10))

        final_score = (
            0.45 * similarity
            + 0.20 * liked_alignment
            + 0.15 * improvement_alignment
            + 0.10 * keyword_overlap
            + 0.10 * rating_quality
        )

        shared_emotions = [
            label for label, _ in get_top_emotions(movie_emotions, 3)
            if label in top_review_emotions
        ]
        improved_aspects = [
            aspect for aspect in negative_aspects
            if aspect_scores.get(aspect, 0.0) < 0.35
        ]
        reason = build_recommendation_reason(movie, shared_emotions, positive_aspects, improved_aspects)

        recommendations.append({
            "title": movie["title"],
            "year": movie.get("year") or movie.get("release_year"),
            "poster_url": movie.get("poster_url"),
            "overview": movie["overview"],
            "match_score": round(final_score, 3),
            "reason": reason,
            "sentiment_fit": sentiment_result["sentiment"],
            "rating": round(float(movie.get("rating", 0.0)), 1)
        })

    recommendations = sorted(
        recommendations,
        key=lambda item: (item["match_score"], item.get("rating", 0.0)),
        reverse=True
    )

    candidate_pool = recommendations[:min(18, len(recommendations))]
    if len(candidate_pool) <= top_n:
        return candidate_pool[:top_n]

    random.shuffle(candidate_pool)
    selected = sorted(
        candidate_pool[:top_n],
        key=lambda item: (item["match_score"], item.get("rating", 0.0)),
        reverse=True
    )
    return selected

movie_profiles = load_movie_profiles()

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
        'model_ready': analyzer.is_trained,
        'movie_profiles_loaded': len(movie_profiles)
    })

@app.route('/api/suggest', methods=['POST'])
def suggest_movies():
    try:
        data = request.get_json() or {}
        preset_mood = (data.get('preset_mood') or '').strip().lower()
        free_text = (data.get('free_text') or '').strip()

        if not preset_mood and not free_text:
            return jsonify({'error': 'Provide preset_mood, free_text, or both.'}), 400

        if not movie_profiles:
            return jsonify({'error': 'Movie profiles are unavailable. Generate movie_emotion_profiles.json first.'}), 503

        mood_vector = build_mood_vector(preset_mood=preset_mood or None, free_text=free_text or None).reshape(1, -1)
        results = []

        for movie in movie_profiles:
            if movie['vote_count'] < 50:
                continue

            movie_vector = build_emotion_vector(movie['emotion_profile']).reshape(1, -1)
            similarity = float(cosine_similarity(mood_vector, movie_vector)[0][0])
            normalized_similarity = max(0.0, min(1.0, (similarity + 1) / 2))
            normalized_rating = max(0.0, min(1.0, movie['rating'] / 10))
            final_score = (0.7 * normalized_similarity) + (0.3 * normalized_rating)

            results.append({
                'title': movie['title'],
                'poster': movie['poster_url'],
                'rating': round(movie['rating'], 1),
                'dominant_emotion': movie['dominant_emotion'],
                'release_year': movie['release_year'],
                'overview': movie['overview'],
                'emotional_match': round(final_score * 100, 1)
            })

        results = sorted(results, key=lambda item: item['emotional_match'], reverse=True)
        if not results:
            return jsonify({'results': [], 'total_candidates': 0})

        candidate_pool = sorted(results, key=lambda item: item['emotional_match'], reverse=True)
        pool_size = min(len(candidate_pool), max(10, min(30, len(candidate_pool))))
        candidate_pool = candidate_pool[:pool_size]
        selection_size = min(10, len(candidate_pool))
        selection = random.sample(candidate_pool, selection_size)
        selection = sorted(selection, key=lambda item: item['emotional_match'], reverse=True)

        return jsonify({
            'results': selection,
            'total_candidates': len(candidate_pool)
        })

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
