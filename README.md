# FrameSense

FrameSense is a movie review sentiment analysis app with a styled Streamlit frontend, a Flask backend, emotion-aware recommendations, and a precomputed movie profile library.

It started as a basic sentiment classifier and has been expanded into a richer experience that can:

- train a movie-review sentiment model from CSV data
- analyze a single review for positive or negative sentiment
- extract an emotional fingerprint from review text
- estimate reviewer style and tone
- compare sentiment versus emotional undertone
- recommend what to watch next based on the review
- score a movie from multiple reviews
- suggest movies from a mood prompt
- browse a poster-based movie profile explorer

## Features

### 1. Train A Sentiment Model

Upload a CSV with:

- `review`
- `sentiment`

The backend trains a TF-IDF + Logistic Regression model and saves it to `src/sentiment_model.pkl`.

### 2. Single Review Analysis

For one review, the app returns:

- sentiment prediction
- confidence and class probabilities
- emotional fingerprint using a Transformers emotion model
- reviewer style classification
- sentiment/emotion mismatch insight
- personalized movie recommendations

### 3. Movie Rating Analysis

Paste multiple reviews for the same movie and the app estimates:

- overall movie rating on a 10-point scale
- positive/negative review split
- average confidence
- individual review predictions

### 4. Movie Ratings Explorer

The frontend can browse a precomputed catalog from `src/movie_emotion_profiles.json`, including:

- poster
- release year
- rating
- overview
- dominant emotion

### 5. Mood-Based Suggestions

Users can choose a preset mood, add a free-text mood description, or combine both. The backend then returns emotionally aligned movie suggestions from the profile library.

## Tech Stack

- `Flask` for the API
- `Streamlit` for the frontend
- `scikit-learn` for sentiment classification
- `transformers` for emotion classification
- `pandas` and `numpy` for data handling
- `plotly` for charts
- `requests` for API calls and TMDB enrichment

## Project Structure

```text
FrameSense---Movie-review-sentiment-analyzer
├── README.md
├── requirements.txt
├── src
│   ├── backend.py
│   ├── frontend.py
│   ├── build_movie_emotion_profiles.py
│   ├── movie_emotion_profiles.json
│   └── sentiment_model.pkl
└── archive/                      # local raw dataset folder, gitignored
```

## Setup

### 1. Clone The Repo

```bash
git clone https://github.com/hxrshlol/FrameSense---Movie-review-sentiment-analyzer.git
cd FrameSense---Movie-review-sentiment-analyzer
```

### 2. Create And Activate A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run The App

Start the backend:

```bash
python3 src/backend.py
```

In a second terminal, start the frontend:

```bash
streamlit run src/frontend.py --server.headless true --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## API Endpoints

The Flask backend exposes these routes:

- `GET /api/status`  
  Returns model state and number of loaded movie profiles.

- `POST /api/train`  
  Trains the sentiment model from an uploaded CSV.

- `POST /api/predict`  
  Runs single-review sentiment + emotion + recommendation analysis.

- `POST /api/analyze-movie`  
  Scores a movie from multiple reviews.

- `POST /api/suggest`  
  Returns mood-based movie suggestions.

## Movie Profile Builder

The file `src/build_movie_emotion_profiles.py` builds the movie catalog from Rotten Tomatoes review data and can enrich it with TMDB metadata such as:

- posters
- overview text
- ratings
- vote counts

Example command:

```bash
python3 src/build_movie_emotion_profiles.py \
  --output src/movie_emotion_profiles.json \
  --max-movies 420 \
  --max-reviews-per-movie 30 \
  --target-per-emotion 40 \
  --tmdb-api-key YOUR_TMDB_API_KEY
```

You can also pass the API key as an environment variable:

```bash
export TMDB_API_KEY=YOUR_TMDB_API_KEY
```

## Dataset Notes

- The repo includes the generated `movie_emotion_profiles.json`.
- The raw `archive/` dataset is intentionally gitignored because the source CSVs are too large for a normal GitHub push.
- If you want to rebuild the catalog from scratch, place the Rotten Tomatoes CSVs in `archive/`:
  - `archive/rotten_tomatoes_movies.csv`
  - `archive/rotten_tomatoes_critic_reviews.csv`

## Current State

At the time of writing, the app includes:

- a trained sentiment model file in `src/sentiment_model.pkl`
- a precomputed movie profile catalog with poster data
- a redesigned cinematic frontend
- randomized but relevant recommendation logic

## Example CSV Format

```csv
review,sentiment
"This movie was absolutely fantastic with great acting!",positive
"Terrible film with poor acting and a boring plot.",negative
"Amazing cinematography and wonderful performances.",positive
```

## Notes

- The sentiment model must be trained or loaded before review-analysis pages will work.
- The movie explorer and mood suggester depend on `src/movie_emotion_profiles.json`.
- TMDB enrichment is optional, but it is recommended if you want posters and richer movie metadata.

## License

This project is currently provided without a separate license file in the repository. Add one if you plan to distribute or reuse it publicly under specific terms.
