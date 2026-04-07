import argparse
import json
import os
from datetime import datetime

import pandas as pd
import requests
from transformers import pipeline

EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "movie_emotion_profiles.json")
DEFAULT_MOVIES_CSV = os.path.join(os.path.dirname(BASE_DIR), "archive", "rotten_tomatoes_movies.csv")
DEFAULT_REVIEWS_CSV = os.path.join(os.path.dirname(BASE_DIR), "archive", "rotten_tomatoes_critic_reviews.csv")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def normalize_emotions(emotions):
    scores = {label: float(emotions.get(label, 0.0)) for label in EMOTION_LABELS}
    total = sum(scores.values())
    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    return scores


def get_review_emotions(classifier, text):
    results = classifier(str(text)[:512], top_k=None, truncation=True)
    if results and isinstance(results[0], dict):
        raw_scores = results
    else:
        raw_scores = results[0]
    return normalize_emotions({item["label"]: item["score"] for item in raw_scores})


def batched_review_emotions(classifier, reviews, batch_size=16):
    normalized_results = []
    trimmed_reviews = [str(review)[:512] for review in reviews]
    for start in range(0, len(trimmed_reviews), batch_size):
        batch = trimmed_reviews[start:start + batch_size]
        batch_results = classifier(batch, top_k=None, truncation=True, batch_size=batch_size)
        for result in batch_results:
            if result and isinstance(result[0], dict):
                raw_scores = result
            else:
                raw_scores = result[0]
            normalized_results.append(
                normalize_emotions({item["label"]: item["score"] for item in raw_scores})
            )
    return normalized_results


def parse_release_year(value):
    if pd.isna(value) or not value:
        return None
    value = str(value).strip()
    try:
        return datetime.fromisoformat(value).year
    except ValueError:
        return None


def fetch_tmdb_metadata(title, release_year, api_key):
    if not api_key:
        return {}

    params = {"api_key": api_key, "query": title}
    if release_year:
        params["year"] = int(release_year)

    response = requests.get(TMDB_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return {}

    match = results[0]
    release_date = match.get("release_date") or ""
    matched_year = release_year or (int(release_date[:4]) if len(release_date) >= 4 else None)
    return {
        "poster_url": f"{TMDB_IMAGE_BASE}{match['poster_path']}" if match.get("poster_path") else None,
        "rating": round(float(match.get("vote_average") or 0.0), 1),
        "vote_count": int(match.get("vote_count") or 0),
        "overview": match.get("overview") or "",
        "release_year": matched_year
    }


def load_existing_profiles(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {profile.get("title"): profile for profile in profiles if profile.get("title")}


def select_balanced_profiles(profiles, target_per_emotion=None):
    if not target_per_emotion:
        return profiles

    selected_titles = set()
    selected_profiles = []

    for emotion in EMOTION_LABELS:
        ranked = sorted(
            profiles,
            key=lambda item: (
                float(item.get("emotion_profile", {}).get(emotion, 0.0)),
                float(item.get("rating", 0.0)),
                int(item.get("vote_count", 0)),
                int(item.get("review_count", 0))
            ),
            reverse=True
        )

        picked_for_emotion = 0
        for profile in ranked:
            title = profile.get("title")
            if not title or title in selected_titles:
                continue
            selected_titles.add(title)
            selected_profiles.append(profile)
            picked_for_emotion += 1
            if picked_for_emotion >= target_per_emotion:
                break

    return sorted(
        selected_profiles,
        key=lambda item: (float(item.get("rating", 0)), int(item.get("vote_count", 0)), item["review_count"]),
        reverse=True
    )


def prepare_dataset(movies_csv, reviews_csv):
    movies = pd.read_csv(movies_csv)
    reviews = pd.read_csv(reviews_csv)

    movie_columns = [
        "rotten_tomatoes_link",
        "movie_title",
        "movie_info",
        "original_release_date",
        "tomatometer_rating",
        "tomatometer_count",
        "audience_rating",
        "audience_count"
    ]
    reviews = reviews[["rotten_tomatoes_link", "review_content"]]
    merged = reviews.merge(movies[movie_columns], on="rotten_tomatoes_link", how="inner")
    merged = merged.dropna(subset=["movie_title", "review_content"])
    merged["movie_title"] = merged["movie_title"].astype(str).str.strip()
    merged["review_content"] = merged["review_content"].astype(str).str.strip()
    merged = merged[(merged["movie_title"] != "") & (merged["review_content"] != "")]
    return merged


def build_profiles(
    movies_csv,
    reviews_csv,
    output_path,
    tmdb_api_key=None,
    max_movies=None,
    max_reviews_per_movie=80,
    target_per_emotion=None
):
    existing_profiles = load_existing_profiles(output_path)
    merged = prepare_dataset(movies_csv, reviews_csv)
    grouped = (
        merged.groupby("movie_title")
        .agg(
            review_content=("review_content", list),
            movie_info=("movie_info", "first"),
            original_release_date=("original_release_date", "first"),
            tomatometer_rating=("tomatometer_rating", "first"),
            tomatometer_count=("tomatometer_count", "first"),
            audience_rating=("audience_rating", "first"),
            audience_count=("audience_count", "first")
        )
        .reset_index()
    )
    grouped["review_count"] = grouped["review_content"].apply(len)
    grouped = grouped[grouped["review_count"] >= 5]
    grouped = grouped.sort_values(["review_count", "tomatometer_count"], ascending=False)
    if max_movies:
        top_grouped = grouped.head(max_movies)
        keep_titles = set(existing_profiles.keys())
        extra_grouped = grouped[grouped["movie_title"].isin(keep_titles)]
        grouped = (
            pd.concat([top_grouped, extra_grouped], ignore_index=True)
            .drop_duplicates(subset=["movie_title"])
            .reset_index(drop=True)
        )

    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    profiles = []

    for _, row in grouped.iterrows():
        emotion_totals = {label: 0.0 for label in EMOTION_LABELS}
        reviews = row["review_content"][:max_reviews_per_movie]

        for scores in batched_review_emotions(classifier, reviews):
            for label in EMOTION_LABELS:
                emotion_totals[label] += scores[label]

        review_count = len(reviews)
        average_scores = {label: emotion_totals[label] / review_count for label in EMOTION_LABELS}
        normalized_scores = normalize_emotions(average_scores)
        dominant_emotion = max(normalized_scores, key=normalized_scores.get)
        release_year = parse_release_year(row["original_release_date"])
        tmdb_metadata = fetch_tmdb_metadata(row["movie_title"], release_year, tmdb_api_key)
        cached_profile = existing_profiles.get(row["movie_title"], {})

        fallback_rating = float(row["tomatometer_rating"]) / 10 if pd.notna(row["tomatometer_rating"]) else 0.0
        fallback_vote_count = int(row["tomatometer_count"]) if pd.notna(row["tomatometer_count"]) else 0

        profiles.append({
            "title": row["movie_title"],
            "release_year": tmdb_metadata.get("release_year") or cached_profile.get("release_year") or release_year,
            "poster_url": tmdb_metadata.get("poster_url") or cached_profile.get("poster_url"),
            "rating": tmdb_metadata.get("rating", cached_profile.get("rating", round(fallback_rating, 1))),
            "vote_count": tmdb_metadata.get("vote_count", cached_profile.get("vote_count", fallback_vote_count)),
            "overview": tmdb_metadata.get("overview") or cached_profile.get("overview") or (row["movie_info"] if pd.notna(row["movie_info"]) else ""),
            "dominant_emotion": dominant_emotion,
            "emotion_profile": {label: round(normalized_scores[label], 4) for label in EMOTION_LABELS},
            "review_count": review_count,
            "source_review_count": len(row["review_content"]),
            "tomatometer_rating": int(row["tomatometer_rating"]) if pd.notna(row["tomatometer_rating"]) else None,
            "audience_rating": int(row["audience_rating"]) if pd.notna(row["audience_rating"]) else None
        })

    profiles = sorted(
        profiles,
        key=lambda item: (float(item.get("rating", 0)), int(item.get("vote_count", 0)), item["review_count"]),
        reverse=True
    )
    profiles = select_balanced_profiles(profiles, target_per_emotion=target_per_emotion)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Saved {len(profiles)} movie profiles to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build movie emotion profiles from the Rotten Tomatoes movies and critic reviews CSVs.")
    parser.add_argument("--movies-csv", default=DEFAULT_MOVIES_CSV, help="Path to rotten_tomatoes_movies.csv")
    parser.add_argument("--reviews-csv", default=DEFAULT_REVIEWS_CSV, help="Path to rotten_tomatoes_critic_reviews.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the output JSON file")
    parser.add_argument("--tmdb-api-key", default=os.getenv("TMDB_API_KEY"), help="Optional TMDB API key for posters and TMDB ratings")
    parser.add_argument("--max-movies", type=int, default=None, help="Optional cap for faster local testing")
    parser.add_argument("--max-reviews-per-movie", type=int, default=80, help="Limit reviews per movie for faster profile generation")
    parser.add_argument("--target-per-emotion", type=int, default=None, help="Keep up to this many unique movies from the top of each emotion ranking")
    args = parser.parse_args()

    build_profiles(
        movies_csv=args.movies_csv,
        reviews_csv=args.reviews_csv,
        output_path=args.output,
        tmdb_api_key=args.tmdb_api_key,
        max_movies=args.max_movies,
        max_reviews_per_movie=args.max_reviews_per_movie,
        target_per_emotion=args.target_per_emotion
    )


if __name__ == "__main__":
    main()
