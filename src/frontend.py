import io
import json
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:5001/api"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIE_PROFILE_PATH = os.path.join(BASE_DIR, "movie_emotion_profiles.json")

NAV_OPTIONS = [
    "Home",
    "Train Model",
    "Single Review Analysis",
    "Movie Rating Analysis",
    "Movie Ratings Explorer",
    "Find Movies For Your Mood",
]

PAGE_META = {
    "Home": {
        "emoji": "🎞️",
        "eyebrow": "Cinematic Sentiment Lab",
        "title": "Movie analysis that feels designed, not just functional.",
        "description": "Train a review classifier, unpack emotional fingerprints, and surface recommendations with a stronger visual experience."
    },
    "Train Model": {
        "emoji": "🧠",
        "eyebrow": "Model Training",
        "title": "Build your review classifier from raw movie opinions.",
        "description": "Upload a CSV, preview the distribution, and train the sentiment engine that powers the rest of the app."
    },
    "Single Review Analysis": {
        "emoji": "🔍",
        "eyebrow": "Single Review",
        "title": "Turn one review into an emotional fingerprint.",
        "description": "See sentiment, emotional tone, reviewer style, and tailored next-watch suggestions in one place."
    },
    "Movie Rating Analysis": {
        "emoji": "🎯",
        "eyebrow": "Batch Review Scoring",
        "title": "Aggregate multiple opinions into one movie verdict.",
        "description": "Blend a stack of reviews into a rating, sentiment mix, and confidence snapshot."
    },
    "Movie Ratings Explorer": {
        "emoji": "🍿",
        "eyebrow": "Explorer",
        "title": "Browse the movie profile library like a recommendation wall.",
        "description": "Search, filter, and scan precomputed emotional profiles with posters, tone tags, and ratings."
    },
    "Find Movies For Your Mood": {
        "emoji": "✨",
        "eyebrow": "Mood Match",
        "title": "Describe the vibe and get emotionally aligned picks.",
        "description": "Blend preset moods with free text and surface movies that match how you want to feel."
    },
}

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;700;800&display=swap');

:root {
    --bg: #08111d;
    --bg-soft: #11203a;
    --panel: rgba(12, 22, 40, 0.78);
    --panel-strong: rgba(10, 18, 34, 0.92);
    --stroke: rgba(165, 205, 255, 0.18);
    --text: #f6f7fb;
    --muted: #b3bfd3;
    --gold: #ffc86f;
    --mint: #73f4c2;
    --rose: #ff7aa8;
    --sky: #6dc5ff;
    --violet: #9e8cff;
    --coral: #ff936d;
}

html, body, [class*="css"]  {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(255, 147, 109, 0.16), transparent 26%),
        radial-gradient(circle at 88% 6%, rgba(109, 197, 255, 0.20), transparent 24%),
        radial-gradient(circle at 24% 80%, rgba(158, 140, 255, 0.13), transparent 24%),
        radial-gradient(circle at 80% 82%, rgba(115, 244, 194, 0.10), transparent 20%),
        linear-gradient(180deg, #08111d 0%, #09111b 100%);
    color: var(--text);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(11, 21, 38, 0.98) 0%, rgba(15, 28, 50, 0.98) 100%);
    border-right: 1px solid var(--stroke);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.02em;
    color: var(--text);
}

p, li, label, .stMarkdown, .stCaption {
    color: var(--muted);
}

[data-testid="stMetric"] {
    background:
        linear-gradient(180deg, rgba(18, 34, 58, 0.92), rgba(10, 21, 38, 0.84));
    border: 1px solid var(--stroke);
    border-radius: 20px;
    padding: 0.8rem 0.95rem;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18);
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: linear-gradient(180deg, var(--gold), var(--rose), var(--sky));
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--text);
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid rgba(255, 200, 111, 0.34);
    background: linear-gradient(135deg, #ffc86f 0%, #ff936d 48%, #ff7aa8 100%);
    color: #07111f;
    font-weight: 800;
    padding: 0.7rem 1.25rem;
    box-shadow: 0 18px 44px rgba(255, 122, 168, 0.26);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(255, 200, 111, 0.58);
    color: #07111f;
}

.stTextArea textarea,
.stTextInput input,
.stSelectbox [data-baseweb="select"] > div,
.stFileUploader section {
    background:
        linear-gradient(180deg, rgba(16, 28, 48, 0.88), rgba(10, 21, 38, 0.82));
    border: 1px solid var(--stroke);
    border-radius: 18px;
    color: var(--text);
}

.stDataFrame, [data-testid="stTable"] {
    background: linear-gradient(180deg, rgba(15, 27, 47, 0.78), rgba(10, 18, 33, 0.82));
    border-radius: 18px;
    border: 1px solid var(--stroke);
    overflow: hidden;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6dc5ff 0%, #9e8cff 48%, #73f4c2 100%);
}

.stAlert {
    border-radius: 18px;
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(18, 32, 53, 0.92), rgba(10, 18, 33, 0.9));
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 2rem 2rem 1.8rem 2rem;
    border-radius: 28px;
    background:
        radial-gradient(circle at 14% 24%, rgba(255, 200, 111, 0.20), transparent 22%),
        radial-gradient(circle at 100% 0%, rgba(109, 197, 255, 0.22), transparent 28%),
        radial-gradient(circle at 65% 100%, rgba(158, 140, 255, 0.16), transparent 24%),
        linear-gradient(135deg, rgba(20, 36, 62, 0.95), rgba(10, 18, 34, 0.98));
    border: 1px solid var(--stroke);
    box-shadow: 0 30px 90px rgba(0, 0, 0, 0.32);
    margin-bottom: 1.3rem;
}

.hero-shell::after {
    content: "";
    position: absolute;
    inset: auto -80px -100px auto;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(115, 244, 194, 0.25) 0%, transparent 70%);
    filter: blur(10px);
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    background: linear-gradient(135deg, rgba(255, 200, 111, 0.14), rgba(109, 197, 255, 0.10));
    border: 1px solid rgba(255, 200, 111, 0.18);
    color: #ffe8b8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: clamp(2.3rem, 5vw, 4.4rem);
    line-height: 0.96;
    margin: 0 0 0.8rem 0;
    max-width: 760px;
}

.hero-copy {
    max-width: 740px;
    font-size: 1.03rem;
    line-height: 1.7;
    margin-bottom: 1.1rem;
}

.hero-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
}

.hero-chip {
    border-radius: 999px;
    padding: 0.55rem 0.95rem;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.07), rgba(158, 140, 255, 0.07));
    border: 1px solid rgba(255, 255, 255, 0.10);
    color: var(--text);
    font-size: 0.9rem;
}

.section-card {
    padding: 1.35rem 1.35rem 1.15rem 1.35rem;
    border-radius: 24px;
    background:
        linear-gradient(180deg, rgba(18, 34, 58, 0.82), rgba(10, 18, 33, 0.84));
    border: 1px solid var(--stroke);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.16);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.section-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--coral), var(--violet), var(--sky));
}

.section-title {
    font-size: 1.4rem;
    margin-bottom: 0.25rem;
}

.section-copy {
    color: var(--muted);
    margin-bottom: 0;
}

.movie-card {
    background:
        linear-gradient(180deg, rgba(16, 31, 53, 0.94), rgba(9, 17, 31, 0.97));
    border: 1px solid var(--stroke);
    border-radius: 22px;
    padding: 1rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.18);
    position: relative;
    overflow: hidden;
}

.movie-card::after {
    content: "";
    position: absolute;
    inset: auto -35px -45px auto;
    width: 120px;
    height: 120px;
    background: radial-gradient(circle, rgba(255, 200, 111, 0.14) 0%, transparent 70%);
    filter: blur(6px);
}

.movie-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
    font-size: 1.25rem;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.6rem 0 0.85rem 0;
}

.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(109, 197, 255, 0.10), rgba(255, 122, 168, 0.08));
    border: 1px solid rgba(255, 255, 255, 0.10);
    color: #e5edfa;
    padding: 0.32rem 0.72rem;
    font-size: 0.82rem;
}

.movie-copy {
    color: var(--muted);
    line-height: 1.65;
    font-size: 0.96rem;
}

.poster-placeholder {
    aspect-ratio: 2 / 3;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background:
        radial-gradient(circle at 20% 20%, rgba(255, 200, 111, 0.22), transparent 25%),
        radial-gradient(circle at 80% 10%, rgba(109, 197, 255, 0.18), transparent 20%),
        linear-gradient(160deg, rgba(21, 40, 66, 0.98), rgba(8, 17, 31, 0.98));
    border: 1px solid var(--stroke);
    color: #f3f6fb;
    font-weight: 800;
    text-align: center;
    padding: 1rem;
    min-height: 280px;
}

.sentiment-hero {
    padding: 1.1rem 1.2rem;
    border-radius: 20px;
    border: 1px solid var(--stroke);
    margin-bottom: 1rem;
}

.sentiment-positive {
    background: linear-gradient(135deg, rgba(18, 68, 58, 0.88), rgba(13, 41, 38, 0.92));
    color: #c7ffe9;
}

.sentiment-negative {
    background: linear-gradient(135deg, rgba(88, 28, 46, 0.88), rgba(48, 16, 27, 0.92));
    color: #ffd6de;
}

.sidebar-logo {
    padding: 1rem;
    border-radius: 22px;
    margin-bottom: 1rem;
    background:
        linear-gradient(135deg, rgba(255, 200, 111, 0.16), rgba(255, 122, 168, 0.08), rgba(109, 197, 255, 0.14));
    border: 1px solid var(--stroke);
}

.sidebar-logo h2 {
    margin: 0;
    font-size: 1.25rem;
}

.sidebar-logo p {
    margin: 0.45rem 0 0 0;
    font-size: 0.9rem;
}

.mini-note {
    margin-top: 0.8rem;
    color: var(--muted);
    font-size: 0.9rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 16px;
    padding: 0.42rem 0.5rem;
    margin-bottom: 0.25rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.03);
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: linear-gradient(135deg, rgba(255, 200, 111, 0.10), rgba(109, 197, 255, 0.08));
    border-color: rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] [role="radiogroup"] p {
    color: #f1f5fb;
    font-weight: 700;
}

[data-testid="stSidebar"] .stRadio > label {
    color: var(--muted);
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

@media (max-width: 900px) {
    .hero-shell {
        padding: 1.5rem 1.25rem;
    }

    .hero-title {
        font-size: 2.25rem;
    }

    .movie-card {
        padding: 0.85rem;
    }
}
</style>
"""


def inject_styles():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def check_backend():
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def fetch_backend_status():
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"is_trained": False, "movie_profiles_loaded": 0}


@st.cache_data(show_spinner=False)
def load_movie_profiles():
    if not os.path.exists(MOVIE_PROFILE_PATH):
        return []
    try:
        with open(MOVIE_PROFILE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def format_rating_stars(rating):
    full_stars = int(round(float(rating or 0)))
    full_stars = max(0, min(10, full_stars))
    return "★" * full_stars + "☆" * (10 - full_stars)


def section_intro(title, copy):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(page_name, status):
    meta = PAGE_META[page_name]
    chips = [
        f"Model {'Ready' if status.get('is_trained') else 'Needs Training'}",
        f"{status.get('movie_profiles_loaded', 0)} Movie Profiles",
        "Emotion + Sentiment + Recommendations",
    ]
    chips_markup = "".join(f'<div class="hero-chip">{chip}</div>' for chip in chips)
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-eyebrow">{meta["emoji"]} {meta["eyebrow"]}</div>
            <h1 class="hero-title">{meta["title"]}</h1>
            <p class="hero-copy">{meta["description"]}</p>
            <div class="hero-stats">{chips_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(status):
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
            <h2>FrameSense</h2>
            <p>Movie reviews, emotional fingerprints, and mood-driven discovery in one cinematic workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    app_mode = st.sidebar.radio("Navigation", NAV_OPTIONS, label_visibility="visible")
    st.sidebar.markdown(
        f"""
        <div class="mini-note">
            Model status: <strong>{'Ready' if status.get('is_trained') else 'Train first'}</strong><br>
            Profiles loaded: <strong>{status.get('movie_profiles_loaded', 0)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return app_mode


def render_movie_cards(movies, include_match=False):
    if not movies:
        st.info("No movies match the current filters.")
        return

    for start in range(0, len(movies), 3):
        columns = st.columns(3)
        for column, movie in zip(columns, movies[start:start + 3]):
            with column:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                if movie.get("poster_url") or movie.get("poster"):
                    st.image(movie.get("poster_url") or movie.get("poster"), use_container_width=True)
                else:
                    st.markdown(
                        f'<div class="poster-placeholder">{movie["title"]}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)

                release_year = movie.get("release_year") or movie.get("year") or "Year unknown"
                pills = [
                    f"📅 {release_year}",
                    f"⭐ {float(movie.get('rating', 0)):.1f}/10",
                ]

                dominant_emotion = movie.get("dominant_emotion")
                if dominant_emotion:
                    pills.append(f"🎭 {dominant_emotion.title()}")
                if include_match and movie.get("emotional_match") is not None:
                    pills.append(f"✨ {movie['emotional_match']:.1f}% match")

                pills_markup = "".join(f'<div class="meta-pill">{pill}</div>' for pill in pills)
                st.markdown(f'<div class="meta-row">{pills_markup}</div>', unsafe_allow_html=True)

                if movie.get("vote_count") is not None:
                    st.caption(f"{format_rating_stars(movie.get('rating', 0))}  •  {int(movie['vote_count'])} votes")
                else:
                    st.caption(format_rating_stars(movie.get("rating", 0)))

                st.markdown(
                    f'<div class="movie-copy">{movie.get("overview", "Overview unavailable.")}</div>',
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)


def render_recommendations(recommendations):
    for recommendation in recommendations:
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        left, right = st.columns([1, 2.2])
        with left:
            if recommendation.get("poster_url"):
                st.image(recommendation["poster_url"], use_container_width=True)
            else:
                st.markdown(
                    f'<div class="poster-placeholder">{recommendation["title"]}</div>',
                    unsafe_allow_html=True
                )
        with right:
            st.markdown(
                f'<div class="movie-title">{recommendation["title"]} ({recommendation.get("year", "Year unknown")})</div>',
                unsafe_allow_html=True
            )
            pills = [
                f"🎯 {recommendation['match_score']:.1%} fit",
                f"⭐ {float(recommendation.get('rating', 0)):.1f}/10" if recommendation.get("rating") is not None else None,
            ]
            pills_markup = "".join(
                f'<div class="meta-pill">{pill}</div>' for pill in pills if pill
            )
            st.markdown(f'<div class="meta-row">{pills_markup}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="movie-copy"><strong>Why it fits:</strong> {recommendation["reason"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="movie-copy" style="margin-top:0.65rem;">{recommendation["overview"]}</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)


def themed_pie_chart(emotions):
    emotion_df = pd.DataFrame({
        "Emotion": list(emotions.keys()),
        "Score": list(emotions.values())
    })
    fig = px.pie(
        emotion_df,
        names="Emotion",
        values="Score",
        color="Emotion",
        color_discrete_sequence=["#ffbe64", "#ff7f96", "#75c6ff", "#73f4c2", "#bca8ff", "#ff9d5c", "#5b6c87"],
        hole=0.45,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f6f7fb", family="Manrope"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_traces(textinfo="label+percent")
    return fig, emotion_df


def main():
    inject_styles()
    status = fetch_backend_status()
    app_mode = render_sidebar(status)
    render_hero(app_mode, status)

    if not check_backend():
        st.error("Backend server is not running. Start `python3 src/backend.py` first.")
        st.info("Then run the frontend with `streamlit run src/frontend.py --server.headless true --server.port 8501`.")
        return

    model_trained = status.get("is_trained", False)

    if app_mode == "Home":
        show_home(status)
    elif app_mode == "Train Model":
        show_train_model()
    elif app_mode == "Single Review Analysis":
        show_single_analysis(model_trained)
    elif app_mode == "Movie Rating Analysis":
        show_movie_analysis(model_trained)
    elif app_mode == "Movie Ratings Explorer":
        show_movie_ratings_page()
    elif app_mode == "Find Movies For Your Mood":
        show_mood_suggester()


def show_home(status):
    section_intro(
        "What This App Does",
        "This workspace blends classic sentiment analysis with emotion-aware movie discovery. It can train on your review dataset, interpret the emotional shape of a single review, and suggest films that fit the tone you want next."
    )

    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.metric("Model Status", "Ready" if status.get("is_trained") else "Needs Training")
    with stat2:
        st.metric("Movie Profiles", status.get("movie_profiles_loaded", 0))
    with stat3:
        st.metric("Core Modes", len(NAV_OPTIONS))

    left, right = st.columns([1.1, 0.9])
    with left:
        section_intro(
            "How To Use It",
            "Train the sentiment model on a CSV with `review` and `sentiment`, analyze single reviews or batches, then explore the profile library or ask for mood-based picks."
        )
        sample_data = {
            "review": [
                "This movie was absolutely fantastic with great acting!",
                "Terrible film with poor acting and a boring plot.",
                "Amazing cinematography and wonderful performances."
            ],
            "sentiment": ["positive", "negative", "positive"]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)

    with right:
        section_intro(
            "What Feels New Here",
            "The app does more than thumbs-up or thumbs-down. It maps emotions, guesses reviewer style, explores mismatches between tone and verdict, and recommends what to watch next."
        )
        st.markdown(
            """
            <div class="section-card">
                <div class="meta-row">
                    <div class="meta-pill">Train a model</div>
                    <div class="meta-pill">Decode emotions</div>
                    <div class="meta-pill">Find your mood match</div>
                </div>
                <p class="section-copy">Use the navigation on the left to move across the experience without the old dropdown flow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_train_model():
    section_intro(
        "Upload Review Data",
        "Bring in a CSV with `review` and `sentiment` columns. The app will preview the file, show the class balance, and train the model that powers the analysis screens."
    )

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully. Found {len(df)} reviews.")

        preview_col, stats_col = st.columns([1.4, 0.8])
        with preview_col:
            st.markdown("### Data Preview")
            st.dataframe(df.head(8), use_container_width=True)

        with stats_col:
            st.markdown("### Dataset Snapshot")
            if "sentiment" in df.columns:
                sentiment_counts = df["sentiment"].value_counts()
                positive = int(sentiment_counts.get("positive", 0))
                negative = int(sentiment_counts.get("negative", 0))
                total = max(len(df), 1)
                st.metric("Positive Reviews", positive)
                st.metric("Negative Reviews", negative)
                st.metric("Positive Share", f"{(positive / total) * 100:.1f}%")
            else:
                st.warning("No `sentiment` column found yet.")

        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/train", files=files, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Model trained successfully.")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{result['accuracy']:.2%}")
                        with col2:
                            st.metric("Training Samples", result["training_size"])
                        with col3:
                            st.metric("Test Samples", result["test_size"])
                    else:
                        st.error(response.json().get("error", "Unknown error"))
                except Exception as exc:
                    st.error(f"Error during training: {exc}")

    except Exception as exc:
        st.error(f"Error reading file: {exc}")


def show_single_analysis(model_trained):
    section_intro(
        "Analyze One Review",
        "Paste a single review to inspect the predicted sentiment, emotional fingerprint, reviewer style, and the new recommendation engine."
    )

    if not model_trained:
        st.warning("Please train the model first in the 'Train Model' section.")
        return

    review_text = st.text_area(
        "Enter movie review",
        placeholder="Tell the app what worked, what failed, and how the movie felt overall...",
        height=160
    )

    if not (st.button("Analyze Review", type="primary") and review_text.strip()):
        return

    with st.spinner("Analyzing review..."):
        try:
            response = requests.post(f"{API_URL}/predict", json={"review": review_text}, timeout=90)
            if response.status_code != 200:
                st.error(response.json().get("error", "Unknown error"))
                return

            result = response.json()
            sentiment = result["sentiment"]
            confidence = result["confidence"]

            sentiment_class = "sentiment-positive" if sentiment == "positive" else "sentiment-negative"
            sentiment_icon = "Positive Verdict" if sentiment == "positive" else "Negative Verdict"
            st.markdown(
                f"""
                <div class="sentiment-hero {sentiment_class}">
                    <h3 style="margin:0;">{sentiment_icon}</h3>
                    <p style="margin:0.45rem 0 0 0;">Confidence: {confidence:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Positive Probability", f"{result['probabilities']['positive']:.2%}")
            with summary_col2:
                st.metric("Negative Probability", f"{result['probabilities']['negative']:.2%}")
            with summary_col3:
                top_emotion = max(result.get("emotions", {}), key=result.get("emotions", {}).get) if result.get("emotions") else "unknown"
                st.metric("Dominant Emotion", top_emotion.title())

            st.markdown("### Review Snapshot")
            st.markdown(
                f'<div class="section-card"><p class="section-copy" style="color:#f6f7fb;">{review_text}</p></div>',
                unsafe_allow_html=True
            )

            if "emotions" in result:
                section_intro("Emotional Fingerprint", "This shows the emotional mix detected in the review rather than just the binary sentiment label.")
                chart_col, table_col = st.columns([1.15, 0.85])
                fig, emotion_df = themed_pie_chart(result["emotions"])
                with chart_col:
                    st.plotly_chart(fig, use_container_width=True)
                with table_col:
                    emotion_percent_df = emotion_df.copy()
                    emotion_percent_df["Percentage"] = (emotion_percent_df["Score"] * 100).round(2).astype(str) + "%"
                    st.dataframe(
                        emotion_percent_df[["Emotion", "Percentage"]],
                        use_container_width=True,
                        hide_index=True
                    )

            if "reviewer_style" in result:
                section_intro("Reviewer Style", "A lightweight heuristic that estimates whether the writing reads more like a critic, a casual audience reaction, or something in between.")
                style = result["reviewer_style"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Style", style["style"].title())
                with col2:
                    st.metric("Tone", style["tone"].title())
                with col3:
                    st.metric("Focus", style["focus"].title())
                st.info(style["explanation"])

            if "mismatch_insight" in result:
                section_intro("Sentiment vs Emotion", "Sometimes a review is positive overall but emotionally heavy, or negative while still carrying warmth. This block surfaces that nuance.")
                mismatch = result["mismatch_insight"]
                if mismatch["mismatch"]:
                    st.warning(f"{mismatch['type'].title()}: {mismatch['summary']}")
                else:
                    st.success(mismatch["summary"])
                st.write(mismatch["detail"])

            if "recommendations" in result:
                section_intro("What To Watch Next", "These picks now pull from the larger movie profile library and vary between runs while staying close to what your review seems to value.")
                render_recommendations(result["recommendations"])

        except Exception as exc:
            st.error(f"Error during analysis: {exc}")


def show_movie_analysis(model_trained):
    section_intro(
        "Rate A Movie From Multiple Reviews",
        "Drop in a stack of opinions, one per line, and the app will estimate an overall rating, sentiment split, and confidence level."
    )

    if not model_trained:
        st.warning("Please train the model first in the 'Train Model' section.")
        return

    reviews = st.text_area(
        "Enter reviews, one per line",
        placeholder="This movie was gripping from start to finish.\nAmazing performances and a really sharp ending.\nVisually gorgeous but a little slow in the middle.",
        height=220
    )

    if not (st.button("Calculate Movie Rating", type="primary") and reviews.strip()):
        return

    review_list = [review.strip() for review in reviews.split("\n") if review.strip()]
    if not review_list:
        st.warning("Please enter at least one review.")
        return

    with st.spinner("Analyzing reviews..."):
        try:
            response = requests.post(f"{API_URL}/analyze-movie", json={"reviews": review_list}, timeout=90)
            if response.status_code != 200:
                st.error(response.json().get("error", "Unknown error"))
                return

            result = response.json()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Rating", f"{result['movie_rating']}/10")
            with col2:
                st.metric("Positive Reviews", result["positive_reviews"])
            with col3:
                st.metric("Negative Reviews", result["negative_reviews"])
            with col4:
                st.metric("Average Confidence", f"{result['average_confidence']:.2%}")

            section_intro("Sentiment Split", "A quick visual check of how the review set leans overall.")
            split_col1, split_col2 = st.columns([1.1, 0.9])
            with split_col1:
                if result["positive_reviews"] > 0:
                    st.progress(result["positive_percentage"] / 100, text=f"Positive: {result['positive_percentage']:.1f}%")
                if result["negative_reviews"] > 0:
                    st.progress(result["negative_percentage"] / 100, text=f"Negative: {result['negative_percentage']:.1f}%")
            with split_col2:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <div class="meta-row">
                            <div class="meta-pill">Total Reviews: {result['total_reviews']}</div>
                            <div class="meta-pill">Rating: {result['movie_rating']}/10</div>
                        </div>
                        <p class="section-copy">The overall score is a direct reflection of how many reviews landed positive versus negative.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            section_intro("Individual Review Breakdowns", "Open each review to see its individual sentiment prediction and confidence score.")
            for index, (review, prediction) in enumerate(zip(review_list, result["individual_predictions"]), start=1):
                with st.expander(f"Review {index}: {review[:70]}...", expanded=False):
                    st.write(f"Sentiment: **{prediction['sentiment'].title()}**")
                    st.write(f"Confidence: **{prediction['confidence']:.2%}**")
                    st.write(review)

        except Exception as exc:
            st.error(f"Error during analysis: {exc}")


def show_movie_ratings_page():
    section_intro(
        "Movie Profile Library",
        "Browse the precomputed catalog by title and dominant emotional tone. This page is fully powered by the JSON profile file."
    )

    movies = load_movie_profiles()
    if not movies:
        st.warning("No movie profile data found yet. Generate movie_emotion_profiles.json first.")
        return

    filter_col1, filter_col2 = st.columns([1.2, 0.8])
    with filter_col1:
        search_query = st.text_input("Search movie title")
    with filter_col2:
        emotion_options = ["All"] + sorted({movie.get("dominant_emotion", "unknown").title() for movie in movies})
        selected_emotion = st.selectbox("Filter by dominant emotion", emotion_options)

    filtered_movies = []
    for movie in movies:
        title_matches = search_query.lower() in movie.get("title", "").lower()
        emotion_matches = selected_emotion == "All" or movie.get("dominant_emotion", "").title() == selected_emotion
        if title_matches and emotion_matches:
            filtered_movies.append(movie)

    filtered_movies = sorted(
        filtered_movies,
        key=lambda item: (float(item.get("rating", 0)), int(item.get("vote_count", 0))),
        reverse=True
    )

    st.caption(f"Showing {len(filtered_movies)} movie(s)")
    render_movie_cards(filtered_movies)


def show_mood_suggester():
    section_intro(
        "Mood-Based Suggestions",
        "Choose a preset feeling, add your own words, or combine both. The backend will find movies that emotionally line up with that mood."
    )

    input_col1, input_col2 = st.columns([0.9, 1.1])
    with input_col1:
        preset_mood = st.selectbox(
            "Preset mood",
            ["", "happy", "sad", "tense", "angry", "excited", "nostalgic", "adventurous", "relaxed"],
            format_func=lambda value: "Choose a preset mood" if value == "" else value.title()
        )
    with input_col2:
        free_text = st.text_area(
            "Describe how you're feeling",
            placeholder="I want something emotional, a little hopeful, and not too heavy.",
            height=140
        )

    if not st.button("Suggest Movies", type="primary"):
        return

    if not preset_mood and not free_text.strip():
        st.warning("Please choose a preset mood, add a free-text description, or both.")
        return

    with st.spinner("Finding movies for your mood..."):
        try:
            response = requests.post(
                f"{API_URL}/suggest",
                json={"preset_mood": preset_mood, "free_text": free_text},
                timeout=60
            )
            payload = response.json()
            if response.status_code != 200:
                st.error(payload.get("error", "Unknown error"))
                return

            results = payload.get("results", [])
            if not results:
                st.info("No movie suggestions were available for the current filters.")
                return

            section_intro("Your Mood Matches", "These suggestions are chosen from the current profile library and shuffled from the strongest candidate pool.")
            render_movie_cards(results, include_match=True)
        except Exception as exc:
            st.error(f"Error while fetching suggestions: {exc}")


if __name__ == "__main__":
    main()
