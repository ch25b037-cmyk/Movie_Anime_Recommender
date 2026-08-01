"""
STOOBID SHAASTRIANS
Hybrid AI Recommendation System — Streamlit Frontend

Combines:
- Premium Netflix-style Glassmorphism UI (Frosted glass cards & sidebar)
- Cinematic intro animation + typography alignment
- Real recommender.py integration (master_movie/master_anime, embeddings)
- Exponential taste-decay embedding updates
- Per-mode disk persistence (profile survives refresh/restart)
- Poster images (TMDB CDN) on search, recommendations, and category rows
- Flip-card rating UI (front = poster, back = star picker)
- Clean, continuous, sideways-scrollable genre shelves (Netflix layout)

This file only imports from recommender.py. recommender.py is NOT modified.
"""

import os
import numpy as np
import streamlit as st
from sklearn.preprocessing import normalize

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="STOOBID SHAASTRIANS",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

# =========================================================
# CUSTOM GLASSMORPHISM CSS
# =========================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

/* Keep header visible (it holds the sidebar toggle), but make it transparent */
header {
    background: transparent !important;
}
[data-testid="stToolbar"] {
    right: 2rem;
}

/* Global type system:
   Montserrat = display face (brand title, section headings) — bold, condensed-feeling, cinematic
   Inter = body/UI face (everything you read/click) — clean, high-legibility on translucent glass */

   
html, body, .stApp, [data-testid="stAppViewContainer"],
.stMarkdown, .stSelectbox, .stTextInput, [data-testid="stWidgetLabel"],
[data-testid="stAlert"], [data-testid="stToastContainer"],
[data-baseweb="popover"], [data-baseweb="select"], .stCaption {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

/* Vivid Multi-Blob Cinema Background — gives backdrop-filter blur
   something colorful to actually diffuse. Flat/near-black backgrounds
   make glassmorphism invisible no matter how strong the blur is. */


html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0d !important;
    background-attachment: fixed !important;
    color: #f5f5f7;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Frosted Glass Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(18, 16, 20, 0.45) !important;
    backdrop-filter: blur(30px) saturate(200%) brightness(1.05) !important;
    -webkit-backdrop-filter: blur(30px) saturate(200%) brightness(1.05) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
    box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.04),
                4px 0 30px rgba(0, 0, 0, 0.35) !important;
}
[data-testid="stSidebar"] > div:first-child {
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* ---- Make the sidebar collapse/expand arrow impossible to miss ---- */
[data-testid="collapsedControl"] {
    background-color: #E50914 !important;
    border-radius: 8px !important;
    padding: 6px !important;
    box-shadow: 0 0 15px rgba(229, 9, 20, 0.8) !important;
    z-index: 999999 !important;
}
[data-testid="collapsedControl"] svg { fill: #ffffff !important; }
[data-testid="collapsedControl"]:hover { background-color: #ff1a25 !important; }
[data-testid="stSidebarCollapseButton"] button {
    background-color: #E50914 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebarCollapseButton"] button svg { fill: #ffffff !important; }

.brand-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    text-transform: uppercase;
    color: #ff0a16;
    text-shadow: 0 0 22px rgba(229, 9, 20, 0.55), 0 0 46px rgba(229, 9, 20, 0.22);
    margin-bottom: 0;
    line-height: 1.05;
}
.brand-title-card {
    font-family: 'Montserrat';
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    text-transform: uppercase;
    color: #ff0a16;
    text-shadow: 0 0 22px rgba(229, 9, 20, 0.55), 0 0 46px rgba(229, 9, 20, 0.22);
    margin-bottom: 0;
    line-height: 1.05;
}
.brand-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9c9c9c;
    margin-top: 2px;
    margin-bottom: 1.2rem;
}

/* Frosted Glass Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.10), rgba(30, 30, 35, 0.35)) !important;
    color: #e6e6e6 !important;
    font-weight: 600;
    letter-spacing: 0.2px;
    backdrop-filter: blur(14px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.stButton > button:hover {
    border-color: #E50914 !important;
    color: #ffffff !important;
    background-color: #241414 !important;
    box-shadow: 0 0 12px rgba(229, 9, 20, 0.4) !important;
    transform: translateY(-1px);
}
button[kind="secondary"] {
    background-color: #E50914 !important;
    border-color: #E50914 !important;
    color: #ffffff !important;
}
button[kind="secondary"]:hover { background-color: #ff1a25 !important; }

.section-heading {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 1.8rem;
    margin-bottom: 0.8rem;
    color: #f5f5f7;
    border-left: 4px solid #E50914;
    padding-left: 10px;
}

/* Translucent Glass Inputs */
.stTextInput > div > div > input {
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.08), rgba(30, 30, 35, 0.35)) !important;
    color: #ffffff !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 1.05rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
    transition: all 0.3s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #E50914 !important;
    box-shadow: 0 0 12px rgba(229, 9, 20, 0.4) !important;
}

/* Netflix Frosted Glass Card */
.poster-card {
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.10) 0%, rgba(25, 25, 30, 0.32) 18%) !important;
    backdrop-filter: blur(18px) saturate(190%) brightness(1.08) !important;
    -webkit-backdrop-filter: blur(18px) saturate(190%) brightness(1.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 0.6rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18),
                inset 0 -1px 0 rgba(0, 0, 0, 0.25),
                0 8px 32px 0 rgba(0, 0, 0, 0.45) !important;
    animation: fadeIn 0.4s ease-in-out;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.poster-card:hover {
    transform: translateY(-4px) scale(1.03);
    border-color: rgba(229, 9, 20, 0.8) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.22),
                0 12px 24px rgba(229, 9, 20, 0.3), 0 0 20px rgba(229, 9, 20, 0.2) !important;
}
.poster-img-wrap {
    width: 100%;
    aspect-ratio: 2 / 3;
    background-color: rgba(20, 20, 25, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.poster-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.poster-placeholder { font-size: 2.4rem; color: #333; }
.poster-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: -0.1px;
    color: #f0f0f0;
    padding: 0.5rem 0.6rem 0.1rem 0.6rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.poster-score {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #ff4d55;
    padding: 0 0.6rem 0.5rem 0.6rem;
    font-weight: 700;
}

/* Frosted Crimson Glass Flipped Card */
.rate-back {
    background: linear-gradient(160deg, rgba(255, 80, 80, 0.16) 0%, rgba(229, 9, 20, 0.12) 100%) !important;
    backdrop-filter: blur(22px) saturate(220%) brightness(1.1) !important;
    -webkit-backdrop-filter: blur(22px) saturate(220%) brightness(1.1) !important;
    border: 1px solid rgba(255, 120, 120, 0.55) !important;
    border-radius: 14px;
    padding: 0.8rem;
    margin-bottom: 0.6rem;
    text-align: center;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 8px 32px 0 rgba(229, 9, 20, 0.25) !important;
    animation: fadeIn 0.35s cubic-bezier(0.25, 1, 0.5, 1);
}
.rate-back .label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #ff8b8f;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #8a8a8a;
    border: 1px dashed rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(14px) saturate(160%);
    -webkit-backdrop-filter: blur(14px) saturate(160%);
    border-radius: 14px;
    margin-top: 1.5rem;
}
.history-item {
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.06), rgba(20, 20, 25, 0.35));
    border: 1px solid rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(10px) saturate(160%);
    -webkit-backdrop-filter: blur(10px) saturate(160%);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.45rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #d8d8d8;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}
hr { border-color: rgba(255, 255, 255, 0.06); }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* =========================================================
   FROSTED GLASS HORIZONTAL SCROLL SHELF ENGINE
   ========================================================= */
div[data-testid="stHorizontalBlock"]:has(div.scroll-anchor) {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    padding: 15px 12px !important;
    gap: 15px !important;
    scroll-behavior: smooth;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06),
                inset 0 0 30px rgba(0, 0, 0, 0.25);
}
div[data-testid="stHorizontalBlock"]:has(div.scroll-anchor) > div {
    min-width: 180px !important; /* Locks width for uniform scrolling */
    flex: 0 0 auto !important;
}
div[data-testid="stHorizontalBlock"]:has(div.scroll-anchor)::-webkit-scrollbar {
    height: 6px;
}
div[data-testid="stHorizontalBlock"]:has(div.scroll-anchor)::-webkit-scrollbar-track {
    background: rgba(20, 20, 20, 0.5);
    border-radius: 4px;
}
div[data-testid="stHorizontalBlock"]:has(div.scroll-anchor)::-webkit-scrollbar-thumb {
    background: rgba(229, 9, 20, 0.75);
    border-radius: 4px;
}
div[data-testid="stHorizontalBlock"]:has(div.scroll-anchor)::-webkit-scrollbar-thumb:hover {
    background: #ff1a25;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# INTRO ANIMATION (plays once per session)
# =========================================================

INTRO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap');

/* Keyframes */
@keyframes sLetterPop {
    0% {
        opacity: 0;
        transform: scale(0.4) rotate(-8deg);
        filter: blur(12px);}
    50% {
        opacity: 1;
        filter: blur(0);}
    100% {
        opacity: 1;
        transform: scale(1) rotate(0deg);
    }
}

@keyframes revealRestText {
    0% {
        max-width: 0;
        opacity: 0;
        filter: blur(12px);
    }
    100% {
        max-width: 1200px; /* Expanded from 600px to fully display SHAASTRIANS */
        opacity: 1;
        filter: blur(0);
    }
}

@keyframes cinematicZoom {
    0% {
        transform: scale(0.85);
    }
    100% {
        transform: scale(1.05);
    }
}

@keyframes introFadeOut {
    0% {
        opacity: 1;
        visibility: visible;
    }
    99% {
        opacity: 0;
        visibility: visible;
    }
    100% {
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
    }
}

#intro-overlay {
    position: fixed;
    inset: 0;
    background-color: #050505;
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    animation: introFadeOut 0.8s cubic-bezier(0.77, 0, 0.175, 1) forwards;
    animation-delay: 3.4s;
}

#intro-container {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Montserrat', "Helvetica Neue", Arial, sans-serif;
    font-weight: 900;
    color: #E50914;
    text-shadow: 0 0 25px rgba(229, 9, 20, 0.85), 0 0 50px rgba(229, 9, 20, 0.4);
    letter-spacing: 2px;
    transform: scale(0.9);
    animation: cinematicZoom 4.2s cubic-bezier(0.1, 0.8, 0.1, 1) forwards;
}

.word {
    display: flex;
    align-items: center;
    white-space: nowrap;
}

.word-1 {
    margin-right: 1.5rem; /* Gap between S S initially */
}

.word-2 {
    /* Kinetic offset target margin spacing */
}

.letter-s {
    font-size: 8rem;
    display: inline-block;
    animation: sLetterPop 1.2s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}

.rest-text {
    font-size: 5.5rem; /* Hierarchy adjustment */
    display: inline-block;
    max-width: 0;
    opacity: 0;
    filter: blur(12px);
    overflow: hidden;
    padding-right: 15px; /* Buffer to prevent trailing text/glow from clipping */
    animation: revealRestText 1.4s cubic-bezier(0.25, 1, 0.5, 1) forwards;
    animation-delay: 1.3s;
    vertical-align: middle;
}

/* Base adjustment to align letter baselines nicely */
.rest-1 {
    padding-top: 0.8rem;
}
.rest-2 {
    padding-top: 0.8rem;
}

</style>

<div id="intro-overlay">
    <div id="intro-container">
        <div class="word word-1">
            <span class="letter-s">S</span>
            <span class="rest-text rest-1">TOOBID</span>
        </div>
        <div class="word word-2">
            <span class="letter-s">S</span>
            <span class="rest-text rest-2">HAASTRIANS</span>
        </div>
    </div>
</div>
"""
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = True
    st.markdown(INTRO_HTML, unsafe_allow_html=True)

# =========================================================
# LOAD RECOMMENDER 
# =========================================================

@st.cache_resource
def load_recommender_modules():
    from recommender import (
        master_movie, master_anime,
        movie_embeddings, anime_embeddings,
        movie_title_to_row, anime_title_to_row,
        search_movies, search_anime,
        recommend_movies_from_embedding, recommend_anime_from_embedding,
        get_movie_user_embedding, get_anime_user_embedding,
    )
    return {
        "master_movie": master_movie,
        "master_anime": master_anime,
        "movie_embeddings": movie_embeddings,
        "anime_embeddings": anime_embeddings,
        "movie_title_to_row": movie_title_to_row,
        "anime_title_to_row": anime_title_to_row,
        "search_movies": search_movies,
        "search_anime": search_anime,
        "recommend_movies_from_embedding": recommend_movies_from_embedding,
        "recommend_anime_from_embedding": recommend_anime_from_embedding,
        "get_movie_user_embedding": get_movie_user_embedding,
        "get_anime_user_embedding": get_anime_user_embedding,
    }

modules = load_recommender_modules()

# =========================================================
# CACHED GENRE SHELVES EXTRACTOR (Onboarding Shelves)
# =========================================================


def _parse_genre_list(raw):
    """
    Parses a genre cell that may be:
      - a real Python list (already parsed by pandas/pickle)
      - a JSON-style stringified list, e.g. '["Action", "Award Winning", "Sci-Fi"]'
      - a Python-repr stringified list, e.g. "['Action', 'Sci-Fi']"
      - a plain delimited string, e.g. "Action|Comedy" or "Action, Comedy"
    and always returns a clean list[str].
    """
    if isinstance(raw, list):
        return [str(g).strip() for g in raw if str(g).strip()]
    if not isinstance(raw, str) or not raw.strip():
        return []
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        import json, ast
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, list):
                    return [str(g).strip() for g in parsed if str(g).strip()]
            except (ValueError, SyntaxError, TypeError):
                continue
    # Fallback: plain delimited string
    return [g.strip() for g in raw.replace("|", ",").split(",") if g.strip()]


@st.cache_data
def get_top_by_genre(mode, n=10):
    """
    Extracts the top n most popular titles for a set of major genres 
    to populate horizontal onboarding discovery shelves.
    """
    if mode == "anime":
        df = modules["master_anime"].copy()
        # Compute popular anime based on score and member size
        df["pop_score"] = df["score"] * np.log1p(df["members"])
        df["genre_list"] = df["genres"].apply(_parse_genre_list)
        # MAL "genres" taxonomy (as opposed to themes/demographics like
        # Isekai, Shounen, Mecha, which live in separate fields and won't
        # match here even if present in the source data)
        major_genres = [
            "Action", "Adventure", "Award Winning", "Comedy", "Drama",
            "Fantasy", "Romance", "Sci-Fi", "Slice of Life", "Supernatural",
        ]
    else:
        df = modules["master_movie"].copy()
        df["pop_score"] = df["popularity"]
        df["genre_list"] = df["genres"].apply(_parse_genre_list)
        major_genres = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Thriller", "Romance"]

    genre_map = {}
    for g in major_genres:
        # Filter titles belonging to the genre and sort by popularity
        filtered = df[df["genre_list"].apply(lambda list_g: g in list_g)]
        top_items = filtered.sort_values(by="pop_score", ascending=False).head(n)
        if not top_items.empty:
            genre_map[g] = top_items["title"].tolist()
            
    return genre_map

# =========================================================
# POSTER LOOKUP
# =========================================================

def get_poster_url(title, mode):
    """Looks up poster_path for `title` in the appropriate master table
    and returns a full TMDB CDN url, or None if unavailable."""
    master_df = modules["master_movie"] if mode == "movie" else modules["master_anime"]
    title_to_row = modules["movie_title_to_row"] if mode == "movie" else modules["anime_title_to_row"]

    row_idx = title_to_row.get(title)
    if row_idx is None:
        return None
    try:
        poster_path = master_df.iloc[row_idx]["poster_path"]
    except Exception:
        return None
    if not poster_path or not isinstance(poster_path, str):
        return None
    return TMDB_IMAGE_BASE + poster_path


def render_poster_block(title, mode, score=None):
    """Renders the poster image + title (+ optional score) as one HTML block."""
    poster_url = get_poster_url(title, mode)
    if poster_url:
        img_html = f'<img src="{poster_url}" alt="{title}" />'
    else:
        img_html = '<div class="poster-placeholder">🎬</div>'

    score_html = ""
    if score is not None:
        score_html = f'<div class="poster-score">Similarity {int(round(score * 100))}%</div>'

    st.markdown(
        f"""
        <div class="poster-card">
            <div class="poster-img-wrap">{img_html}</div>
            <div class="poster-title">{title}</div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# EMBEDDING UPDATE (exponential taste decay)
# =========================================================

RATING_WEIGHTS = {1: -1.5, 2: -0.75, 3: 0.0, 4: 0.75, 5: 1.5, "skip": 0.0}


def update_embedding_exponential(user_embedding, item_title, rating, item_embeddings, item_title_to_row, beta=0.30):
    if item_title not in item_title_to_row:
        return user_embedding
    emb = item_embeddings[item_title_to_row[item_title]].reshape(1, -1)
    weight = RATING_WEIGHTS.get(rating, 0.0)
    updated = (1.0 - beta) * user_embedding + beta * (weight * emb)
    return normalize(updated, norm="l2").astype(np.float32)

# =========================================================
# PER-MODE DISK PERSISTENCE
# =========================================================

def file_paths_for(mode):
    if mode == "anime":
        return "user_taste_anime.npy", "watch_history_anime.txt"
    return "user_taste_movie.npy", "watch_history_movie.txt"


def load_mode_state(mode):
    profile_file, history_file = file_paths_for(mode)
    get_emb_fn = modules["get_anime_user_embedding"] if mode == "anime" else modules["get_movie_user_embedding"]

    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            watched = [line.strip() for line in f.readlines() if line.strip()]
    else:
        watched = []

    if os.path.exists(profile_file):
        user_emb = np.load(profile_file)
    elif watched:
        user_emb = get_emb_fn(watched)
    else:
        user_emb = None

    st.session_state.watched_items = watched
    st.session_state.user_emb = user_emb


def process_rating(title, rating):
    mode = st.session_state.mode
    embeddings = modules["anime_embeddings"] if mode == "anime" else modules["movie_embeddings"]
    title_to_row = modules["anime_title_to_row"] if mode == "anime" else modules["movie_title_to_row"]
    get_emb_fn = modules["get_anime_user_embedding"] if mode == "anime" else modules["get_movie_user_embedding"]
    profile_file, history_file = file_paths_for(mode)

    if st.session_state.user_emb is None:
        user_emb = get_emb_fn([title])
    else:
        user_emb = update_embedding_exponential(
            st.session_state.user_emb, title, rating, embeddings, title_to_row, beta=0.30
        )

    if title not in st.session_state.watched_items:
        st.session_state.watched_items.append(title)

    st.session_state.user_emb = user_emb
    np.save(profile_file, user_emb)
    with open(history_file, "w", encoding="utf-8") as f:
        for m in st.session_state.watched_items:
            f.write(m + "\n")

# =========================================================
# SESSION STATE INIT
# =========================================================

if "mode" not in st.session_state:
    st.session_state.mode = "movie"
if "watched_items" not in st.session_state:
    load_mode_state(st.session_state.mode)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="brand-title" >STOOBID</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-title">SHAASTRIANS</div>', unsafe_allow_html=True)
    st.markdown("---")

    old_mode = st.session_state.mode
    mode_label = st.selectbox("MODE", ["Movies", "Anime"], index=0 if old_mode == "movie" else 1)
    new_mode = "movie" if mode_label == "Movies" else "anime"
    if new_mode != old_mode:
        st.session_state.mode = new_mode
        load_mode_state(new_mode)
        st.rerun()

    st.markdown('<div class="section-heading" style="margin-top:0.6rem;">Recently Rated</div>', unsafe_allow_html=True)
    if st.session_state.watched_items:
        for item in reversed(st.session_state.watched_items[-8:]):
            st.markdown(f'<div class="history-item">{item}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#7a7a7a; font-size:0.85rem;">Nothing rated yet.</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Reset Profile", use_container_width=True):
        profile_file, history_file = file_paths_for(st.session_state.mode)
        if os.path.exists(profile_file):
            os.remove(profile_file)
        if os.path.exists(history_file):
            os.remove(history_file)
        st.session_state.watched_items = []
        st.session_state.user_emb = None
        st.success("Profile reset.")
        st.rerun()

mode = st.session_state.mode

# =========================================================
# TOP BRAND HEADER
# =========================================================

st.markdown('<div class="brand-title">STOOBID SHAASTRIANS</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Hybrid AI Recommendation System</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# =========================================================
# SEARCH SECTION
# =========================================================

label = "Movies" if mode == "movie" else "Anime"
st.markdown(f'<div class="section-heading">Search {label}</div>', unsafe_allow_html=True)

query = st.text_input(
    "search",
    placeholder=f"Search {label.lower()}...",
    label_visibility="collapsed",
)

if query:
    search_fn = modules["search_anime"] if mode == "anime" else modules["search_movies"]
    try:
        search_results = search_fn(query, top_n=5)
    except TypeError:
        # fall back if search fn doesn't accept top_n
        search_results = search_fn(query)[:5]

    if search_results:
        cols = st.columns(len(search_results))
        for idx, title in enumerate(search_results):
            card_key = f"flip_search_{mode}_{title}"
            if card_key not in st.session_state:
                st.session_state[card_key] = False

            with cols[idx]:
                if not st.session_state[card_key]:
                    render_poster_block(title, mode)
                    if st.button("RATE", key=f"btn_search_{mode}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = True
                        st.rerun()
                else:
                    st.markdown('<div class="rate-back"><div class="label">Select Rating</div>', unsafe_allow_html=True)
                    star_cols = st.columns(5)
                    picked = None
                    for r in range(1, 6):
                        if star_cols[r - 1].button(f"{r}★", key=f"search_star_{mode}_{r}_{idx}_{title}"):
                            picked = r
                    if picked is not None:
                        process_rating(title, picked)
                        st.session_state[card_key] = False
                        st.toast(f"Rated '{title}' {picked}★")
                        st.rerun()
                    if st.button("Skip", key=f"search_skip_{mode}_{idx}_{title}", use_container_width=True):
                        process_rating(title, "skip")
                        st.session_state[card_key] = False
                        st.rerun()
                    if st.button("Cancel", key=f"search_cancel_{mode}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No matches found. Check your spelling!")

# =========================================================
# RECOMMENDATIONS SECTION
# =========================================================

# Only render Recommendations Header and shelf if the user embedding has been initialized (Warm-start)
if st.session_state.user_emb is not None:
    st.markdown('<div class="section-heading">Recommended for you</div>', unsafe_allow_html=True)
    
    recommend_fn = modules["recommend_anime_from_embedding"] if mode == "anime" else modules["recommend_movies_from_embedding"]
    try:
        recs = recommend_fn(st.session_state.user_emb, st.session_state.watched_items, top_k=10)
    except Exception as e:
        recs = []
        st.error(f"Error generating recommendations: {e}")

    if recs:
        rec_cols = st.columns(5)
        for idx, (title, score) in enumerate(recs):
            card_key = f"flip_rec_{mode}_{title}"
            if card_key not in st.session_state:
                st.session_state[card_key] = False

            with rec_cols[idx % 5]:
                if not st.session_state[card_key]:
                    render_poster_block(title, mode, score=float(score))
                    if st.button("RATE", key=f"btn_rec_{mode}_{idx}_{title}", use_container_width=True, type="secondary"):
                        st.session_state[card_key] = True
                        st.rerun()
                else:
                    st.markdown('<div class="rate-back"><div class="label">Select Rating</div>', unsafe_allow_html=True)
                    star_cols = st.columns(5)
                    picked = None
                    for r in range(1, 6):
                        if star_cols[r - 1].button(f"{r}★", key=f"rec_star_{mode}_{r}_{idx}_{title}"):
                            picked = r
                    if picked is not None:
                        process_rating(title, picked)
                        st.session_state[card_key] = False
                        st.toast(f"Rated '{title}' {picked}★")
                        st.rerun()
                    if st.button("Skip", key=f"rec_skip_{mode}_{idx}_{title}", use_container_width=True, type="secondary"):
                        process_rating(title, "skip")
                        st.session_state[card_key] = False
                        st.rerun()
                    if st.button("Cancel", key=f"rec_cancel_{mode}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CATEGORY DISCOVERY SHELVES (Sideways-Scrollable Onboarding)
# =========================================================

genre_shelves = get_top_by_genre(mode, n=10)

# Display popular onboarding genre rows based on profile state:
if st.session_state.user_emb is None:
    # Cold-Start: Show open, scrollable onboarding rows immediately
    st.markdown('<div class="section-heading">Explore Popular Categories</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8a8a8a; font-size:0.92rem; margin-bottom:1.5rem;">Rate some popular titles from your favorite categories to jumpstart your personalized recommendations.</div>', unsafe_allow_html=True)
    
    for genre, titles in genre_shelves.items():
        st.markdown(f'<div class = "brand-title-card" style=" margin-top:1.4rem; margin-bottom:0.6rem;"> {genre}</div>', unsafe_allow_html=True)
        cols = st.columns(len(titles))
        for idx, title in enumerate(titles):
            card_key = f"flip_genre_{mode}_{genre}_{title}"
            if card_key not in st.session_state:
                st.session_state[card_key] = False
            
            with cols[idx]:
                if idx == 0:
                    st.markdown('<div class="scroll-anchor"></div>', unsafe_allow_html=True)
                
                if not st.session_state[card_key]:
                    render_poster_block(title, mode)
                    if st.button("RATE", key=f"btn_genre_{mode}_{genre}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = True
                        st.rerun()
                else:
                    st.markdown('<div class="rate-back"><div class="label">Select Rating</div>', unsafe_allow_html=True)
                    star_cols = st.columns(5)
                    picked = None
                    for r in range(1, 6):
                        if star_cols[r - 1].button(f"{r}★", key=f"genre_star_{mode}_{genre}_{r}_{idx}_{title}"):
                            picked = r
                    if picked is not None:
                        process_rating(title, picked)
                        st.session_state[card_key] = False
                        st.toast(f"Rated '{title}' {picked}★")
                        st.rerun()
                    if st.button("Skip", key=f"genre_skip_{mode}_{genre}_{idx}_{title}", use_container_width=True, type="secondary"):
                        process_rating(title, "skip")
                        st.session_state[card_key] = False
                        st.rerun()
                    if st.button("Cancel", key=f"genre_cancel_{mode}_{genre}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Warm-Start: Keep every genre row in a single horizontal scrolling row below the recommendations
    st.markdown('<div class="section-heading">Explore Popular Categories</div>', unsafe_allow_html=True)
    
    for genre, titles in genre_shelves.items():
        st.markdown(f'<div class = "brand-title-card" style=" margin-top:1.4rem; margin-bottom:0.6rem;"> {genre}</div>', unsafe_allow_html=True)
        cols = st.columns(len(titles))
        for idx, title in enumerate(titles):
            card_key = f"flip_genre_{mode}_{genre}_{title}"
            if card_key not in st.session_state:
                st.session_state[card_key] = False
            
            with cols[idx]:
                if idx == 0:
                    st.markdown('<div class="scroll-anchor"></div>', unsafe_allow_html=True)
                
                if not st.session_state[card_key]:
                    render_poster_block(title, mode)
                    if st.button("RATE", key=f"btn_genre_{mode}_{genre}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = True
                        st.rerun()
                else:
                    st.markdown('<div class="rate-back"><div class="label">Select Rating</div>', unsafe_allow_html=True)
                    star_cols = st.columns(5)
                    picked = None
                    for r in range(1, 6):
                        if star_cols[r - 1].button(f"{r}★", key=f"genre_star_{mode}_{genre}_{r}_{idx}_{title}"):
                            picked = r
                    if picked is not None:
                        process_rating(title, picked)
                        st.session_state[card_key] = False
                        st.toast(f"Rated '{title}' {picked}★")
                        st.rerun()
                    if st.button("Skip", key=f"genre_skip_{mode}_{genre}_{idx}_{title}", use_container_width=True, type="secondary"):
                        process_rating(title, "skip")
                        st.session_state[card_key] = False
                        st.rerun()
                    if st.button("Cancel", key=f"genre_cancel_{mode}_{genre}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)