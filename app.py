"""
STOOBID SHAASTRIANS
Hybrid AI Recommendation System — Streamlit Frontend

Combines:
- Netflix-style dark theme + intro animation + branding
- Real recommender.py integration (master_movie/master_anime, embeddings)
- Exponential taste-decay embedding updates
- Per-mode disk persistence (profile survives refresh/restart)
- Poster images (TMDB CDN) on search + recommendation cards
- Flip-card rating UI (front = poster, back = star picker)

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
# CUSTOM CSS  (Netflix dark theme)
# =========================================================

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

/* Keep header visible (it holds the sidebar toggle), but make it transparent
   and hide only the toolbar icons we don't want */
header {
    background: transparent !important;
}
[data-testid="stToolbar"] {
    right: 2rem;
}
[data-testid="stToolbar"] > div:not(:has(button[title="View app menu"])) {
    /* fallback no-op, see note below */
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #111111;
    color: #f2f2f2;
    font-family: "Helvetica Neue", Arial, sans-serif;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background-color: #0b0b0b;
    border-right: 1px solid #262626;
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
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 1px;
    color: #E50914;
    margin-bottom: 0;
}
.brand-subtitle {
    font-size: 0.95rem;
    color: #9c9c9c;
    margin-top: -6px;
    margin-bottom: 1.2rem;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid #333333;
    background-color: #1b1b1b;
    color: #e6e6e6;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    border-color: #E50914;
    color: #ffffff;
    background-color: #241414;
    box-shadow: 0 0 10px rgba(229, 75, 75, 0.35);
}
button[kind="primary"] {
    background-color: #E50914 !important;
    border-color: #E50914 !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover { background-color: #ff1a25 !important; }

.section-heading {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1.6rem;
    margin-bottom: 0.8rem;
    color: #f2f2f2;
    border-left: 4px solid #E50914;
    padding-left: 10px;
}

.stTextInput > div > div > input {
    background-color: #1b1b1b;
    color: #ffffff;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 1.05rem;
}
.stTextInput > div > div > input:focus {
    border-color: #E50914;
    box-shadow: 0 0 0 1px #E50914;
}

.poster-card {
    background: linear-gradient(145deg, #1b1b1b, #161616);
    border: 1px solid #262626;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 0.6rem;
    animation: fadeIn 0.4s ease-in-out;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.poster-card:hover {
    transform: scale(1.02);
    border-color: #E50914;
    box-shadow: 0 0 18px rgba(229, 9, 20, 0.3);
}
.poster-img-wrap {
    width: 100%;
    aspect-ratio: 2 / 3;
    background-color: #1f1f1f;
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
.poster-placeholder { font-size: 2.4rem; color: #444; }
.poster-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #f0f0f0;
    padding: 0.5rem 0.6rem 0.1rem 0.6rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.poster-score {
    font-size: 0.72rem;
    color: #ff4d55;
    padding: 0 0.6rem 0.5rem 0.6rem;
    font-weight: 600;
}
.rate-back {
    background: linear-gradient(90deg, rgba(229,9,20,0.16), rgba(229,9,20,0.02));
    border: 1px solid #E50914;
    border-radius: 14px;
    padding: 0.8rem;
    margin-bottom: 0.6rem;
    text-align: center;
    animation: fadeIn 0.3s ease-in-out;
}
.rate-back .label {
    font-size: 0.75rem;
    color: #ff8b8f;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #8a8a8a;
    border: 1px dashed #333333;
    border-radius: 14px;
    margin-top: 1.5rem;
}
.history-item {
    background-color: #161616;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
    color: #d8d8d8;
}
hr { border-color: #262626; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
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
    0 {
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
        <div class="word word-1">
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
    st.markdown('<div class="brand-title" style="font-size:1.6rem;">🎬 STOOBID</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">SHAASTRIANS</div>', unsafe_allow_html=True)
    st.markdown("---")

    old_mode = st.session_state.mode
    mode_label = st.selectbox("Mode", ["Movies", "Anime"], index=0 if old_mode == "movie" else 1)
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
                    if st.button("Rate", key=f"btn_search_{mode}_{idx}_{title}", use_container_width=True):
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

st.markdown('<div class="section-heading">Recommended for you</div>', unsafe_allow_html=True)

if st.session_state.user_emb is not None:
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
                    if st.button("Rate", key=f"btn_rec_{mode}_{idx}_{title}", use_container_width=True):
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
                    if st.button("Skip", key=f"rec_skip_{mode}_{idx}_{title}", use_container_width=True):
                        process_rating(title, "skip")
                        st.session_state[card_key] = False
                        st.rerun()
                    if st.button("Cancel", key=f"rec_cancel_{mode}_{idx}_{title}", use_container_width=True):
                        st.session_state[card_key] = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div style="font-weight:600; color:#cfcfcf; margin-bottom:0.3rem;">No recommendations yet.</div>
                <div>Start rating titles to build your taste profile.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-weight:600; color:#cfcfcf; margin-bottom:0.3rem;">No recommendations yet.</div>
            <div>Start rating titles to build your taste profile.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )