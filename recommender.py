import numpy as np
import pandas as pd
import faiss
import os
from sklearn.preprocessing import normalize
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build relative paths automatically
master_movie = pd.read_csv(os.path.join(BASE_DIR, "Preprocessed data", "master_movies.csv"))
master_anime = pd.read_csv(os.path.join(BASE_DIR, "Preprocessed data", "master_anime.csv"))

movie_embeddings = np.load(os.path.join(BASE_DIR, "features", "movie_final_embeddings.npy")).astype(np.float32)
anime_embeddings = np.load(os.path.join(BASE_DIR, "features", "anime_final_embeddings.npy")).astype(np.float32)

movie_index = faiss.read_index(os.path.join(BASE_DIR, "Indexes", "movie.index"))
anime_index = faiss.read_index(os.path.join(BASE_DIR, "Indexes", "anime.index"))



movie_title_to_row = {
    title: idx
    for idx, title in enumerate(master_movie["title"])
}
anime_title_to_row = {
    title: idx
    for idx, title in enumerate(master_anime["title"])
}

import re

def normalize_title(title):

    title = title.lower()

    title = re.sub(r"\(\d{4}\)", "", title)

    title = title.strip()

    if title.endswith(", the"):
        title = "the " + title[:-5]

    elif title.endswith(", a"):
        title = "a " + title[:-3]

    elif title.endswith(", an"):
        title = "an " + title[:-4]

    title = re.sub(r"[^\w\s]", " ", title)
    
    return title.strip()

master_movie["search_title"] = master_movie["title"].apply(normalize_title)
master_anime["search_title"] = master_anime["title"].apply(normalize_title)

movie_titles = master_movie["title"].to_numpy()
movie_search_titles = master_movie["search_title"].to_numpy()

anime_titles = master_anime["title"].to_numpy()
anime_search_titles = master_anime["search_title"].to_numpy()

movie_popularity = master_movie["popularity"].tolist()

def search_movies(query, top_n=10):

    query = normalize_title(query)

    starts = []
    word_starts = []

    for title, search_title, popularity in zip(
        movie_titles,
        movie_search_titles,
        movie_popularity
    ):

        if search_title.startswith(query):
            starts.append((popularity, title))

        elif any(
            word.startswith(query)
            for word in search_title.split()
        ):
            word_starts.append((popularity, title))

    starts.sort(reverse=True)
    word_starts.sort(reverse=True)
    
    results = starts + word_starts

    return [title for _, title in results[:top_n]]

anime_popularity =  (
    master_anime["score"] *
    np.log1p(master_anime["members"])
).tolist()

def search_anime(query, top_n=10):

    query = normalize_title(query)

    starts = []
    word_starts = []

    for title, search_title, popularity in zip(
        anime_titles,
        anime_search_titles,
        anime_popularity
    ):

        if search_title.startswith(query):
            starts.append((popularity, title))

        elif any(
            word.startswith(query)
            for word in search_title.split()
        ):
            word_starts.append((popularity, title))

    starts.sort(reverse=True)
    word_starts.sort(reverse=True)

    results = starts + word_starts

    return [title for _, title in results[:top_n]]

def get_movie_user_embedding(movie_titles):

    rows = []

    for title in movie_titles:

        if title in movie_title_to_row:
            rows.append(movie_title_to_row[title])

    if len(rows) == 0:
        raise ValueError("No valid movie titles found.")

    user_embedding = np.mean(movie_embeddings[rows], axis=0)

    user_embedding = normalize(
        user_embedding.reshape(1, -1),
        norm="l2"
    )

    return user_embedding.astype(np.float32)

def get_anime_user_embedding(anime_titles):

    rows = []

    for title in anime_titles:

        if title in anime_title_to_row:
            rows.append(anime_title_to_row[title])

    if len(rows) == 0:
        raise ValueError("No valid movie titles found.")

    user_embedding = np.mean(anime_embeddings[rows], axis=0)

    user_embedding = normalize(
        user_embedding.reshape(1, -1),
        norm="l2"
    )

    return user_embedding.astype(np.float32)

def recommend_movies(movie_titles, top_k=10):

    user_embedding = get_movie_user_embedding(movie_titles)

    D, I = movie_index.search(user_embedding, k=top_k + len(movie_titles))

    recommendations = []

    for score, idx in zip(D[0], I[0]):

        title = master_movie.iloc[idx]["title"]

        if title not in movie_titles:

            recommendations.append(
                (title, float(score))
            )

        if len(recommendations) >= top_k:
            break

    return recommendations

import re

def same_franchise(input_title, rec_title):

    input_title = normalize_title(input_title)
    rec_title = normalize_title(rec_title)

    input_title = re.sub(r"[^\w\s]", " ", input_title)
    rec_title = re.sub(r"[^\w\s]", " ", rec_title)

    input_words = set(input_title.split())
    rec_words = set(rec_title.split())

    stopwords = {
    "movie", "ova", "special", "tv", "season",
    "part", "episode", "ii", "iii", "iv",
    "the", "a", "an", "no",
    "one", "two", "three"
    }

    input_words -= stopwords
    rec_words -= stopwords

    return len(input_words & rec_words) > 0


def recommend_anime(anime_titles, top_k=10):

    anime_titles_input = set(anime_titles)

    user_embedding = get_anime_user_embedding(anime_titles)

    search_k = 30

    D, I = anime_index.search(user_embedding, search_k)

    recommendations = []

    for score, idx in zip(D[0], I[0]):

        title = master_anime.iloc[idx]["title"]

        # Skip exact anime selected by user
        if title in anime_titles_input:
            continue

        # Skip same franchise
        skip = False

        for fav in anime_titles_input:
            if same_franchise(fav, title):
                skip = True
                break

        if skip:
            continue

        recommendations.append(
            (title, round(float(score), 4))
        )

        if len(recommendations) >= top_k:
            break

    return recommendations

# ----------------------------
# Dynamic Movie Recommendation
# ----------------------------

def update_movie_embedding(user_embedding, movie_title, rating, alpha=0.20):

    if movie_title not in movie_title_to_row:
        return user_embedding

    movie_emb = movie_embeddings[
        movie_title_to_row[movie_title]
    ]

    movie_emb = movie_emb.reshape(1, -1)

    rating_weights = {
        1: -1.5,
        2: -0.75,
        3: 0.0,
        4: 0.75,
        5: 1.5,
        "skip": 0.0
    }

    weight = rating_weights.get(rating, 0.0)

    updated = (1.0 - alpha) * user_embedding + alpha * (weight * movie_emb)

    updated = normalize(updated, norm="l2")

    return updated.astype(np.float32)


def recommend_movies_from_embedding(
    user_embedding,
    watched_movies,
    top_k=10
):

    watched_movies = set(watched_movies)

    D, I = movie_index.search(
        user_embedding,
        k=top_k + len(watched_movies)
    )

    recommendations = []

    for score, idx in zip(D[0], I[0]):

        title = master_movie.iloc[idx]["title"]

        if title in watched_movies:
            continue

        recommendations.append(
            (title, float(score))
        )

        if len(recommendations) >= top_k:
            break

    return recommendations

# ----------------------------
# Dynamic Anime Recommendation
# ----------------------------

def update_anime_embedding(user_embedding, anime_title, rating, alpha=0.20):

    if anime_title not in anime_title_to_row:
        return user_embedding

    anime_emb = anime_embeddings[
        anime_title_to_row[anime_title]
    ]

    anime_emb = anime_emb.reshape(1, -1)

    rating_weights = {
        1: -1.5,
        2: -0.75,
        3: 0.0,
        4: 0.75,
        5: 1.5,
        "skip": 0.0
    }

    weight = rating_weights.get(rating, 0.0)

    updated = (1.0 - alpha) * user_embedding + alpha * (weight * anime_emb)

    updated = normalize(updated, norm="l2")

    return updated.astype(np.float32)


def recommend_anime_from_embedding(
    user_embedding,
    watched_anime,
    top_k=10
):

    watched_anime = set(watched_anime)

    D, I = anime_index.search(
        user_embedding,
        40
    )

    recommendations = []

    for score, idx in zip(D[0], I[0]):

        title = master_anime.iloc[idx]["title"]

        if title in watched_anime:
            continue

        skip = False

        for fav in watched_anime:

            if same_franchise(fav, title):
                skip = True
                break

        if skip:
            continue

        recommendations.append(
            (title, float(score))
        )

        if len(recommendations) >= top_k:
            break

    return recommendations

if __name__ == "__main__":
    import os

    # 1. Choose Recommendation Mode
    print("\n" + "="*50)
    print("      SELECT RECOMMENDATION MODE")
    print("="*50)
    print("1. Movies Recommender")
    print("2. Anime Recommender")
    mode_choice = input("\nEnter choice (1-2): ").strip()

    if mode_choice == "2":
        mode = "anime"
        PROFILE_FILE = "user_taste_anime.npy"
        WATCH_HISTORY_FILE = "watch_history_anime.txt"
        embeddings = anime_embeddings
        title_to_row = anime_title_to_row
        index = anime_index
        master_df = master_anime
        search_fn = search_anime
        recommend_fn = recommend_anime_from_embedding
        get_emb_fn = get_anime_user_embedding
    else:
        mode = "movie"
        PROFILE_FILE = "user_taste_movie.npy"
        WATCH_HISTORY_FILE = "watch_history_movie.txt"
        embeddings = movie_embeddings
        title_to_row = movie_title_to_row
        index = movie_index
        master_df = master_movie
        search_fn = search_movies
        recommend_fn = recommend_movies_from_embedding
        get_emb_fn = get_movie_user_embedding

    # 2. Load existing history or start fresh
    if os.path.exists(WATCH_HISTORY_FILE):
        with open(WATCH_HISTORY_FILE, "r", encoding="utf-8") as f:
            watched_items = [line.strip() for line in f.readlines() if line.strip()]
    else:
        watched_items = []

    # 3. Load existing taste profile or initialize empty
    if os.path.exists(PROFILE_FILE):
        user_emb = np.load(PROFILE_FILE)
        print(f"\n[INFO] Loaded existing dynamically evolved {mode} taste profile from disk!")
    else:
        if watched_items:
            user_emb = get_emb_fn(watched_items)
        else:
            user_emb = None
        print(f"\n[INFO] Starting with a fresh, empty {mode} taste profile.")

    # 4. Generic EWMA update function for both Movies and Anime
    def update_embedding_exponential(user_embedding, item_title, rating, item_embeddings, item_title_to_row, beta=0.30):
        if item_title not in item_title_to_row:
            return user_embedding
        emb = item_embeddings[item_title_to_row[item_title]].reshape(1, -1)
        rating_weights = {
            1: -1.5,
            2: -0.75,
            3: 0.0,
            4: 0.75,
            5: 1.5,
            "skip": 0.0
        }
        weight = rating_weights.get(rating, 0.0)
        updated = (1.0 - beta) * user_embedding + beta * (weight * emb)
        return normalize(updated, norm="l2").astype(np.float32)

    # 5. Interactive Loop
    while True:
        print("\n" + "="*50)
        print(f"      DYNAMIC {mode.upper()} RECOMMENDATION SYSTEM")
        print("="*50)
        
        # Display current history
        if watched_items:
            print(f"Current History: {', '.join(watched_items[-5:])} (showing last 5)")
        else:
            print("Current History: Empty")
        
        # Display current recommendations
        print("\n--- Current Top Recommendations ---")
        if user_emb is not None:
            try:
                recs = recommend_fn(user_emb, watched_items, top_k=5)
                for i, (title, score) in enumerate(recs, 1):
                    print(f"  {i}. {title} (score: {score:.4f})")
            except Exception as e:
                print(f"  Error generating recommendations: {e}")
        else:
            print(f"  No recommendations yet. Please rate some {mode} titles first!")

        print("\nMenu Options:")
        print(f"1. Rate / Add an {mode}")
        print("2. Reset profile")
        print("3. Save and Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            query = input(f"\nEnter the title of the {mode} you want to rate: ").strip()
            if not query:
                print("Empty input.")
                continue
            
            exact_match = None
            # Check for exact case-sensitive match first
            if query in title_to_row:
                exact_match = query
            else:
                # Automate search and select to remove redundant search step
                print(f"\nNo exact match found for '{query}'. Searching database...")
                search_results = search_fn(query, top_n=5)
                
                if not search_results:
                    print("  No matches found in the database. Try another search spelling.")
                    continue
                
                print("\nWe found these matches:")
                for idx, title in enumerate(search_results, 1):
                    print(f"  {idx}. {title}")
                
                selection = input(f"\nSelect title (1-{len(search_results)}) or press Enter to cancel: ").strip()
                if selection.isdigit():
                    sel_idx = int(selection) - 1
                    if 0 <= sel_idx < len(search_results):
                        exact_match = search_results[sel_idx]
                    else:
                        print("Invalid selection.")
                        continue
                else:
                    print("Rating cancelled.")
                    continue
            
            # Rate the matched movie/anime
            if exact_match:
                rating_input = input(f"Enter rating for '{exact_match}' (1-5, or 'skip'): ").strip().lower()
                if rating_input in ["1", "2", "3", "4", "5", "skip"]:
                    if rating_input != "skip":
                        rating = int(rating_input)
                    else:
                        rating = "skip"
                    
                    # Update profile
                    if user_emb is None:
                        user_emb = get_emb_fn([exact_match])
                    else:
                        user_emb = update_embedding_exponential(
                            user_emb, exact_match, rating, embeddings, title_to_row, beta=0.30
                        )
                    
                    if exact_match not in watched_items:
                        watched_items.append(exact_match)
                    print(f"✓ Rated '{exact_match}' as {rating_input}. Taste vector evolved!")
                else:
                    print("Invalid rating.")

        elif choice == "2":
            confirm = input(f"Are you sure you want to reset your {mode} profile? (y/n): ").strip().lower()
            if confirm == "y":
                watched_items = []
                user_emb = None
                if os.path.exists(PROFILE_FILE):
                    os.remove(PROFILE_FILE)
                if os.path.exists(WATCH_HISTORY_FILE):
                    os.remove(WATCH_HISTORY_FILE)
                print("✓ Profile successfully reset!")

        elif choice == "3":
            # Save variables and exit
            if user_emb is not None:
                np.save(PROFILE_FILE, user_emb)
            if watched_items:
                with open(WATCH_HISTORY_FILE, "w", encoding="utf-8") as f:
                    for m in watched_items:
                        f.write(m + "\n")
            print(f"\nSaving {mode} taste vector and exiting. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")