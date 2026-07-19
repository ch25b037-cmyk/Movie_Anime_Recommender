import pandas as pd
import gc
# ===========================
# Load Movie Datasets
# ===========================
movie = pd.read_csv("C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/movie.csv")
rat = pd.read_csv("C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/rating.csv")
link = pd.read_csv("C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/link.csv")
gen_score = pd.read_csv("C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/genome_scores.csv")
gen_tag = pd.read_csv("C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/genome_tags.csv")
tmdb = pd.read_csv("C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/TMDB.csv")
print("✓ Movie datasets loaded")

# ===========================
# Optimize Movie Dtypes
# ===========================
movie["movieId"] = movie["movieId"].astype("int32")
rat["userId"] = rat["userId"].astype("int32")
rat["movieId"] = rat["movieId"].astype("int32")
rat["rating"] = rat["rating"].astype("float16")
link["movieId"] = link["movieId"].astype("int32")
link = link.rename(columns={"tmdbId": "id"})
link["id"] = link["id"].fillna(-1).astype("int32")
gen_score["movieId"] = gen_score["movieId"].astype("int32")
gen_score["tagId"] = gen_score["tagId"].astype("int32")
gen_score["relevance"] = gen_score["relevance"].astype("float16")
gen_tag["tagId"] = gen_tag["tagId"].astype("int32")
tmdb["id"] = tmdb["id"].astype("int32")
tmdb["vote_average"] = tmdb["vote_average"].astype("float16")
tmdb["vote_count"] = tmdb["vote_count"].astype("int32")
tmdb["popularity"] = tmdb["popularity"].astype("float32")

print("✓ Movie dtypes optimized")

# ===========================
# Save Movie Files
# ===========================
save_path = "C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/Preprocessed data/"
movie.to_csv(save_path + "movie.csv", index=False)
rat.to_csv(save_path + "mov_rating.csv", index=False)
link.to_csv(save_path + "link.csv", index=False)
gen_score.to_csv(save_path + "gen_score.csv", index=False)
gen_tag.to_csv(save_path + "gen_tag.csv", index=False)
tmdb.to_csv(save_path + "TMDB.csv", index=False)
print("✓ Movie files saved")

# ===========================
# Free Memory
# ===========================
del movie, rat, link, gen_score, gen_tag, tmdb
gc.collect()
print("✓ Movie data removed from memory")

master = pd.read_csv(
    "C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Movie/master_movie.csv"
)

# Integer columns
master["movieId"] = master["movieId"].astype("int32")
master["id"] = master["id"].astype("int32")
master["vote_count"] = master["vote_count"].astype("int32")

# Float columns
master["vote_average"] = master["vote_average"].astype("float32")
master["popularity"] = master["popularity"].astype("float32")

# Save back
master.to_csv(
    "C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/Preprocessed data/master_movies.csv",
    index=False
)

print("✅ master_movies.csv updated successfully!")