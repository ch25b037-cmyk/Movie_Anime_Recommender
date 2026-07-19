import pandas as pd

# -------------------------------
# Load datasets
# -------------------------------

anime = pd.read_csv(
    "C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/Anime/anime.csv"
)

anime_meta = pd.read_csv(
    "C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/raw data/anime_full_dataset.csv"
)
anime_meta["anime_id"] = pd.to_numeric(
    anime_meta["anime_id"],
    errors="coerce"
).astype("Int32")
print("Datasets Loaded")

# -------------------------------
# Keep only anime present in original dataset
# -------------------------------

master_anime = anime_meta[
    anime_meta["anime_id"].isin(anime["anime_id"])
].copy()

print(f"Anime retained : {len(master_anime)}")

# -------------------------------
# Handle missing values
# -------------------------------

master_anime["synopsis"] = master_anime["synopsis"].fillna("")
master_anime["score"] = master_anime["score"].fillna(0)
master_anime["scored_by"] = master_anime["scored_by"].fillna(0)

# Optional (recommended)

master_anime["themes"] = master_anime["themes"].fillna("")
master_anime["demographics"] = master_anime["demographics"].fillna("")
master_anime["studios"] = master_anime["studios"].fillna("")
master_anime["season"] = master_anime["season"].fillna("")
master_anime["source"] = master_anime["source"].fillna("Unknown")

# -------------------------------
# Optimize dtypes
# -------------------------------

master_anime["anime_id"] = master_anime["anime_id"].astype("int32")

master_anime["year"] = (
    pd.to_numeric(master_anime["year"], errors="coerce")
    .fillna(0)
    .astype("int16")
)

master_anime["members"] = (
    pd.to_numeric(master_anime["members"], errors="coerce")
    .fillna(0)
    .astype("int32")
)

master_anime["favorites"] = (
    pd.to_numeric(master_anime["favorites"], errors="coerce")
    .fillna(0)
    .astype("int32")
)

master_anime["rank"] = (
    pd.to_numeric(master_anime["rank"], errors="coerce")
    .fillna(0)
    .astype("int32")
)

master_anime["popularity"] = (
    pd.to_numeric(master_anime["popularity"], errors="coerce")
    .fillna(0)
    .astype("int32")
)

master_anime["score"] = (
    pd.to_numeric(master_anime["score"], errors="coerce")
    .fillna(0)
    .astype("float32")
)

print("Dtypes Optimized")

# -------------------------------
# Save
# -------------------------------

master_anime.to_csv(
    "C:/Users/Shaelesh/Desktop/IITM/PoRs/MLops/Recommedation system/Recomm Prototype/Preprocessed data/master_anime.csv",
    index=False,
)

print("master_anime.csv saved successfully!")