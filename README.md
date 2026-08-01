Stoobid Shaastrians: Hybrid Movie & Anime Recommendation System

A content-based hybrid recommendation system for Movies and Anime that provides real-time personalized recommendations using semantic embeddings, FAISS similarity search, and dynamic user preference adaptation.

Unlike traditional collaborative filtering systems, this recommender is designed to work even for completely new users (cold-start) by requiring only a handful of favourite titles.

Deployed Website Link: https://movieanimerecommender-stoobid-shaastrians.streamlit.app/

---

Features

- Movie Recommendation System
- Anime Recommendation System
- Semantic Embedding-Based Recommendations
- FAISS Approximate Nearest Neighbor Search
- Dynamic User Preference Evolution
- Cold-Start Friendly
- Popularity-Aware Search Suggestions
- Anime Franchise Filtering
- Interactive Streamlit Web Interface

---

Project Architecture

                 Raw Datasets
                      │
                      ▼
               Data Preprocessing
                      │
                      ▼
             Feature Engineering
                      │
                      ▼
          Dense Embedding Generation
                      │
                      ▼
              FAISS Index Creation
                      │
                      ▼
          User Embedding Generation
                      │
                      ▼
       Similarity Search using FAISS
                      │
                      ▼
           Top-K Recommendations
                      │
                      ▼
        User Feedback (1–5 / Skip)
                      │
                      ▼
     Dynamic User Embedding Update
                      │
                      ▼
        Updated Recommendations

---

Datasets Used

Movies

- MovieLens Dataset
  
  - User Ratings
  - User Interaction History

- TMDB Dataset
  
  - Genres
  - Plot Overview
  - Popularity
  - Vote Average
  - Release Date
  - Additional Metadata

---

Anime

- MyAnimeList (MAL) Dataset

Contains

- Title
- Genres
- Synopsis
- Members
- Score
- Popularity
- Studios
- Type
- Additional Metadata

---

Methodology

1. Data Preprocessing

The raw datasets are cleaned and standardized by

- Removing duplicates
- Handling missing values
- Normalizing titles
- Cleaning textual metadata
- Merging useful attributes

Outputs

master_movies.csv
master_anime.csv

---

2. Feature Engineering

Each Movie/Anime is represented as a dense semantic embedding generated using multiple metadata features.

Features include

- Genres
- Plot/Synopsis
- Popularity
- Ratings
- Metadata

Outputs

movie_final_embeddings.npy
anime_final_embeddings.npy

---

3. FAISS Indexing

Facebook AI Similarity Search (FAISS) is used for efficient nearest-neighbor retrieval.

Separate indices are maintained for

movie.index
anime.index

This enables real-time recommendation generation.

---

4. User Embedding

The user initially selects a few favourite titles.

A user embedding is created by averaging their corresponding embeddings.

User Embedding

=

Average(
Embedding₁,
Embedding₂,
...
Embeddingₙ
)

The embedding is L2-normalized before retrieval.

---

5. Recommendation Generation

The user embedding is queried against the FAISS index.

The system

- Retrieves nearest neighbors
- Removes already watched titles
- Filters same-franchise anime
- Returns Top-K recommendations

---

6. Dynamic Preference Learning

Unlike static recommenders, this system updates the user representation immediately after feedback.

Whenever the user rates a recommendation,

Updated User Embedding

=

(1 − α)

×

Old User Embedding

+

α

×

Rating Weight

×

Content Embedding

This allows recommendations to evolve continuously without retraining the model.

---

Evaluation Strategy

Instead of Leave-One-Out evaluation, the recommender is evaluated using a 5-Item Input Protocol.

For every eligible user:

1. Randomly select 5 liked titles
2. Use these as the recommender input
3. Treat all remaining liked titles as ground truth
4. Generate Top-10 recommendations
5. Compute evaluation metrics

This closely simulates the real-world usage of the application.

---

Results

Movies

Metric| Score
Users Evaluated| 5000
Hit Rate@10| 0.8534
Precision@10| 0.2433
Recall@10| 0.0148
NDCG@10| 0.2618

---

Anime

Metric| Score
Users Evaluated| 15072
Hit Rate@10| 0.7979
Precision@10| 0.2010
Recall@10| 0.0119
NDCG@10| 0.2069

---

Project Structure

.
├── raw data/
│   ├── Movie/
│   └── Anime/
│
├── Preprocessed data/
│   ├── master_movies.csv
│   └── master_anime.csv
│
├── features/
│   ├── movie_final_embeddings.npy
│   └── anime_final_embeddings.npy
│
├── Indexes/
│   ├── movie.index
│   └── anime.index
│
├── Notebooks/
│   ├── preprocessing
│   ├── feature engineering
│   ├── FAISS
│   └── evaluation
│
├── recommender.py
├── app.py
└── README.md

---

Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- FAISS
- Streamlit
- MovieLens Dataset
- TMDB Dataset
- MyAnimeList Dataset

---

Future Improvements

- Confidence-aware embedding updates
- Diversity-aware recommendation reranking
- Long-term preference modelling using Exponential Moving Average (EMA)
- Explainable recommendations ("Recommended because you liked...")
- Hybrid collaborative + content-based recommendation

---

Authors

Developed by the Stoobid Shaastrians team as part of an AI/ML recommendation system project.
