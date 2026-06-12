import pickle
from pathlib import Path
import numpy as np
import polars as pl
from surprise import dump

# Import the parsing tool we built for Phase 10
from src.models.metadata import load_movie_titles

class RecommendationEngine:
    """
    Unified Inference Engine that abstracts prediction calls for both 
    Collaborative Filtering (SVD) and Implicit Latent Factor (ALS) models.
    Automatically enriches raw numeric recommendations with descriptive movie titles.
    """
    def __init__(self, model_path, data_path=None):
        print("[*] Initializing Recommendation Engine Core...")
        self.model_path = Path(model_path)
        self.data_path = Path(data_path) if data_path else None
        
        # 1. Load the localized serialized machine learning model binaries
        if "als" in self.model_path.name.lower():
            self.algo_type = "ALS"
            with open(self.model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.user_to_idx = bundle["user_to_idx"]
            self.movie_to_idx = bundle["movie_to_idx"]
            self.idx_to_movie = bundle["idx_to_movie"]
        else:
            self.algo_type = "SVD"
            try:
                _, self.model = dump.load(str(self.model_path))
            except Exception:
                with open(self.model_path, "rb") as f:
                    loaded_obj = pickle.load(f)
                if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
                    self.model = loaded_obj[1]
                else:
                    self.model = loaded_obj
            
        print(f"[+] Serialized {self.algo_type} model loaded successfully.")
        
        # 2. Phase 10: Dynamically integrate descriptive movie names
        base_dir = Path(__file__).resolve().parents[2]
        metadata_txt = base_dir / "data" / "raw" / "movie_titles.txt"
        metadata_csv = base_dir / "data" / "raw" / "movie_titles.csv"
        
        if metadata_txt.exists():
            self.movie_titles = load_movie_titles(metadata_txt)
        elif metadata_csv.exists():
            self.movie_titles = load_movie_titles(metadata_csv)
        else:
            print("[!] Warning: No movie metadata file found in data/raw/. Falling back to raw IDs.")
            self.movie_titles = {}

        # 3. Cache user history to prevent recommending items they've already watched
        self.user_history = {}
        if self.data_path and self.data_path.exists():
            print("[*] Building historical user interaction indices...")
            df = pl.read_parquet(self.data_path).select(["user_id", "movie_id"])
            
            grouped = df.group_by("user_id").agg(pl.col("movie_id"))
            for row in grouped.iter_rows(named=True):
                self.user_history[int(row["user_id"])] = {int(mid) for mid in row["movie_id"]}
                
            self.all_items = df["movie_id"].unique().to_numpy().tolist()
            print(f"[+] Engine ready. Cached histories for {len(self.user_history):,} users. Total item pool: {len(self.all_items):,} items.")
        else:
            print("[!] Warning: Interaction tracking path data omitted. Deduplication disabled.")
            self.all_items = []

    def get_top_k_recommendations(self, user_id, k=10):
        """
        Generates personalized, non-overlapping top-K recommendations for a target user.
        Returns a beautifully structured list of dictionaries with titles and scores.
        """
        user_id = int(user_id)
        watched_items = self.user_history.get(user_id, set())
        raw_predictions = []

        # --- EXTRACT VIA VECTORIZED ALS EMBEDDING MATH (BYPASSING C++ WRAPPERS) ---
        if self.algo_type == "ALS":
            if user_id not in self.user_to_idx:
                return []
                
            u_idx = self.user_to_idx[user_id]
            
            # Extract user latent factor vector
            user_factor = self.model.user_factors[u_idx]
            
            # Dot-product calculation against all items across the embedding landscape
            all_scores = self.model.item_factors.dot(user_factor)
            
            # Enumerate scores and rank them
            for internal_id, score in enumerate(all_scores):
                movie_id = self.idx_to_movie.get(internal_id)
                if movie_id is not None and int(movie_id) not in watched_items:
                    raw_predictions.append((int(movie_id), float(score)))
            
            # Sort scores down from best match to lowest
            raw_predictions.sort(key=lambda x: x[1], reverse=True)

        # --- EXTRACT VIA CLASSIC SURPRISE SVD ---
        else:
            if not self.all_items:
                return []
                
            for item_id in self.all_items:
                if int(item_id) in watched_items:
                    continue
                    
                try:
                    if hasattr(self.model.trainset, 'to_inner_iid'):
                        pred = self.model.predict(user_id, item_id)
                    else:
                        pred = self.model.predict(str(user_id), str(item_id))
                except Exception:
                    pred = self.model.predict(str(user_id), str(item_id))
                    
                raw_predictions.append((int(item_id), float(pred.est)))
                
            raw_predictions.sort(key=lambda x: x[1], reverse=True)

        # --- FORMAT OUTPUT WITH ENRICHED METADATA TITLES ---
        formatted_recs = []
        for mid, score in raw_predictions[:k]:
            title_string = self.movie_titles.get(mid, f"Unknown Movie ID: {mid}")
            formatted_recs.append({
                "movie_id": mid,
                "title": title_string,
                "estimated_score": round(score, 4)
            })
            
        return formatted_recs