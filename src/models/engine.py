import time
from pathlib import Path
from typing import List, Dict, Any
import polars as pl
from surprise.dump import load

class RecommendationEngine:
    def __init__(self, model_path: Path, data_path: Path):
        """
        Initializes the recommendation system execution framework by loading the serialized model
        and pre-computing historical user interaction maps for speed optimized inference lookup.
        """
        print(f"[*] Initializing Recommendation Engine Core...")
        
        # Load the serialized Surprise model state
        if not model_path.exists():
            raise FileNotFoundError(f"Model binary not found at: {model_path}")
        _, self.model = load(str(model_path))
        print("[+] Serialized ML model loaded successfully.")

        # Load data to build interaction profiles
        if not data_path.exists():
            raise FileNotFoundError(f"Interaction data not found at: {data_path}")
        
        print("[*] Building historical user interaction indices...")
        df = pl.read_parquet(data_path)
        
        # Extract all unique movie IDs globally available in the system
        self.all_movie_ids = df["movie_id"].unique().to_list()
        
        # Build a high-speed Python dictionary tracking user -> set(watched_movie_ids)
        # This prevents recommending a movie a user has already evaluated.
        grouped = df.group_by("user_id").agg(pl.col("movie_id"))
        self.user_history: Dict[int, set] = {
            row["user_id"]: set(row["movie_id"]) for row in grouped.iter_rows(named=True)
        }
        print(f"[+] Engine ready. Cached histories for {len(self.user_history):,} users. Total item pool: {len(self.all_movie_ids):,} items.")

    def get_top_k_recommendations(self, user_id: int, k: int = 10) -> List[Dict[str, Any]]:
        """
        Generates top K movie recommendations for a specific user ID by scoring all unseen items.
        Handles cold-start users gracefully using global model biases.
        """
        # Determine movies already watched by the user
        watched_movies = self.user_history.get(user_id, set())
        
        # Generate candidate set: All movies minus movies already watched
        candidates = [m_id for m_id in self.all_movie_ids if m_id not in watched_movies]
        
        # Performance Fallback: If a user has watched everything or no candidates remain
        if not candidates:
            return []
            
        # Score candidates using the trained latent factor configurations
        scored_candidates = []
        for m_id in candidates:
            # model.predict returns an object containing 'est' (estimated rating)
            prediction = self.model.predict(uid=user_id, iid=m_id)
            scored_candidates.append({
                "movie_id": m_id,
                "predicted_rating": round(prediction.est, 3)
            })
            
        # Sort candidates by predicted rating descending, limit to K
        scored_candidates.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return scored_candidates[:k]

def main():
    base_dir = Path(__file__).resolve().parents[2]
    model_path = base_dir / "outputs" / "svd_model.pkl"
    data_path = base_dir / "data" / "processed" / "sample_10pct.parquet"
    
    try:
        # Spin up the compiled engine
        engine = RecommendationEngine(model_path, data_path)
        
        # Test profiling on a highly active user identified during our Phase 3 EDA
        test_user = 305344
        print(f"\n[*] Querying top 5 recommendations for Test User: {test_user}...")
        
        start_inference = time.time()
        recommendations = engine.get_top_k_recommendations(user_id=test_user, k=5)
        inference_duration = time.time() - start_inference
        
        print(f"[+] Recommendations compiled in {inference_duration*1000:.2f}ms:")
        for idx, rec in enumerate(recommendations):
            print(f"  Rank {idx+1}: Movie ID {rec['movie_id']} (Predicted Score: {rec['predicted_rating']})")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] Engine runtime failed: {e}")

if __name__ == "__main__":
    main()