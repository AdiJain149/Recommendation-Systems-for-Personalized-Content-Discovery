import os
import time
import pickle
from pathlib import Path
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
import mlflow

# Framework training imports
from surprise import Dataset, Reader, SVD, accuracy as surprise_accuracy
from surprise.model_selection import train_test_split as surprise_split
from implicit.als import AlternatingLeastSquares

# Import our custom evaluation metrics engine
from src.models.metrics import calculate_ranking_metrics
from src.models.engine import RecommendationEngine

def build_ground_truth_and_samples(data_path: Path, test_user_count: int = 500, seed: int = 42):
    """
    Extracts explicit validation ground truth from data split to compute ranking metrics.
    Ensures all IDs are consistently cast to integers.
    """
    print("[*] Creating validation ground truth maps via Polars...")
    df = pl.read_parquet(data_path)
    
    # Stratified grouping: gather all movies watched per user
    user_history = df.group_by("user_id").agg(pl.col("movie_id"))
    
    user_gt = {
        int(row["user_id"]): {int(mid) for mid in row["movie_id"]} 
        for row in user_history.iter_rows(named=True)
    }
    
    # Sample a fixed subset of users to test during validation to optimize speed
    np.random.seed(seed)
    unique_users = list(user_gt.keys())
    sample_users = np.random.choice(
        unique_users, 
        size=min(test_user_count, len(unique_users)), 
        replace=False
    ).tolist()
    
    return user_gt, sample_users


def evaluate_engine_ranking(engine: RecommendationEngine, test_users: list, ground_truth: dict, k: int = 10) -> dict:
    """Computes mean ranking metrics for SVD engine ensuring valid item overlaps."""
    p_all, r_all, map_all, ndcg_all = [], [], [], []
    
    # Extract the exact pool of valid item IDs in our slice to filter out noise
    valid_item_pool = set()
    for items in ground_truth.values():
        valid_item_pool.update(items)
        
    for uid in test_users:
        actuals = ground_truth.get(int(uid), set())
        if not actuals:
            continue
            
        # Extract predictions from engine
        recs = engine.get_top_k_recommendations(user_id=uid, k=k)
        
        # Filter predictions: only keep items present in our current dataset slice
        pred_ids = [int(r["movie_id"]) for r in recs if int(r["movie_id"]) in valid_item_pool]
        
        # Fallback if filtering clears the list completely
        if not pred_ids:
            pred_ids = [int(r["movie_id"]) for r in recs][:k]
            
        metrics = calculate_ranking_metrics(pred_ids, actuals, k=k)
        p_all.append(metrics["precision"])
        r_all.append(metrics["recall"])
        map_all.append(metrics["map"])
        ndcg_all.append(metrics["ndcg"])
        
    return {
        f"precision_at_{k}": float(np.mean(p_all)) if p_all else 0.0,
        f"recall_at_{k}": float(np.mean(r_all)) if r_all else 0.0,
        f"map_at_{k}": float(np.mean(map_all)) if map_all else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcg_all)) if ndcg_all else 0.0
    }

def run_tracked_svd(data_path: Path, output_dir: Path, test_users: list, ground_truth: dict):
    """Trains SVD and logs parameters, traditional errors, and uniform ranking metrics to MLflow."""
    print("\n" + "="*60)
    print("[MLFLOW RUN] Initiating Uniformly Tracked SVD Pipeline")
    print("="*60)
    
    df = pl.read_parquet(data_path).select(["user_id", "movie_id", "rating"])
    pandas_df = df.to_pandas()
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(pandas_df[["user_id", "movie_id", "rating"]], reader)
    trainset, testset = surprise_split(data, test_size=0.2, random_state=42)
    
    factors, epochs, lr = 50, 20, 0.005
    
    with mlflow.start_run(run_name="SVD_Structured_Run"):
        mlflow.log_param("algorithm", "SVD")
        mlflow.log_param("n_factors", factors)
        mlflow.log_param("n_epochs", epochs)
        mlflow.log_param("lr_all", lr)
        
        # Fit SVD baseline
        start_time = time.time()
        model = SVD(n_factors=factors, n_epochs=epochs, lr_all=lr, random_state=42)
        model.fit(trainset)
        duration = time.time() - start_time
        
        # Traditional ratings error evaluation
        predictions = model.test(testset)
        rmse = surprise_accuracy.rmse(predictions, verbose=False)
        mae = surprise_accuracy.mae(predictions, verbose=False)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("training_duration_sec", duration)
        
        # Save model artifact locally
        local_pkl = output_dir / "svd_model.pkl"
        from surprise.dump import dump as surprise_dump
        surprise_dump(str(local_pkl), algo=model)
        
        print("[*] Instantiating SVD engine for IR metric evaluation...")
        # FIX: Directly pass the trained SVD model object into the engine instance
        engine = RecommendationEngine(local_pkl, data_path)
        engine.model = model
        engine.algo_type = "SVD"
        
        ranking_metrics = evaluate_engine_ranking(engine, test_users, ground_truth, k=10)
        
        mlflow.log_metrics(ranking_metrics)
        mlflow.log_artifact(str(local_pkl), artifact_path="model_binaries")
        
        print("\n[+] SVD Run Completed and Registered Successfully:")
        print(f"    RMSE: {rmse:.4f} | NDCG@10: {ranking_metrics['ndcg_at_10']:.4f} | Duration: {duration:.2f}s")

def run_tracked_als(data_path: Path, output_dir: Path, test_users: list, ground_truth: dict):
    """Trains Implicit ALS with optimized native mappings and logs uniform tracking metrics to MLflow."""
    print("\n" + "="*60)
    print("[MLFLOW RUN] Initiating Uniformly Tracked Implicit ALS Pipeline")
    print("="*60)
    
    df = pl.read_parquet(data_path).select(["user_id", "movie_id", "rating"])
    
    print("[*] Vectorizing matrix mappings utilizing Polars hash-joins...")
    unique_users = df["user_id"].unique().to_numpy()
    unique_movies = df["movie_id"].unique().to_numpy()
    
    user_map_df = pl.DataFrame({"user_id": unique_users, "user_idx": np.arange(len(unique_users), dtype=np.int32)})
    movie_map_df = pl.DataFrame({"movie_id": unique_movies, "movie_idx": np.arange(len(unique_movies), dtype=np.int32)})
    
    mapped_df = df.join(user_map_df, on="user_id", how="left").join(movie_map_df, on="movie_id", how="left")
    
    user_idxs = mapped_df["user_idx"].to_numpy()
    movie_idxs = mapped_df["movie_idx"].to_numpy()
    ratings = mapped_df["rating"].to_numpy().astype(np.float32)
    
    # Create Item-User matrix directly for implicit's training layout
    item_user_matrix = csr_matrix(
        (ratings, (movie_idxs, user_idxs)), 
        shape=(len(unique_movies), len(unique_users))
    )
    # Generate the transposed view for fast row-based slice lookups during evaluation
    user_item_matrix = item_user_matrix.T.tocsr()
    
    factors, iterations, reg = 64, 15, 0.1
    
    with mlflow.start_run(run_name="ALS_Structured_Run"):
        mlflow.log_param("algorithm", "ALS_Implicit")
        mlflow.log_param("n_factors", factors)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("regularization", reg)
        
        start_time = time.time()
        model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iterations, random_state=42, num_threads=1)
        
        print("[*] Fitting ALS Matrix Factorization components...")
        model.fit(item_user_matrix, show_progress=False)
        duration = time.time() - start_time
        
        mlflow.log_metric("training_duration_sec", duration)
        mlflow.log_metric("matrix_density", float(item_user_matrix.nnz / (item_user_matrix.shape[0] * item_user_matrix.shape[1])))
        
        local_pkl = output_dir / "als_model.pkl"
        output_bundle = {
            "model": model,
            "user_to_idx": {int(uid): int(idx) for idx, uid in enumerate(unique_users)},
            "movie_to_idx": {int(mid): int(idx) for idx, mid in enumerate(unique_movies)},
            "idx_to_movie": {int(idx): int(mid) for mid, idx in enumerate(unique_movies)}
        }
        with open(local_pkl, "wb") as f:
            pickle.dump(output_bundle, f)
            
        print("[*] Evaluating ALS latent spaces via uniform IR metrics...")
        p_all, r_all, map_all, ndcg_all = [], [], [], []
        user_to_idx = output_bundle["user_to_idx"]
        idx_to_movie = output_bundle["idx_to_movie"]
        
        for uid in test_users:
            actuals = ground_truth.get(int(uid), set())
            if int(uid) not in user_to_idx or not actuals:
                continue
                
            u_offset = user_to_idx[int(uid)]
            
            # Defensive check: skip if the index is out of bounds for the trained model factors
            if u_offset >= model.user_factors.shape[0]:
                continue
                
            try:
                ids, _ = model.recommend(u_offset, user_item_matrix[u_offset], N=10, filter_already_liked_items=True)
                pred_ids = []
                for i in ids:
                    movie_id = idx_to_movie.get(int(i))
                    if movie_id is not None:
                        pred_ids.append(int(movie_id))
                
                metrics = calculate_ranking_metrics(pred_ids, actuals, k=10)
                p_all.append(metrics["precision"])
                r_all.append(metrics["recall"])
                map_all.append(metrics["map"])
                ndcg_all.append(metrics["ndcg"])
            except Exception:
                continue
            
        ranking_results = {
            "precision_at_10": float(np.mean(p_all)) if p_all else 0.0,
            "recall_at_10": float(np.mean(r_all)) if r_all else 0.0,
            "map_at_10": float(np.mean(map_all)) if map_all else 0.0,
            "ndcg_at_10": float(np.mean(ndcg_all)) if ndcg_all else 0.0
        }
        
        mlflow.log_metrics(ranking_results)
        mlflow.log_artifact(str(local_pkl), artifact_path="model_binaries")
        
        print("\n[+] ALS Run Completed and Registered Successfully:")
        print(f"    NDCG@10: {ranking_results['ndcg_at_10']:.4f} | Duration: {duration:.2f}s")


def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "processed" / "sample_10pct.parquet"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_absolute_path = "sqlite:///D:\\ADI JAIN\\Projects\\Recommendation Systems for Personalized Content Discovery\\mlflow.db"
    print(f"[*] Connecting to database tracking backend at: {db_absolute_path}")
    mlflow.set_tracking_uri(db_absolute_path)
    
    mlflow.set_experiment("Netflix_Recommendation_System")
    
    ground_truth, test_users = build_ground_truth_and_samples(data_path, test_user_count=500)
    
    run_tracked_svd(data_path, output_dir, test_users, ground_truth)
    run_tracked_als(data_path, output_dir, test_users, ground_truth)
    
    print("\n" + "="*60)
    print("[SUCCESS] Tracking execution finalized. Launch 'mlflow ui' to compare models.")
    print("="*60 + "\n")


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    print("[*] Starting main orchestration script execution...")
    main()