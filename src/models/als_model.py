import os
import time
import pickle
from pathlib import Path
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

def train_als_model(data_path: Path, output_dir: Path):
    """
    Loads Parquet data, maps IDs to contiguous indices, builds a high-performance 
    Compressed Sparse Row (CSR) matrix, and trains an accelerated ALS model.
    """
    print(f"[*] Loading training target data: {data_path.name}")
    start_time = time.time()
    
    # Read core columns using Polars
    df = pl.read_parquet(data_path).select(["user_id", "movie_id", "rating"])
    
    # Implicit ALS requires contiguous integer IDs starting at 0 for matrix indexing.
    # We create mapping tables to translate real Netflix IDs to Matrix Offsets.
    print("[*] Creating contiguous user and item indexing maps...")
    unique_users = df["user_id"].unique().to_numpy()
    unique_movies = df["movie_id"].unique().to_numpy()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
    
    # Map the columns to their internal index values
    user_idxs = df["user_id"].map_elements(lambda x: user_to_idx[x], return_dtype=pl.Int32).to_numpy()
    movie_idxs = df["movie_id"].map_elements(lambda x: movie_to_idx[x], return_dtype=pl.Int32).to_numpy()
    ratings = df["rating"].to_numpy().astype(np.float32)
    
    # Construct a Compressed Sparse Row (CSR) User-Item Interaction Matrix
    # Shape: (Number of unique users, Number of unique movies)
    print("[*] Constructing Compressed Sparse Row (CSR) interaction matrix...")
    user_item_matrix = csr_matrix(
        (ratings, (user_idxs, movie_idxs)), 
        shape=(len(unique_users), len(unique_movies))
    )
    
    # Initialize the implicit ALS engine accelerated via native C++ threads
    print("[*] Initializing Implicit ALS Engine (Factors: 64, Regularization: 0.1, Iterations: 15)...")
    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.1,
        iterations=15,
        random_state=42,
        num_threads=0 # 0 auto-detects and uses all available CPU cores on your laptop
    )
    
    # Train the model. implicit expects an Item-User matrix, so we pass the transpose (.T)
    print("[*] Commencing alternating matrix optimization phases...")
    model.fit(user_item_matrix.T, show_progress=True)
    print("[+] ALS optimization loops completed.")
    
    # Bundle the model with its translation maps so the inference layer can read it
    output_bundle = {
        "model": model,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": {idx: mid for mid, idx in movie_to_idx.items()}
    }
    
    # Ensure serialization folder exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / "als_model.pkl"
    
    print(f"[*] Serializing ALS bundle artifact to: {output_file_path}")
    with open(output_file_path, "wb") as f:
        pickle.dump(output_bundle, f)
        
    elapsed = time.time() - start_time
    print(f"[SUCCESS] ALS model successfully compiled and stored in {elapsed:.2f}s.\n")

def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "processed" / "sample_10pct.parquet"
    output_dir = base_dir / "outputs"
    
    if not data_path.exists():
        print(f"[ERROR] Clean data target missing at: {data_path}")
        return
        
    # Set environment variable to suppress implicit library warnings about OpenBLAS
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    train_als_model(data_path, output_dir)

if __name__ == "__main__":
    main()