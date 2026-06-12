import os
import time
from pathlib import Path
import polars as pl
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dump import dump

def train_svd_baseline(data_path: Path, output_dir: Path):
    """
    Loads Parquet data via Polars, interfaces with scikit-surprise, 
    trains a baseline SVD model, evaluates metrics, and stores the artifact.
    """
    print(f"[*] Loading training target data: {data_path.name}")
    start_time = time.time()
    
    # Read required interaction features from parquet
    # SVD requires user, item, rating column format
    df = pl.read_parquet(data_path).select(["user_id", "movie_id", "rating"])
    
    # Convert Polars to a temporary Pandas DataFrame for Surprise compatibility
    # Explicitly casting ensures zero data alignment issues inside Surprise's Reader
    print("[*] Reformatting interaction tensors for Surprise engine...")
    pandas_df = df.to_pandas()
    
    # Initialize the Surprise custom reader targeting our explicit rating boundaries
    reader = Reader(rating_scale=(1, 5))
    
    # Load dataset from the formatted pandas dataframe
    data = Dataset.load_from_df(pandas_df[["user_id", "movie_id", "rating"]], reader)
    
    # Perform a deterministic 80/20 train/test split for validation profiling
    print("[*] Creating stratified validation splits (80% Train / 20% Test)...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Configure production hyperparameters for baseline SVD
    print("[*] Training SVD baseline model (Latent Factors: 50, Epochs: 20)...")
    model = SVD(n_factors=50, n_epochs=20, random_state=42, verbose=True)
    
    # Execute structural training fit
    model.fit(trainset)
    print("[+] Model fitting complete. Running test evaluations...")
    
    # Run test predictions
    predictions = model.test(testset)
    
    # Compute industry standard validation scores
    rmse = accuracy.rmse(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)
    
    # Ensure serialization output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = output_dir / "svd_model.pkl"
    
    # Serialize the complete trained model state using Surprise's built-in dump utility
    print(f"[*] Serializing trained model artifact to: {model_file_path}")
    dump(str(model_file_path), algo=model)
    
    elapsed = time.time() - start_time
    print(f"[SUCCESS] SVD Baseline pipeline completed in {elapsed:.2f}s.")
    print(f"Final Performance -> RMSE: {rmse:.4f} | MAE: {mae:.4f}\n")

def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "processed" / "sample_10pct.parquet"
    output_dir = base_dir / "outputs"
    
    if not data_path.exists():
        print(f"[ERROR] Clean data target missing at: {data_path}")
        print("Please rerun Phase 2 to regenerate data samples.")
        return
        
    train_svd_baseline(data_path, output_dir)

if __name__ == "__main__":
    main()