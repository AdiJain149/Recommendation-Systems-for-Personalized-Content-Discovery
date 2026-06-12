import time
from pathlib import Path
import polars as pl

def run_eda_diagnostics(file_path: Path):
    """
    Executes an optimized memory-efficient profiling workflow on the target dataset
    using Polars expressions. Outputs critical metrics required for ML design.
    """
    print(f"[*] Loading data target for profiling: {file_path.name}")
    start_time = time.time()
    
    # Read the parquet file lazily
    lf = pl.scan_parquet(file_path)
    
    # Collect total row and unique entity counts simultaneously to optimize the query plan
    counts = lf.select([
        pl.len().alias("total_ratings"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("movie_id").n_unique().alias("unique_movies"),
        pl.col("rating").mean().alias("average_rating")
    ]).collect()
    
    total_ratings = counts["total_ratings"][0]
    unique_users = counts["unique_users"][0]
    unique_movies = counts["unique_movies"][0]
    avg_rating = counts["average_rating"][0]
    
    # Calculate Matrix Sparsity using float precision
    total_possible_interactions = unique_users * unique_movies
    sparsity = (1.0 - (total_ratings / total_possible_interactions)) * 100.0
    
    # Compute Rating Distribution
    rating_dist = lf.group_by("rating").agg(
        pl.len().alias("count")
    ).with_columns(
        (pl.col("count") / total_ratings * 100).round(2).alias("percentage")
    ).sort("rating").collect()
    
    # Compute Top Activity Profiles (User and Movie)
    top_active_users = lf.group_by("user_id").len().sort("len", descending=True).limit(5).collect()
    top_rated_movies = lf.group_by("movie_id").len().sort("len", descending=True).limit(5).collect()
    
    # Print Industry-Grade Diagnostics Report
    print("\n" + "="*50)
    print("         NETFLIX DATASET PROFILE REPORT         ")
    print("="*50)
    print(f"Dataset File:          {file_path.name}")
    print(f"Total Logged Ratings:  {total_ratings:,}")
    print(f"Unique Users Inspected: {unique_users:,}")
    print(f"Unique Movies Tagged:  {unique_movies:,}")
    print(f"Average System Rating: {avg_rating:.2f} / 5.00")
    print(f"Matrix Sparsity Score: {sparsity:.4f}%")
    print("-"*50)
    print("RATING VALUE DISTRIBUTION:")
    for row in rating_dist.iter_rows(named=True):
        print(f"  Rating {row['rating']}: {row['count']:,} rows ({row['percentage']:.2f}%)")
    print("-"*50)
    print("TOP 3 MOST ACTIVE USERS:")
    for idx, row in enumerate(top_active_users.limit(3).iter_rows(named=True)):
        print(f"  {idx+1}. User ID {row['user_id']}: {row['len']:,} ratings submitted")
    print("-"*50)
    print("TOP 3 MOST ACCUMULATED MOVIES:")
    for idx, row in enumerate(top_rated_movies.limit(3).iter_rows(named=True)):
        print(f"  {idx+1}. Movie ID {row['movie_id']}: {row['len']:,} ratings received")
    print("="*50)
    
    elapsed = time.time() - start_time
    print(f"[SUCCESS] Metrics calculated successfully in {elapsed:.2f}s.\n")

def main():
    base_dir = Path(__file__).resolve().parents[2]
    target_file = base_dir / "data" / "processed" / "sample_10pct.parquet"
    
    if not target_file.exists():
        print(f"[ERROR] Target analysis file does not exist at: {target_file}")
        print("Please check PHASE 2 execution outcomes.")
        return
        
    run_eda_diagnostics(target_file)

if __name__ == "__main__":
    main()