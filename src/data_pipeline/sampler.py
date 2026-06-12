import os
import time
from pathlib import Path
import polars as pl

def create_deterministic_samples(processed_dir: Path, seed: int = 42):
    """
    Reads processed full parquet data, samples unique users deterministically,
    and isolates 1% and 10% datasets preserving entire user profiles.
    """
    start_time = time.time()
    print("[*] Initializing sampling sequence...")

    # Discover all processed parquet partitions
    parquet_files = sorted(list(processed_dir.glob("combined_data_*.parquet")))
    
    if not parquet_files:
        print(f"[ERROR] No processed parquet files found in {processed_dir}.")
        print("Please ensure PHASE 1 completed successfully.")
        return

    # Load and combine all parquet partitions lazily to save RAM
    print(f"[*] Reading {len(parquet_files)} parquet partition(s) via LazyFrame...")
    lazy_df = pl.scan_parquet(parquet_files)

    # Collect unique users to perform stratified user sampling
    print("[*] Extracting unique user IDs...")
    unique_users_df = lazy_df.select("user_id").unique().collect()
    total_users = len(unique_users_df)
    print(f"[+] Total unique users found in dataset: {total_users:,}")

    # Process both sample sizes
    sample_fractions = {"1pct": 0.01, "10pct": 0.10}
    
    for label, fraction in sample_fractions.items():
        print(f"\n[*] Generating {label} sample (fraction: {fraction})...")
        
        # Deterministically sample the unique users
        sampled_users = unique_users_df.sample(
            fraction=fraction, 
            seed=seed, 
            with_replacement=False
        )
        
        # Filter full dataset using an inner join on the sampled users
        sampled_data = lazy_df.join(sampled_users.lazy(), on="user_id", how="inner").collect()
        
        output_path = processed_dir / f"sample_{label}.parquet"
        
        # Write clean, optimized sample partition
        print(f"[*] Saving {label} data partition with {len(sampled_data):,} rows...")
        sampled_data.write_parquet(output_path, compression="zstd")
        print(f"[SUCCESS] Saved: {output_path.name}")

    elapsed = time.time() - start_time
    print(f"\n[FINISHED] Sampling workflow finished successfully in {elapsed:.2f}s.")

def main():
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    create_deterministic_samples(processed_dir)

if __name__ == "__main__":
    main()