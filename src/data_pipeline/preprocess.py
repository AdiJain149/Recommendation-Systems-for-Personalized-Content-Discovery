import os
import sys
import time
from pathlib import Path
import polars as pl

def parse_raw_netflix_file(file_path: Path, output_dir: Path) -> Path:
    """
    Parses a single Netflix raw text file line-by-line to protect system memory.
    Converts custom formatting into a structured, typed Parquet file.
    """
    print(f"[*] Starting processing on: {file_path.name}")
    start_time = time.time()
    
    # Accumulation buffers
    user_ids = []
    ratings = []
    dates = []
    movie_ids = []
    
    current_movie_id = None
    row_count = 0
    chunk_idx = 1
    
    output_parquet_path = output_dir / f"{file_path.stem}.parquet"

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Detect Movie ID separator (e.g., "1:")
            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                # Parse standard CSV row: UserID,Rating,Date
                user_id, rating, date_str = line.split(",")
                
                user_ids.append(int(user_id))
                ratings.append(int(rating))
                dates.append(date_str)
                movie_ids.append(current_movie_id)
                row_count += 1

    print(f"[+] Finished parsing text in memory. Converting {row_count:,} rows to Polars DataFrame...")
    
    # Enforce schemas and optimal memory data types strictly
    df = pl.DataFrame(
        {
            "user_id": user_ids,
            "movie_id": movie_ids,
            "rating": ratings,
            "date": dates
        },
        schema={
            "user_id": pl.Int32,
            "movie_id": pl.Int32,
            "rating": pl.Int8,     # Ratings are only 1-5, Int8 saves massive RAM
            "date": pl.String
        }
    )
    
    # Cast strings to concrete physical Date object optimized by Polars
    print("[*] Optimizing column data types...")
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    
    # Write to optimized disk storage format
    print(f"[*] Saving to disk: {output_parquet_path}")
    df.write_parquet(output_parquet_path, compression="zstd")
    
    elapsed_time = time.time() - start_time
    print(f"[SUCCESS] Processed {file_path.name} in {elapsed_time:.2f}s -> {output_parquet_path.name}\n")
    return output_parquet_path

def run_pipeline():
    # Setup paths relative to execution root folder
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    raw_files = sorted(list(raw_dir.glob("combined_data_*.txt")))
    
    if not raw_files:
        print(f"[ERROR] No raw files found in '{raw_dir}'.")
        print("Please place 'combined_data_1.txt' inside that folder to proceed.")
        sys.exit(1)
        
    print(f"[FOUND] Found {len(raw_files)} raw data text files to process.")
    for file_path in raw_files:
        parse_raw_netflix_file(file_path, processed_dir)

if __name__ == "__main__":
    run_pipeline()