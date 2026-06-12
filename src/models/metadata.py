from pathlib import Path

def load_movie_titles(file_path: Path) -> dict:
    """
    Parses Netflix style movie titles metadata file dynamically handling comma variances.
    Returns a dictionary mapping int(movie_id) -> "Movie Title (Year)"
    """
    movie_map = {}
    if not file_path.exists():
        print(f"[!] Warning: Metadata file not found at {file_path}. Using fallback IDs.")
        return movie_map
        
    print(f"[*] Parsing movie titles metadata from: {file_path.name}")
    with open(file_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Netflix format uses: movie_id,year,title
            parts = line.split(",", 2)
            if len(parts) >= 3:
                try:
                    mid = int(parts[0])
                    year = parts[1]
                    title = parts[2]
                    movie_map[mid] = f"{title} ({year})"
                except ValueError:
                    continue
            elif len(parts) == 2:
                try:
                    mid = int(parts[0])
                    movie_map[mid] = parts[1]
                except ValueError:
                    continue
    return movie_map