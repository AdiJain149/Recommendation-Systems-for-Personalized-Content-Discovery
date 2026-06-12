import sys
from pathlib import Path

# Force the project root directory into the path so imports work perfectly
base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.models.engine import RecommendationEngine

def run_demo():
    print("=" * 60)
    print("🎬 NETFLIX PERSONALIZED CONTENT DISCOVERY DEMO 🎬")
    print("=" * 60)
    
    # Define absolute resource targets
    svd_path = base_dir / "outputs" / "svd_model.pkl"
    als_path = base_dir / "outputs" / "als_model.pkl"
    data_path = base_dir / "data" / "processed" / "sample_10pct.parquet"
    
    # 1. Select the Engine Architecture
    print("\nSelect Recommendation Model Engine:")
    print(" [1] Surprise SVD (Explicit Rating Predictor)")
    print(" [2] Implicit ALS (Collaborative Filtering Matrix Factorization)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    if choice == "1":
        model_target = svd_path
        print("\n[*] Loading SVD Engine Components...")
    elif choice == "2":
        model_target = als_path
        print("\n[*] Loading Implicit ALS Engine Components...")
    else:
        print("[!] Invalid choice. Defaulting to Implicit ALS.")
        model_target = als_path

    # Initialize our unified engine wrapper
    try:
        engine = RecommendationEngine(model_path=model_target, data_path=data_path)
    except Exception as e:
        print(f"\n[!] Initialization Error: Lookups failed. Did you run train_and_track first?")
        print(f"Error details: {e}")
        return

    # 2. Interactive User Query Loop
    print("\n[+] System online. Ready for personalized discovery queries.")
    while True:
        print("-" * 60)
        user_input = input("Enter a User ID to query (or type 'exit' to quit): ").strip()
        
        if user_input.lower() == 'exit':
            print("\nShutting down recommendation core framework. Goodbye!")
            break
            
        if not user_input.isdigit():
            print("[!] Please enter a valid numerical User ID.")
            continue
            
        uid = int(user_input)
        
        # Pull top 5 recommendations
        print(f"[*] Calculating top recommendations for User #{uid}...")
        recs = engine.get_top_k_recommendations(user_id=uid, k=5)
        
        if not recs:
            print(f"[!] Cold Start Warning: No historical profile data mapped for User #{uid}.")
            continue
            
        print(f"\n✨ TOP 5 PERSONALIZED RECOMMENDATIONS FOR USER {uid} ✨")
        print(f"{'Position':<10}{'Movie Title':<40}{'Match Score':<12}")
        print("-" * 62)
        for idx, item in enumerate(recs, start=1):
            print(f" #{idx:<8}{item['title']:<40}{item['estimated_score']:<12}")
        print()

if __name__ == "__main__":
    run_demo()