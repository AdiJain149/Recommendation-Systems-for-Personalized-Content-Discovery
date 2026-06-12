import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Import our production engine class securely from our package workspace
from src.models.engine import RecommendationEngine

# Create a global placeholder container to retain our engine state across routes safely
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown tasks.
    Loads the ML recommendation engine ONCE into memory on app launch.
    """
    print("[*] API Lifecycle Boot Sequence Initiating...")
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "outputs" / "svd_model.pkl"
    data_path = base_dir / "data" / "processed" / "sample_10pct.parquet"
    
    try:
        # Instantiate the engine and assign it to the globally accessible app state container
        ml_models["engine"] = RecommendationEngine(model_path, data_path)
        print("[SUCCESS] ML Infrastructure embedded into application lifespan state.")
        yield
    except Exception as e:
        print(f"[CRITICAL] Failed to load ML infrastructure during startup: {e}")
        raise e
    finally:
        # Clear engine memory allocations cleanly during shutdown phase
        ml_models.clear()
        print("[*] API Lifecycle Shutdown complete.")

# Initialize the main FastAPI application instance with custom metadata and our lifespan hooks
app = FastAPI(
    title="Netflix Recommendation Engine API",
    version="1.0.0",
    lifespan=lifespan
)

# Define Pydantic Schemas to enforce strict contract standards on incoming and outgoing JSON
class RecommendationItem(BaseModel):
    movie_id: int
    predicted_rating: float

class RecommendationResponse(BaseModel):
    user_id: int
    count: int
    recommendations: List[RecommendationItem]


@app.get("/", tags=["Health"])
async def root():
    """Simple health-check endpoint to verify server status."""
    return {"status": "ONLINE", "service": "netflix-recsys-api"}


@app.get(
    "/api/v1/recommend/{user_id}", 
    response_model=RecommendationResponse, 
    tags=["Recommendations"]
)
async def get_recommendations(
    user_id: int, 
    k: int = Query(default=10, ge=1, le=100, description="Number of recommendations to retrieve")
):
    """
    Retrieves the top K tailored recommendations for a targeted user profile.
    Automatically flags structural validation errors via native constraints.
    """
    engine: RecommendationEngine = ml_models.get("engine")
    if not engine:
        raise HTTPException(status_code=500, detail="Recommendation engine is uninitialized.")
    
    # Check if user profile exists in historical data maps
    # If not, the engine handles them gracefully, but we log a trace for visibility
    is_cold_start = user_id not in engine.user_history
    
    try:
        # Execute rapid inference lookup loop
        recs = engine.get_top_k_recommendations(user_id=user_id, k=k)
        
        # Structure payload back to match validation response contract
        return {
            "user_id": user_id,
            "count": len(recs),
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine execution exception: {str(e)}")