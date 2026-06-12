# src/models/metrics.py
import numpy as np
from typing import List, Set

def calculate_ranking_metrics(predicted_top_k: List[int], ground_truth: Set[int], k: int) -> dict:
    """
    Computes Information Retrieval ranking metrics (Precision@K, Recall@K, MAP@K, NDCG@K)
    for a single user profile.
    """
    # Truncate recommendations to target K threshold
    predicted = predicted_top_k[:k]
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "map": 0.0, "ndcg": 0.0}
    
    # Calculate binary relevance array (1 if recommended item is in ground truth, else 0)
    relevance = [1 if item in ground_truth else 0 for item in predicted]
    hits = sum(relevance)
    
    # 1. Precision@K = Hits / K
    precision = hits / k
    
    # 2. Recall@K = Hits / Total Actual Interactions
    recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    # 3. Mean Average Precision @ K (MAP@K)
    precisions_at_hits = []
    running_hits = 0
    for i, rel in enumerate(relevance):
        if rel == 1:
            running_hits += 1
            precisions_at_hits.append(running_hits / (i + 1))
    ap = np.mean(precisions_at_hits) if precisions_at_hits else 0.0
    
    # 4. Normalized Discounted Cumulative Gain (NDCG@K)
    # DCG = sum(rel_i / log2(i + 1 + 1))
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
    # IDCG is the perfect ranking order (all hits packed at the front)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
    
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "map": ap,
        "ndcg": ndcg
    }