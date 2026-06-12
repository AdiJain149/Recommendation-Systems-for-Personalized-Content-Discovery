# Recommendation Systems for Personalized Content Discovery

A production-oriented recommendation system built using the Netflix Prize Dataset. This project implements collaborative filtering and matrix factorization techniques to generate personalized movie recommendations while demonstrating a complete machine learning workflow from data preprocessing to deployment.

---

## Overview

With the exponential growth of digital content, users often face difficulty discovering relevant content from large catalogs. This project addresses that challenge by developing an end-to-end recommendation engine capable of learning user preferences from historical interactions and generating personalized movie recommendations.

The system integrates data engineering, machine learning, experiment tracking, API deployment, and interactive visualization into a unified pipeline.

---

## Features

- Personalized movie recommendations
- Collaborative filtering approach
- Matrix factorization using SVD and ALS
- Efficient preprocessing using Polars and PyArrow
- Experiment tracking with MLflow
- REST API deployment using FastAPI
- Interactive dashboard using Streamlit
- Modular and scalable architecture

---


## Tech Stack
- **Data Engineering:** Polars
- **Modeling:** SVD (Surprise), ALS (Implicit)
- **Deployment:** FastAPI, Streamlit, Docker
- **MLOps:** MLflow

---

## Dataset

The project utilizes the **Netflix Prize Dataset**, a benchmark dataset widely used in recommendation system research.

### Dataset Statistics

| Attribute | Value |
|------------|---------|
| Users | ~480,000 |
| Movies | ~17,700 |
| Ratings | 100+ Million |
| Rating Scale | 1–5 |

The raw rating files are transformed into optimized Parquet files to enable efficient storage and faster processing.

---

## System Architecture

```text
Raw Netflix Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Parquet Storage
        │
        ▼
Sampling Pipeline
        │
        ▼
SVD / ALS Training
        │
        ▼
Model Evaluation
        │
        ▼
MLflow Tracking
        │
        ▼
FastAPI Backend
        │
        ▼
Streamlit Dashboard
```

---

## Models Implemented

### Singular Value Decomposition (SVD)

- Implemented using the Surprise library
- Learns latent user and movie factors
- Predicts ratings for unseen user-movie pairs
- Optimized for explicit feedback recommendation

### Alternating Least Squares (ALS)

- Implemented using the Implicit library
- Efficient for sparse matrices
- Suitable for large-scale recommendation tasks
- Learns latent user and item embeddings

---

## Model Performance

### SVD Results

| Metric | Score |
|----------|---------|
| RMSE | 0.8483 |
| MAE | 0.6566 |

The achieved performance demonstrates the effectiveness of matrix factorization techniques on sparse user-item interaction data.

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/AdiJain149/Recommendation-Systems-for-Personalized-Content-Discovery.git

cd Recommendation-Systems-for-Personalized-Content-Discovery
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Start FastAPI Backend

```bash
uvicorn app:app --reload
```

Backend available at:

```text
http://localhost:8000
```

---

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

Dashboard available at:

```text
http://localhost:8501
```

---

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "ONLINE",
  "service": "netflix-recsys-api"
}
```

---

### Get Recommendations

```http
GET /recommend/{user_id}
```

Returns Top-K personalized movie recommendations for the specified user.

---

## Experiment Tracking

MLflow is used for:

- Tracking model parameters
- Logging evaluation metrics
- Comparing experiments
- Managing trained model artifacts

Tracked Metrics:

- RMSE
- MAE
- Training Time
- Matrix Density

---

## Project Structure

```text
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── models/
│   ├── svd/
│   └── als/
│
├── api/
│   └── FastAPI Backend
│
├── dashboard/
│   └── Streamlit App
│
├── mlflow/
│   └── Experiment Logs
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## Future Improvements

- Movie title mapping support
- Precision@K and Recall@K evaluation
- NDCG and MAP metrics
- Cold-start recommendation handling
- Docker containerization
- Cloud deployment
- Hybrid recommendation models
- Explainable recommendations

---

## Learning Outcomes

This project demonstrates practical experience in:

- Recommendation Systems
- Machine Learning Engineering
- Data Processing Pipelines
- Matrix Factorization
- REST API Development
- Experiment Tracking
- Dashboard Development
- End-to-End ML Deployment

---

## Authors

**Rishabh Gupta**  
**Adi Jain**

Department of Computer Science and Engineering  
Indian Institute of Technology Roorkee

