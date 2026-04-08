import os
import pandas as pd
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extraction import FeatureExtractor
from src.embedding_models import EmbeddingGenerator

# Global caching variables
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup - run on startup
    print("Loading AI Models and Data... (This takes a moment)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app_state["extractor"] = FeatureExtractor()
    app_state["embed_gen"] = EmbeddingGenerator(device=device)
    
    data_path = "data/framenet_subset.csv"
    if not os.path.exists(data_path):
        raise RuntimeError(f"Dataset not found at {data_path}. Please run `main.py` first.")
        
    df = pd.read_csv(data_path)
    # Cache 5000 samples for high quality predictions without melting RAM
    ref_df = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)
    app_state["ref_df"] = ref_df
    
    print("Pre-computing 'minilm' reference embeddings using structured local windows...")
    app_state["ref_embeddings"] = app_state["embed_gen"].get_embeddings("minilm", ref_df, context_type="window")
    
    print("Startup complete. Backend is ready!")
    yield
    # Cleanup (if needed)

app = FastAPI(title="Semantic Frame Induction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

class RequestData(BaseModel):
    sentence: str
    target_predicate: str
    embed_type: str = "minilm"
    num_similar: int = 5

@app.post("/predict")
async def predict_frame(data: RequestData):
    extractor = app_state["extractor"]
    embed_gen = app_state["embed_gen"]
    ref_df = app_state["ref_df"]
    
    # Check if we have precomputed embeddings for the requested type
    # For speed, we force minilm since it's cached, but we can fall back dynamically
    if data.embed_type == "minilm":
        ref_embeddings = app_state["ref_embeddings"]
    else:
        # Fallback (slow path)
        ref_embeddings = embed_gen.get_embeddings(data.embed_type, ref_df, context_type="window")
    
    # 1. Linguistic Feature Extraction
    doc, pred_token, args = extractor.process_sentence(data.sentence, data.target_predicate)
    
    if not pred_token:
        raise HTTPException(status_code=400, detail=f"Could not find the predicate '{data.target_predicate}' in the sentence.")
        
    extracted_args = [{"role": k, "text": v} for k, v in args.items()] if args else []
    
    # 2. Embedding Generation for User Sentence (Windowed for focused accuracy)
    user_df = pd.DataFrame([{"sentence": data.sentence, "target_predicate": data.target_predicate}])
    user_emb = embed_gen.get_embeddings(data.embed_type, user_df, context_type="window")
    
    # 3. Latent Semantic Search (Nearest Neighbors)
    sims = cosine_similarity(user_emb, ref_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:data.num_similar]
    
    results = []
    top_frames = []
    
    for idx in top_indices:
        sim_score = float(sims[idx])
        gold_frame = str(ref_df.iloc[idx]['frame_label'])
        top_frames.append(gold_frame)
        
        results.append({
            "similarity": f"{sim_score:.3f}",
            "sentence": str(ref_df.iloc[idx]['sentence']),
            "predicate": str(ref_df.iloc[idx]['target_predicate']),
            "gold_frame": gold_frame
        })
        
    # Voting Mechanism
    predicted_frame = max(set(top_frames), key=top_frames.count)
    
    return {
        "predicate_info": {
            "text": pred_token.text,
            "lemma": pred_token.lemma_
        },
        "arguments": extracted_args,
        "predicted_frame": predicted_frame,
        "similar_examples": results
    }

# Mount static frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(frontend_path, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
