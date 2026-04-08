import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extraction import FeatureExtractor
from src.embedding_models import EmbeddingGenerator

st.set_page_config(page_title="Semantic Frame Induction", layout="wide")

@st.cache_resource
def load_models():
    extractor = FeatureExtractor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_gen = EmbeddingGenerator(device=device)
    return extractor, embed_gen

# Temporarily removed cache_data to force reload
def load_dataset():
    data_path = "data/framenet_subset.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

def main():
    st.title("🧩 Semantic Frame Induction & Analysis")
    st.markdown("This application extracts predicate-argument graphs and induces semantic frames using unsupervised contextual embeddings.")
    
    with st.spinner("Loading NLP Models (SpaCy, Transformers)..."):
        extractor, embed_gen = load_models()
        
    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Please run `main.py` first to extract the FrameNet sample.")
        st.stop()
        
    st.sidebar.header("Configuration")
    embed_type = st.sidebar.selectbox("Contextual Embedding", ['minilm', 'bert', 'spacy', 'tfidf'])
    num_similar = st.sidebar.slider("Number of Similar Examples to retrieve", 1, 10, 5)

    st.header("1. Analyze a Custom Sentence")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_sentence = st.text_input("Enter a sentence:", value="The investor bought major shares in the tech company.")
    with col2:
        user_predicate = st.text_input("Target Predicate (Verb):", value="buy")

    if st.button("Extract & Induce Frame"):
        # 1. Feature Extraction
        st.subheader("Dependency & Predicate-Argument Parsing")
        doc, pred_token, args = extractor.process_sentence(user_sentence, user_predicate)
        
        if not pred_token:
            st.error(f"Could not find the predicate '{user_predicate}' in the sentence.")
        else:
            # Display Features
            st.write(f"**Target Predicate:** `{pred_token.text}` (lemma: `{pred_token.lemma_}`)")
            if args:
                st.table(pd.DataFrame([args]).T.rename(columns={0: 'Extracted Argument'}))
            else:
                st.write("No direct syntactic arguments found.")
                
            # 2. Embedding & Similarity Search
            st.subheader("Latent Semantic Clustering (Nearest Neighbors)")
            with st.spinner(f"Generating {embed_type.upper()} representation..."):
                # We need to compute embeddings for the reference dataset if not already in memory
                # In a real app, you'd cache the dataset embeddings. For demo, we compute on a small slice (or assume it's precomputed).
                # To keep it fast for the UI, let's take a 500-sentence sample from the ref dataset
                
                ref_df = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)
                ref_embeddings = embed_gen.get_embeddings(embed_type, ref_df, context_type="sentence")
                
                # Compute for user sentence
                user_df = pd.DataFrame([{"sentence": user_sentence, "target_predicate": user_predicate}])
                user_emb = embed_gen.get_embeddings(embed_type, user_df, context_type="sentence")
                
                # Cosine Similarity
                sims = cosine_similarity(user_emb, ref_embeddings)[0]
                top_indices = np.argsort(sims)[::-1][:num_similar]
                
                st.markdown(f"**Top {num_similar} closest induced frames from the dataset:**")
                results = []
                for idx in top_indices:
                    results.append({
                        "Similarity": f"{sims[idx]:.3f}",
                        "Sentence": ref_df.iloc[idx]['sentence'],
                        "Predicate": ref_df.iloc[idx]['target_predicate'],
                        "Gold Frame": ref_df.iloc[idx]['frame_label']
                    })
                
                st.dataframe(pd.DataFrame(results))
                
                # Induced Frame Prediction (Simple Voting)
                top_frames = [r['Gold Frame'] for r in results]
                predicted_frame = max(set(top_frames), key=top_frames.count)
                st.success(f"**Predicted Semantic Frame (Unsupervised k-NN):** {predicted_frame}")

if __name__ == "__main__":
    main()
