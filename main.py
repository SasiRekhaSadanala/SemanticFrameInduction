import argparse
import time
import os
import pandas as pd
import numpy as np

from src.load_framenet import load_framenet_data
from src.feature_extraction import FeatureExtractor
from src.embedding_models import EmbeddingGenerator
from src.clustering_methods import FrameInducer, GraphFrameInducer
from src.evaluation_metrics import evaluate_clusters
from src.visualization import plot_embeddings, plot_confusion_heatmap

def run_experiment():
    parser = argparse.ArgumentParser(description="Semantic Frame Induction Experiment Runner")
    parser.add_argument("--full", action="store_true", help="Run on full FrameNet dataset")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== STEP 1: Data Preparation ===")
    df = load_framenet_data(full=args.full, max_sentences=2000 if not args.full else 999999)
    print(f"Loaded {len(df)} sentences.")
    
    # Restrict to top 15 frames for clearer visualization if not full
    if not args.full:
        top_frames = df['frame_label'].value_counts().nlargest(15).index
        df = df[df['frame_label'].isin(top_frames)]
        print(f"Subsampled to {len(df)} sentences across {len(top_frames)} frames for development.")

    true_labels = df['frame_label'].tolist()
    # Convert string labels to integer IDs for evaluation metrics
    unique_frames = list(set(true_labels))
    label_to_id = {label: i for i, label in enumerate(unique_frames)}
    y_true = np.array([label_to_id[label] for label in true_labels])
    n_clusters = len(unique_frames)

    print(f"\nGround Truth: {n_clusters} unique semantic frames.")

    print("\n=== STEP 2: Linguistic Features & Graph Prep ===")
    extractor = FeatureExtractor()
    graph_inducer = GraphFrameInducer()
    print("Building Predicate-Argument Graph...")
    graph = extractor.build_predicate_argument_graph(df['sentence'].tolist(), df['target_predicate'].tolist())
    
    # Run Graph-based clustering (Louvain)
    print("\nRunning Graph-based Clustering (Louvain)...")
    start_time = time.time()
    partition = graph_inducer.cluster_louvain(graph)
    y_pred_graph = graph_inducer.extract_labels_for_sentences(partition, df['sentence'].tolist(), df['target_predicate'].tolist(), extractor)
    graph_time = time.time() - start_time
    
    # Evaluate Graph Clustering (Only where predictions were made)
    valid_idx = y_pred_graph != -1
    if sum(valid_idx) > 0:
        graph_metrics = evaluate_clusters(y_true[valid_idx], y_pred_graph[valid_idx])
    else:
        graph_metrics = {"ARI": 0, "NMI": 0, "V-measure": 0, "Purity": 0, "F1 (Hungarian)": 0}

    results = []
    results.append({
        "Representation": "Predicate-Argument Graph",
        "Clustering": "Louvain",
        "Runtime (s)": round(graph_time, 2),
        **graph_metrics
    })

    print("\n=== STEP 3: Contextual Representation & Clustering ===")
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embed_gen = EmbeddingGenerator(device=device)

    representations = ['tfidf', 'spacy', 'minilm', 'bert']
    clustering_methods = ['kmeans', 'agglomerative', 'spectral']
    
    inducer = FrameInducer(n_clusters=n_clusters)

    for rep in representations:
        print(f"\n--- Representation: {rep.upper()} ---")
        start_embed = time.time()
        # Generate sentence-level embeddings
        embeddings = embed_gen.get_embeddings(rep, df, context_type="sentence")
        embed_time = time.time() - start_embed
        print(f"Embedding shape: {embeddings.shape} (Time: {embed_time:.2f}s)")
        
        for method in clustering_methods:
            start_clust = time.time()
            if method == 'kmeans':
                y_pred = inducer.cluster_kmeans(embeddings)
            elif method == 'agglomerative':
                y_pred = inducer.cluster_agglomerative(embeddings)
            elif method == 'spectral':
                y_pred = inducer.cluster_spectral(embeddings)
            
            clust_time = time.time() - start_clust
            metrics = evaluate_clusters(y_true, y_pred)
            total_time = embed_time + clust_time
            
            # Save visual results for the best / select runs (e.g., MiniLM + Spectral)
            if rep == 'minilm' and method == 'spectral':
                plot_embeddings(embeddings, y_pred, f"MiniLM + Spectral Clusters", f"{args.out_dir}/minilm_spectral_tsne.png", method='tsne')
            
            results.append({
                "Representation": rep.upper(),
                "Clustering": method.capitalize(),
                "Runtime (s)": round(total_time, 2),
                **metrics
            })
            print(f"[{method.capitalize()}] ARI: {metrics['ARI']} | NMI: {metrics['NMI']}")

    print("\n=== STEP 4: Evaluation Results ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    results_file = os.path.join(args.out_dir, "comparison_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Generate True label embeddings for reference
    best_embeds = embed_gen.get_embeddings('minilm', df, context_type="sentence")
    plot_embeddings(best_embeds, true_labels, "True Frame Clusters (MiniLM)", f"{args.out_dir}/true_frames_tsne.png", method='tsne')

if __name__ == "__main__":
    run_experiment()
