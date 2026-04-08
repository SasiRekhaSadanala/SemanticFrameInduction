import numpy as np
import networkx as nx
import community as community_louvain # python-louvain package
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class FrameInducer:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        
    def cluster_kmeans(self, embeddings):
        print(f"Running K-Means (k={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)
        
    def cluster_agglomerative(self, embeddings):
        print(f"Running Agglomerative Clustering (n={self.n_clusters})...")
        agg = AgglomerativeClustering(n_clusters=self.n_clusters)
        return agg.fit_predict(embeddings)
        
    def cluster_spectral(self, embeddings):
        print(f"Running Spectral Clustering (n={self.n_clusters})...")
        # Compute affinity matrix based on cosine similarity
        affinity_matrix = cosine_similarity(embeddings)
        # Shift to non-negative affinities
        affinity_matrix = (affinity_matrix + 1) / 2
        
        spectral = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=42)
        return spectral.fit_predict(affinity_matrix)
        
    def cluster_dbscan(self, embeddings, eps=0.5, min_samples=5):
        print(f"Running DBSCAN (eps={eps})...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        return dbscan.fit_predict(embeddings)

class GraphFrameInducer:
    def __init__(self):
        pass
        
    def cluster_louvain(self, graph):
        """
        Uses the Louvain method for community detection on the NetworkX graph.
        """
        print("Running Louvain Graph Clustering...")
        # Louvain works on undirected graphs
        undirected_G = graph.to_undirected()
        
        # We need edge weights
        for u, v, d in undirected_G.edges(data=True):
            if 'weight' not in d:
                d['weight'] = 1.0
                
        partition = community_louvain.best_partition(undirected_G, weight='weight', resolution=1.0)
        return partition
        
    def cluster_label_propagation(self, graph):
        """
        Uses asynchronous label propagation from NetworkX.
        """
        print("Running Label Propagation Graph Clustering...")
        undirected_G = graph.to_undirected()
        communities = list(nx.algorithms.community.asyn_lpa_communities(undirected_G, weight='weight'))
        
        # Convert list of sets to node->community dictionary
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition
        
    def extract_labels_for_sentences(self, partition, documents, target_predicates, extractor):
        """
        Given the graph partitioning and original sentences, 
        assign the community label of the predicate node to the sentence.
        """
        labels = []
        for doc_text, target in zip(documents, target_predicates):
            pred_node = f"PRED_{target}"
            if pred_node in partition:
                labels.append(partition[pred_node])
            else:
                labels.append(-1) # Outlier / unknown
        return np.array(labels)
