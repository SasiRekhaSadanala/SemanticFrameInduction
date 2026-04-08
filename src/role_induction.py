from sklearn.cluster import AgglomerativeClustering
import numpy as np

class RoleInducer:
    def __init__(self, n_roles=10):
        self.n_roles = n_roles
        
    def induce_roles(self, argument_embeddings):
        """
        Clusters argument embeddings to discover latent semantic roles.
        arguments: list of embedded representations of syntactic dependencies (e.g. subject embeddings)
        """
        if len(argument_embeddings) < self.n_roles:
            return np.arange(len(argument_embeddings))
            
        print(f"Clustering {len(argument_embeddings)} arguments into {self.n_roles} latent roles...")
        clustering = AgglomerativeClustering(n_clusters=self.n_roles)
        return clustering.fit_predict(argument_embeddings)
