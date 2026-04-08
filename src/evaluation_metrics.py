from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, f1_score
from scipy.optimize import linear_sum_assignment
import numpy as np

def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity.
    """
    from sklearn import metrics
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def weighted_f1_hungarian(y_true, y_pred):
    from sklearn import metrics
    
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    mapping = {classes_pred[c]: classes_true[r] for r, c in zip(row_ind, col_ind)}
    y_pred_mapped = [mapping.get(p, p) for p in y_pred]
    
    return f1_score(y_true, y_pred_mapped, average='weighted')

def evaluate_clusters(y_true, y_pred):
    """
    Computes ARI, NMI, V-measure, Cluster Purity, and Hungarian F1.
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    purity = cluster_purity(y_true, y_pred)
    f1_hungarian = weighted_f1_hungarian(y_true, y_pred)
    
    # Enforce range [0.0, 1.0] for ARI as it can be negative for near-random assignments
    ari = max(0.0, float(ari))
    
    return {
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "V-measure": round(v_measure, 4),
        "Purity": round(purity, 4),
        "F1 (Hungarian)": round(f1_hungarian, 4)
    }
