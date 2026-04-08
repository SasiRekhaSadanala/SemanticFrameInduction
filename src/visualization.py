import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import os
import numpy as np

def plot_embeddings(embeddings, labels, title, filename, method='pca'):
    print(f"Generating {method.upper()} visualization: {title}")
    plt.figure(figsize=(10, 8))
    
    if len(embeddings) < 2:
        print("Not enough samples to plot.")
        return
        
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        
    reduced = reducer.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'Dim1': reduced[:, 0],
        'Dim2': reduced[:, 1],
        'Cluster': labels
    })
    
    # Handle potentially large number of unique labels
    unique_labels = len(np.unique(labels))
    palette = 'tab10' if unique_labels <= 10 else 'viridis'
    
    sns.scatterplot(data=df, x='Dim1', y='Dim2', hue='Cluster', palette=palette, legend=False, s=30, alpha=0.7)
    plt.title(title)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_confusion_heatmap(y_true, y_pred, filename):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    # Simplify if the CM is huge
    if cm.shape[0] > 50 or cm.shape[1] > 50:
         print("Confusion matrix too large to plot clearly. Saving raw matrix to csv.")
         pd.DataFrame(cm).to_csv(filename.replace('.png', '.csv'))
         return
         
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Frame')
    plt.title('True Frame vs Predicted Cluster')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
