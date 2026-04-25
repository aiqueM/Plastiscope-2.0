#!/usr/bin/env python3
"""
Cluster Chunk Embeddings

Performs unsupervised clustering on chunk embeddings.

Usage:
    python cluster_chunks.py --embeddings chunk_embeddings.npz --n-clusters 3
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def cluster_embeddings(embeddings, n_clusters, normalize=True):
    """Cluster embeddings using K-means"""
    print(f"\n{'='*60}")
    print(f"CLUSTERING CHUNK EMBEDDINGS")
    print(f"{'='*60}")
    print(f"Number of chunks: {len(embeddings):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Normalize embeddings: {normalize}")
    
    # Optionally normalize embeddings
    if normalize:
        print("\nNormalizing embeddings...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings
    
    # Perform K-means clustering
    print(f"\nRunning K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    print(f"✓ Clustering complete!")
    
    # Print cluster sizes
    print(f"\nCluster sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count:,} chunks ({count/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels, kmeans

def save_results(cluster_labels, output_file):
    """Save clustering results"""
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save cluster assignments
    np.savez_compressed(output_file, cluster_labels=cluster_labels)
    print(f"✓ Cluster labels saved to: {output_file}")
    
    # Save as CSV too for easier viewing
    csv_file = output_file.replace('.npz', '.csv')
    df = pd.DataFrame({
        'chunk_index': range(len(cluster_labels)),
        'cluster': cluster_labels
    })
    df.to_csv(csv_file, index=False)
    print(f"✓ Cluster labels saved to: {csv_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Cluster chunk embeddings using K-means",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Example:
  # Cluster chunks into 3 groups
  python cluster_chunks.py --embeddings chunk_embeddings.npz --n-clusters 3
  
  # Then evaluate against ground truth
  python evaluate_clustering.py --embeddings chunk_embeddings.npz --clusters clusters.npz
        """
    )
    
    parser.add_argument("--embeddings", required=True,
                       help="Input embeddings file (.npz)")
    parser.add_argument("--n-clusters", type=int, required=True,
                       help="Number of clusters")
    parser.add_argument("--output", default="clusters.npz",
                       help="Output file for cluster labels (default: clusters.npz)")
    parser.add_argument("--no-normalize", action='store_true',
                       help="Don't normalize embeddings before clustering")
    
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    data = np.load(args.embeddings)
    embeddings = data['embeddings']
    print(f"✓ Loaded {len(embeddings):,} chunk embeddings")
    
    # Cluster
    cluster_labels, kmeans = cluster_embeddings(
        embeddings, 
        args.n_clusters,
        normalize=not args.no_normalize
    )
    
    # Save results
    save_results(cluster_labels, args.output)
    
    print(f"\n{'='*60}")
    print("CLUSTERING COMPLETE!")
    print("="*60)
    print(f"\nNext step: Evaluate against ground truth")
    print(f"  python evaluate_clustering.py --clusters {args.output}")

if __name__ == "__main__":
    main()
