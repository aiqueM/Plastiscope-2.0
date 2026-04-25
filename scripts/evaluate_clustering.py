#!/usr/bin/env python3
"""
Evaluate Clustering Results

Compares clustering results to ground truth (which sequence each chunk came from).
This reveals if DNABERT-S embeddings naturally separate by genome of origin.

Metrics:
- Adjusted Rand Index (ARI): 1.0 = perfect match, 0.0 = random
- Normalized Mutual Information (NMI): 1.0 = perfect, 0.0 = independent
- Purity: Fraction of correctly assigned chunks
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def calculate_purity(true_labels, cluster_labels):
    """Calculate clustering purity"""
    # For each cluster, find the most common true label
    contingency_matrix = pd.crosstab(cluster_labels, true_labels)
    
    # Sum the max of each cluster (most common true label per cluster)
    purity = contingency_matrix.max(axis=1).sum() / len(cluster_labels)
    return purity

def evaluate_clustering(true_labels, cluster_labels):
    """Evaluate clustering against ground truth"""
    print(f"\n{'='*60}")
    print(f"CLUSTERING EVALUATION")
    print(f"{'='*60}")
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    purity = calculate_purity(true_labels, cluster_labels)
    
    print(f"\nMetrics:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"  Purity: {purity:.4f}")
    
    print(f"\nInterpretation:")
    if ari > 0.8:
        print("  🎉 EXCELLENT: Clusters closely match genome boundaries!")
    elif ari > 0.5:
        print("  👍 GOOD: Clusters show strong genome structure")
    elif ari > 0.3:
        print("  🤔 MODERATE: Some genome structure detected")
    else:
        print("  ❌ POOR: Clusters don't match genome boundaries")
    
    return ari, nmi, purity

def create_confusion_matrix(true_labels, cluster_labels):
    """Create and display confusion matrix"""
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*60}")
    print("(Rows = Clusters, Columns = True Genomes)\n")
    
    contingency = pd.crosstab(
        cluster_labels, 
        true_labels,
        rownames=['Cluster'],
        colnames=['Genome']
    )
    
    print(contingency)
    print()
    
    # Show what each cluster represents
    print("Cluster composition:")
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        chunks_in_cluster = sum(mask)
        true_in_cluster = true_labels[mask]
        most_common = Counter(true_in_cluster).most_common(1)[0]
        
        print(f"  Cluster {cluster_id}: {chunks_in_cluster} chunks")
        print(f"    → {most_common[1]}/{chunks_in_cluster} ({most_common[1]/chunks_in_cluster*100:.1f}%) from {most_common[0]}")
    
    return contingency

def analyze_per_genome(true_labels, cluster_labels, metadata_df):
    """Analyze clustering quality per genome"""
    print(f"\n{'='*60}")
    print(f"PER-GENOME ANALYSIS")
    print(f"{'='*60}\n")
    
    for genome_name in sorted(metadata_df['sequence_name'].unique()):
        genome_mask = metadata_df['sequence_name'] == genome_name
        genome_chunks = sum(genome_mask)
        genome_clusters = cluster_labels[genome_mask]
        
        # Find dominant cluster for this genome
        cluster_counts = Counter(genome_clusters)
        dominant_cluster, dominant_count = cluster_counts.most_common(1)[0]
        purity_score = dominant_count / genome_chunks
        
        print(f"{genome_name}:")
        print(f"  Total chunks: {genome_chunks}")
        print(f"  Dominant cluster: {dominant_cluster} ({dominant_count} chunks, {purity_score*100:.1f}%)")
        print(f"  Cluster distribution: {dict(cluster_counts)}")
        print()

def save_detailed_results(true_labels, cluster_labels, metadata_df, output_file):
    """Save detailed results with both true and predicted labels"""
    print(f"\n{'='*60}")
    print(f"SAVING DETAILED RESULTS")
    print(f"{'='*60}")
    
    # Add cluster assignments to metadata
    results_df = metadata_df.copy()
    results_df['cluster'] = cluster_labels
    results_df['true_genome'] = true_labels
    
    results_file = output_file.replace('.txt', '_detailed.csv')
    results_df.to_csv(results_file, index=False)
    print(f"✓ Detailed results saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clustering against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python evaluate_clustering.py \\
    --clusters clusters.npz \\
    --metadata chunk_embeddings_metadata.csv \\
    --output evaluation_results.txt
        """
    )
    
    parser.add_argument("--clusters", required=True,
                       help="Cluster labels file (.npz or .csv)")
    parser.add_argument("--metadata", required=True,
                       help="Chunk metadata file (contains ground truth)")
    parser.add_argument("--output", default="evaluation_results.txt",
                       help="Output file for results (default: evaluation_results.txt)")
    
    args = parser.parse_args()
    
    # Load cluster labels
    print(f"Loading cluster labels from {args.clusters}...")
    if args.clusters.endswith('.npz'):
        cluster_data = np.load(args.clusters)
        cluster_labels = cluster_data['cluster_labels']
    else:
        cluster_df = pd.read_csv(args.clusters)
        cluster_labels = cluster_df['cluster'].values
    print(f"✓ Loaded {len(cluster_labels):,} cluster assignments")
    
    # Load metadata (ground truth)
    print(f"\nLoading metadata from {args.metadata}...")
    metadata_df = pd.read_csv(args.metadata)
    true_labels = metadata_df['sequence_name'].values
    print(f"✓ Loaded {len(true_labels):,} ground truth labels")
    
    # Check alignment
    if len(cluster_labels) != len(true_labels):
        print(f"\n❌ ERROR: Mismatch in number of chunks!")
        print(f"  Clusters: {len(cluster_labels)}")
        print(f"  Metadata: {len(true_labels)}")
        return
    
    # Evaluate
    ari, nmi, purity = evaluate_clustering(true_labels, cluster_labels)
    
    # Create confusion matrix
    contingency = create_confusion_matrix(true_labels, cluster_labels)
    
    # Per-genome analysis
    analyze_per_genome(true_labels, cluster_labels, metadata_df)
    
    # Save detailed results
    save_detailed_results(true_labels, cluster_labels, metadata_df, args.output)
    
    # Save summary report
    with open(args.output, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLUSTERING EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("METRICS:\n")
        f.write(f"  Adjusted Rand Index (ARI): {ari:.4f}\n")
        f.write(f"  Normalized Mutual Information (NMI): {nmi:.4f}\n")
        f.write(f"  Purity: {purity:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(contingency.to_string())
        f.write("\n\n")
        
        f.write("INTERPRETATION:\n")
        if ari > 0.8:
            f.write("  EXCELLENT: DNABert-S embeddings strongly separate genomes!\n")
        elif ari > 0.5:
            f.write("  GOOD: DNABert-S embeddings show clear genome structure\n")
        elif ari > 0.3:
            f.write("  MODERATE: Some genome structure is captured\n")
        else:
            f.write("  POOR: Embeddings don't separate well by genome\n")
    
    print(f"✓ Summary report saved to: {args.output}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nARI Score: {ari:.4f}")
    if ari > 0.5:
        print("✓ DNABert-S embeddings capture genome-specific patterns!")
    else:
        print("⚠ Embeddings may not strongly differentiate genomes")

if __name__ == "__main__":
    main()
