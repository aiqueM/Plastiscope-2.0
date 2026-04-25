#!/usr/bin/env python3
"""
Clustering evaluation script for accession-labeled FASTA headers.

Works with headers like:
  CP115456.1_m64060_201027_182301/140839279/25621_34742
  CP117193.1_m64060_201027_182301/78971409/34337_47701
  NZ_CP117996.1_m64060_201027_182301/102368678/17814_27560

The accession prefix identifies the species - NO separate mapping files needed!
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Evaluate clustering results against species labels")
parser.add_argument("--clusters", required=True, help="Path to cluster NPZ file")
parser.add_argument("--metadata", required=True, help="Path to metadata CSV file")
parser.add_argument("--output-confusion", default="confusion_matrix_results.csv", help="Output confusion matrix CSV")
parser.add_argument("--output-fig", default="confusion_matrix_visualization.png", help="Output figure PNG")

args = parser.parse_args()

# -----------------------------
# CONFIG
# -----------------------------
CLUSTERS_NPZ = args.clusters
METADATA_CSV = args.metadata
OUT_CONFUSION_CSV = args.output_confusion
OUT_FIG = args.output_fig

# -----------------------------
# Load cluster assignments
# -----------------------------
print("Loading cluster labels...")
clusters = np.load(CLUSTERS_NPZ)

# Expected key: 'cluster_labels'
if "cluster_labels" not in clusters.files:
    raise KeyError(
        f"'cluster_labels' not found in {CLUSTERS_NPZ}. उपलब्ध keys: {clusters.files}"
    )

cluster_labels = np.asarray(clusters["cluster_labels"]).ravel().astype(int)
print(f"✓ Loaded {len(cluster_labels)} cluster assignments")


# -----------------------------
# Load metadata
# -----------------------------
print("\nLoading metadata...")
metadata = pd.read_csv(METADATA_CSV)
print(f"✓ Loaded {len(metadata)} metadata entries")

if "sequence_name" not in metadata.columns:
    raise KeyError(
        f"'sequence_name' column not found in {METADATA_CSV}. Found columns: {list(metadata.columns)}"
    )

if len(metadata) != len(cluster_labels):
    raise ValueError(
        f"Length mismatch: metadata has {len(metadata)} rows but cluster_labels has {len(cluster_labels)} labels."
    )

import re

# -----------------------------
# Extract species from accession-labeled headers
# -----------------------------
def extract_species(seq_name: str) -> str:
    """
    Extract species accession from labeled header.

    Examples:
      CP115456.1_m64060_201027_182301/... → CP115456.1
      CP117193.1_m64060_201027_182301/... → CP117193.1
      NZ_CP117996.1_m64060_201027_182301/... → NZ_CP117996.1

    Default behavior: return everything before the first underscore.
    """
    m = re.match(r'^(NZ_CP\d+\.\d+|CP\d+\.\d+)', str(seq_name))
    return m.group(1) if m else "Unknown"


print("\nExtracting species labels from headers...")
metadata["species"] = metadata["sequence_name"].apply(extract_species)

print("\nSample metadata with extracted species:")
print(metadata[["sequence_name", "species"]].head(10))

# Species distribution
species_counts = metadata["species"].value_counts()
print("\nSpecies distribution:")
print(species_counts)

# Ground truth labels (strings)
true_labels = metadata["species"].astype(str).values


# -----------------------------
# Metrics (ARI, NMI, Purity)
# -----------------------------
print("\n" + "=" * 70)
print("CLUSTERING EVALUATION")
print("=" * 70)

ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)


def purity_score(y_true, y_pred) -> float:
    contingency = pd.crosstab(y_pred, y_true)
    return np.sum(np.amax(contingency.values, axis=1)) / np.sum(contingency.values)


purity = purity_score(true_labels, cluster_labels)

print(f"\nMetrics:")
print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
print(f"  Purity: {purity:.4f}")

if ari > 0.8:
    print(f"\n  ✓ EXCELLENT: Clusters match species boundaries very well")
elif ari > 0.5:
    print(f"\n  ⚠ MODERATE: Clusters partially match species")
else:
    print(f"\n  ✗ POOR: Clusters don't match species boundaries")


# -----------------------------
# Species x Cluster table (CROSSTAB)
# -----------------------------
print("\n" + "=" * 70)
print("SPECIES x CLUSTER TABLE (CROSSTAB)")
print("=" * 70)

# Desired species order (keep the same as your original script)
species_order = ["CP115456.1", "CP117193.1", "NZ_CP117996.1"]

# Build cross-tab: rows=species, cols=cluster IDs
cm_df = pd.crosstab(
    metadata["species"],
    cluster_labels,
    rownames=["True Species"],
    colnames=["Cluster"],
    dropna=False,
)

# Reorder rows and cols (and ensure ints)
cm_df = cm_df.reindex(species_order).fillna(0).astype(int)
cm_df = cm_df.reindex(sorted(cm_df.columns), axis=1)

print("\n(Rows = True Species | Columns = Cluster ID)")
print(cm_df)

# Add totals
cm_df_with_total = cm_df.copy()
cm_df_with_total["Total"] = cm_df_with_total.sum(axis=1)

totals = cm_df_with_total.sum(axis=0)
totals.name = "Total"
cm_with_totals = pd.concat([cm_df_with_total, pd.DataFrame([totals])])

print("\nWith totals:")
print(cm_with_totals)


# -----------------------------
# Species distribution analysis
# -----------------------------
print("\n" + "=" * 70)
print("SPECIES DISTRIBUTION ANALYSIS")
print("=" * 70)

species_names = {
    "CP115456.1": "Sphingobium yanoikuyae",
    "CP117193.1": "Acidovorax temperans",
    "NZ_CP117996.1": "Microbacterium sp.",
}

for sp in species_order:
    if sp not in cm_df_with_total.index:
        print(f"\n{sp} (not found in metadata)")
        continue

    total = cm_df_with_total.loc[sp, "Total"]
    print(f"\n{sp} - {species_names.get(sp, 'Unknown')} ({total} reads):")

    for cluster_id in cm_df.columns:
        count = cm_df.loc[sp, cluster_id]
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  Cluster {cluster_id}: {count:4d} ({pct:5.1f}%) {bar}")

    # Dominant cluster
    if total > 0:
        dominant_cluster_id = cm_df.loc[sp].idxmax()
        dominant_pct = (cm_df.loc[sp, dominant_cluster_id] / total * 100)
        print(f"  → Dominant: Cluster {dominant_cluster_id} ({dominant_pct:.1f}%)")


# -----------------------------
# Cluster purity analysis
# -----------------------------
print("\n" + "=" * 70)
print("CLUSTER PURITY ANALYSIS")
print("=" * 70)

for cluster_id in cm_df.columns:
    total = cm_df[cluster_id].sum()
    print(f"\nCluster {cluster_id} ({total} reads):")

    for sp in species_order:
        count = cm_df.loc[sp, cluster_id] if sp in cm_df.index else 0
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {sp:15s} ({species_names.get(sp, 'Unknown'):25s}): {count:4d} ({pct:5.1f}%) {bar}")

    # Dominant species
    if total > 0:
        dominant_sp = cm_df[cluster_id].idxmax()
        purity_pct = (cm_df.loc[dominant_sp, cluster_id] / total * 100)
        print(f"  → Purity: {purity_pct:.1f}% ({dominant_sp})")


# -----------------------------
# Save results
# -----------------------------
cm_with_totals.to_csv(OUT_CONFUSION_CSV)
print(f"\n✓ Saved species x cluster table to: {OUT_CONFUSION_CSV}")


# -----------------------------
# Visualization
# -----------------------------
print("✓ Generating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap counts (no totals column)
sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"Cluster {c}" for c in cm_df.columns],
    yticklabels=species_order,
    cbar_kws={"label": "Count"},
    ax=ax1,
)
ax1.set_title("Species x Cluster (Counts)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Cluster Assignment")
ax1.set_ylabel("True Species (Accession)")

# Row percentages
counts = cm_df.to_numpy()
row_sums = counts.sum(axis=1, keepdims=True)
cm_pct = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0) * 100

sns.heatmap(
    cm_pct,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    vmin=0,
    vmax=100,
    xticklabels=[f"Cluster {c}" for c in cm_df.columns],
    yticklabels=species_order,
    cbar_kws={"label": "Percentage (%)"},
    ax=ax2,
)
ax2.set_title("Species x Cluster (Row %)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Cluster Assignment")
ax2.set_ylabel("True Species (Accession)")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"✓ Saved visualization to: {OUT_FIG}")


# -----------------------------
# Summary
# -----------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Average species separation: max cluster share per species
avg_separation = (counts.max(axis=1) / np.maximum(counts.sum(axis=1), 1) * 100).mean()

# Average cluster purity: max species share per cluster
avg_purity = (counts.max(axis=0) / np.maximum(counts.sum(axis=0), 1) * 100).mean()

print(f"""
Average species separation: {avg_separation:.1f}%
Average cluster purity:     {avg_purity:.1f}%

✓ Species labels were automatically extracted from accession-labeled headers!
  No separate mapping files needed.
""")

print("=" * 70)

