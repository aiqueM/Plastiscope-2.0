#!/usr/bin/env python3

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def count_bp_in_fasta(path: Path) -> int:
    """
    Count total base pairs in a FASTA or FASTA.GZ file.
    Ignores header lines beginning with '>'.
    """
    total = 0

    if path.suffix == ".gz":
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    with opener(path, mode, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith(">"):
                continue
            total += len(line.strip())

    return total


def normalize_genome_name(path: Path) -> str:
    """
    Normalize genome names so the same genome can be matched across folders.
    Removes common FASTA suffixes.
    """
    name = path.name

    for suffix in [".gz", ".fasta", ".fa", ".fna"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # handle double suffix cases like .fasta.gz
    for suffix in [".fasta", ".fa", ".fna"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


def choose_bp_unit(values: List[int]) -> Tuple[str, float]:
    """
    Choose a readable plotting unit.
    """
    if not values:
        return "bp", 1.0

    med = sorted(values)[len(values) // 2]

    if med >= 1_000_000_000:
        return "Gbp", 1_000_000_000.0
    if med >= 1_000_000:
        return "Mbp", 1_000_000.0
    if med >= 1_000:
        return "Kbp", 1_000.0
    return "bp", 1.0


def process_folder(folder: Path, pattern: str, folder_label: str) -> List[dict]:
    """
    Process all FASTA files in one folder.
    Returns one row per genome.
    """
    rows = []
    fasta_files = sorted(folder.glob(pattern))

    for fp in fasta_files:
        bp_total = count_bp_in_fasta(fp)
        row = {
            "folder": folder_label,
            "genome_file": fp.name,
            "genome_name": normalize_genome_name(fp),
            "bp_total": bp_total,
        }
        rows.append(row)
        print(f"[OK] {folder_label}\t{fp.name}\t{bp_total}")

    return rows


def compute_unique_sets(folder_rows: Dict[str, List[dict]]) -> Dict[str, set]:
    """
    For each folder, compute the set of genome names unique to that folder.
    """
    name_sets = {
        label: {r["genome_name"] for r in rows}
        for label, rows in folder_rows.items()
    }

    unique_sets = {}
    for label, names in name_sets.items():
        other_names = set()
        for other_label, other_set in name_sets.items():
            if other_label != label:
                other_names.update(other_set)
        unique_sets[label] = names - other_names

    return unique_sets


def write_delimited_file(path: Path, rows: List[dict], fieldnames: List[str], delimiter: str = "\t") -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def plot_stacked_distribution(
    folder_rows: Dict[str, List[dict]],
    ordered_labels: List[str],
    unique_sets: Dict[str, set],
    out_png: Path,
    title: str
) -> None:
    """
    Create stacked bar plots:
    - blue = common genomes
    - red = unique genomes
    """
    all_values = []
    for label in ordered_labels:
        all_values.extend([r["bp_total"] for r in folder_rows[label]])

    unit_label, divisor = choose_bp_unit(all_values)

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    })

    fig, axes = plt.subplots(
        nrows=len(ordered_labels),
        ncols=1,
        figsize=(24, 28),
        constrained_layout=True
    )

    if len(ordered_labels) == 1:
        axes = [axes]

    common_color = "#1f77b4"   # blue
    unique_color = "#d62728"   # red

    for ax, label in zip(axes, ordered_labels):
        rows = sorted(folder_rows[label], key=lambda r: r["bp_total"])
        unique_names = unique_sets[label]

        x = list(range(1, len(rows) + 1))
        y = [r["bp_total"] / divisor for r in rows]
        colors = [
            unique_color if r["genome_name"] in unique_names else common_color
            for r in rows
        ]

        ax.bar(x, y, color=colors, width=0.9)

        n_total = len(rows)
        n_unique = len(unique_names)
        n_common = n_total - n_unique

        handles = [
            Patch(facecolor=common_color, label=f"Common genomes: {n_common}"),
            Patch(facecolor=unique_color, label=f"Unique genomes: {n_unique}")
        ]

        ax.legend(handles=handles, loc="upper left")
        ax.set_title(label)
        ax.set_ylabel(f"Genome length ({unit_label})")
        ax.set_xlabel("Genomes (sorted by total length)")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(title, fontsize=34)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate genome length distributions across four folders."
    )
    parser.add_argument("--complete", required=True, help="Folder for complete genomes")
    parser.add_argument("--chromosomes", required=True, help="Folder for chromosome genomes")
    parser.add_argument("--scaffolds", required=True, help="Folder for scaffold genomes")
    parser.add_argument("--contigs", required=True, help="Folder for contig genomes")
    parser.add_argument("--pattern", default="*.fasta*", help="Glob pattern (default: *.fasta*)")
    parser.add_argument("--outdir", default="genome_length_outputs", help="Output directory")
    parser.add_argument("--run-label", default="poster", help="Suffix for output files")

    args = parser.parse_args()

    ordered = [
        ("Complete", Path(args.complete).expanduser().resolve()),
        ("Chromosomes", Path(args.chromosomes).expanduser().resolve()),
        ("Scaffolds", Path(args.scaffolds).expanduser().resolve()),
        ("Contigs", Path(args.contigs).expanduser().resolve()),
    ]

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    folder_rows: Dict[str, List[dict]] = {}
    all_rows: List[dict] = []

    for label, folder in ordered:
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"ERROR: folder not found or not a directory: {folder}")

        rows = process_folder(folder, args.pattern, label)
        folder_rows[label] = rows
        all_rows.extend(rows)

        per_folder_tsv = outdir / f"{label.lower()}_bp_totals_{args.run_label}.tsv"
        write_delimited_file(
            per_folder_tsv,
            rows,
            fieldnames=["folder", "genome_file", "genome_name", "bp_total"],
            delimiter="\t"
        )

    unique_sets = compute_unique_sets(folder_rows)

    summary_rows = []
    for label, _ in ordered:
        n_total = len(folder_rows[label])
        n_unique = len(unique_sets[label])
        n_common = n_total - n_unique
        summary_rows.append({
            "folder": label,
            "n_genomes": n_total,
            "n_common": n_common,
            "n_unique": n_unique,
        })

    summary_tsv = outdir / f"folder_summary_{args.run_label}.tsv"
    write_delimited_file(
        summary_tsv,
        summary_rows,
        fieldnames=["folder", "n_genomes", "n_common", "n_unique"],
        delimiter="\t"
    )

    master_csv = outdir / f"all_genome_lengths_{args.run_label}.csv"
    write_delimited_file(
        master_csv,
        all_rows,
        fieldnames=["folder", "genome_file", "genome_name", "bp_total"],
        delimiter=","
    )

    plot_path = outdir / f"stacked_genome_length_distributions_{args.run_label}.png"
    plot_stacked_distribution(
        folder_rows=folder_rows,
        ordered_labels=[label for label, _ in ordered],
        unique_sets=unique_sets,
        out_png=plot_path,
        title="Genome Length Distributions Across Assembly Levels"
    )

    print("\n=== DONE ===")
    print(f"Output directory:   {outdir}")
    print(f"Master CSV:         {master_csv}")
    print(f"Summary TSV:        {summary_tsv}")
    print(f"Plot PNG:           {plot_path}")


if __name__ == "__main__":
    main()
