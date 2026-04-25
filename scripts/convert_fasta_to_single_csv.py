#!/usr/bin/env python3

import random
import argparse
from Bio import SeqIO
import csv
from pathlib import Path


def extract_sequences_from_fasta(fasta_path, sequence_length=10000):
    """Extract fixed-length non-overlapping DNA chunks from a .fasta file."""
    sequences = []

    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_str = str(record.seq).upper()

            # Extract non-overlapping chunks of specified length
            for i in range(0, len(seq_str) - sequence_length + 1, sequence_length):
                chunk = seq_str[i:i + sequence_length]

                # Keep only full-length chunks without N
                if len(chunk) == sequence_length and "N" not in chunk:
                    sequences.append(chunk)

    return sequences


def create_pairs_from_sequences(sequences, species_name):
    """Create pairs of sequences from the same species/file."""
    pairs = []

    random.shuffle(sequences)

    # Pair consecutive sequences
    for i in range(0, len(sequences) - 1, 2):
        pairs.append({
            "seq1": sequences[i],
            "seq2": sequences[i + 1],
            "species": species_name
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Convert FASTA files into one CSV of same-species fixed-length DNA chunk pairs"
    )

    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        type=str,
        help="Directory containing .fasta files"
    )

    parser.add_argument(
        "-o", "--output-csv",
        required=True,
        type=str,
        help="Output CSV filename, e.g. train_complete.csv"
    )

    parser.add_argument(
        "--length",
        type=int,
        default=10000,
        help="Chunk length in bp (default: 10000)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.fasta",
        help="File pattern to match (default: *.fasta)"
    )

    parser.add_argument(
        "--track-species",
        action="store_true",
        help="Add species column to output CSV"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return

    fasta_files = sorted(input_dir.glob(args.pattern))

    if not fasta_files:
        print(f"Error: no files matching '{args.pattern}' found in {input_dir}")
        return

    print(f"Found {len(fasta_files)} FASTA files")

    all_pairs = []
    species_stats = {}

    for fasta_file in fasta_files:
        # Use filename without extension as species identifier
        species_name = fasta_file.stem

        print(f"Processing {fasta_file.name} (species: {species_name})...")

        sequences = extract_sequences_from_fasta(
            fasta_file,
            sequence_length=args.length
        )

        pairs = create_pairs_from_sequences(sequences, species_name)
        all_pairs.extend(pairs)

        species_stats[species_name] = {
            "filename": fasta_file.name,
            "num_sequences": len(sequences),
            "num_pairs": len(pairs)
        }

        print(f"  Extracted {len(sequences)} chunks")
        print(f"  Created {len(pairs)} pairs")

    if not all_pairs:
        print("Error: no pairs were created. Check your FASTA files and chunk length.")
        return

    random.shuffle(all_pairs)

    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"Writing output to: {output_csv}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        if args.track_species:
            writer.writerow(["sequence1", "sequence2", "species"])

        for pair in all_pairs:
            if args.track_species:
                writer.writerow([pair["seq1"], pair["seq2"], pair["species"]])
            else:
                writer.writerow([pair["seq1"], pair["seq2"]])

    print("\n" + "=" * 60)
    print("Species Summary")
    print("=" * 60)
    print(f"{'Species':<30} {'Chunks':<10} {'Pairs':<10}")
    print("-" * 60)

    for species_name, stats in sorted(
        species_stats.items(),
        key=lambda x: x[1]["num_pairs"],
        reverse=True
    )[:10]:
        print(f"{species_name:<30} {stats['num_sequences']:<10} {stats['num_pairs']:<10}")

    if len(species_stats) > 10:
        print(f"... and {len(species_stats) - 10} more species")

    print("\nDone.")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
