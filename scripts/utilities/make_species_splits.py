#!/usr/bin/env python3

import csv
import os
import random
import argparse
from collections import defaultdict


def read_species_csv(csv_path):
    """
    Read a 3-column CSV with header:
    sequence1,sequence2,species
    """
    rows = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"sequence1", "sequence2", "species"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing)}"
            )

        for i, row in enumerate(reader, start=2):  # start=2 because header is line 1
            seq1 = (row.get("sequence1") or "").strip()
            seq2 = (row.get("sequence2") or "").strip()
            species = (row.get("species") or "").strip()

            if not seq1 or not seq2 or not species:
                print(f"Skipping incomplete row at line {i}")
                continue

            rows.append({
                "sequence1": seq1,
                "sequence2": seq2,
                "species": species
            })

    if not rows:
        raise ValueError("No valid rows found in input CSV.")

    return rows


def group_rows_by_species(rows):
    species_to_rows = defaultdict(list)
    for row in rows:
        species_to_rows[row["species"]].append(row)
    return species_to_rows


def split_species(species_list, train_ratio, val_ratio, test_ratio, seed):
    """
    Split unique species into train/val/test buckets.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )

    species_list = list(species_list)
    rng = random.Random(seed)
    rng.shuffle(species_list)

    n = len(species_list)
    if n < 3:
        raise ValueError(
            f"Need at least 3 unique species for train/val/test split, found {n}"
        )

    train_count = int(round(n * train_ratio))
    val_count = int(round(n * val_ratio))

    # Make sure test gets the remainder
    if train_count < 1:
        train_count = 1
    if val_count < 1:
        val_count = 1

    if train_count + val_count >= n:
        # Leave at least 1 species for test
        val_count = max(1, n - train_count - 1)

    test_count = n - train_count - val_count

    if test_count < 1:
        # Rebalance to guarantee non-empty test split
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        test_count = n - train_count - val_count

    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            f"Could not create non-empty train/val/test species splits. "
            f"Counts: train={train_count}, val={val_count}, test={test_count}"
        )

    train_species = species_list[:train_count]
    val_species = species_list[train_count:train_count + val_count]
    test_species = species_list[train_count + val_count:]

    return set(train_species), set(val_species), set(test_species)


def collect_rows(species_set, species_to_rows):
    collected = []
    for sp in species_set:
        collected.extend(species_to_rows[sp])
    return collected


def write_two_column_csv(rows, output_path):
    """
    Write 2-column CSV with no header:
    sequence1,sequence2
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([row["sequence1"], row["sequence2"]])


def write_three_column_csv(rows, output_path):
    """
    Write 3-column CSV with header:
    sequence1,sequence2,species
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence1", "sequence2", "species"])
        for row in rows:
            writer.writerow([row["sequence1"], row["sequence2"], row["species"]])


def summarize_split(name, rows, species_set):
    print(f"\n{name.upper()} SPLIT")
    print(f"  Species count: {len(species_set)}")
    print(f"  Row count:     {len(rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create species-based train/val/test splits from a 3-column CSV."
    )

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input 3-column CSV with header: sequence1,sequence2,species"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output split files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of species for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of species for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of species for testing (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible species split"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="species_split",
        help="Prefix for output filenames"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Reading input CSV...")
    rows = read_species_csv(args.input_csv)
    species_to_rows = group_rows_by_species(rows)
    unique_species = list(species_to_rows.keys())

    print(f"Total rows: {len(rows)}")
    print(f"Unique species: {len(unique_species)}")

    print("\nSplitting by species...")
    train_species, val_species, test_species = split_species(
        unique_species,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    train_rows = collect_rows(train_species, species_to_rows)
    val_rows = collect_rows(val_species, species_to_rows)
    test_rows = collect_rows(test_species, species_to_rows)

    summarize_split("train", train_rows, train_species)
    summarize_split("val", val_rows, val_species)
    summarize_split("test", test_rows, test_species)

    # 2-column no-header files for current trainer
    train_2col = os.path.join(args.output_dir, f"{args.prefix}_train_2col.csv")
    val_2col = os.path.join(args.output_dir, f"{args.prefix}_val_2col.csv")
    test_2col = os.path.join(args.output_dir, f"{args.prefix}_test_2col.csv")

    # 3-column header files for traceability
    train_3col = os.path.join(args.output_dir, f"{args.prefix}_train_3col.csv")
    val_3col = os.path.join(args.output_dir, f"{args.prefix}_val_3col.csv")
    test_3col = os.path.join(args.output_dir, f"{args.prefix}_test_3col.csv")

    print("\nWriting 2-column no-header files...")
    write_two_column_csv(train_rows, train_2col)
    write_two_column_csv(val_rows, val_2col)
    write_two_column_csv(test_rows, test_2col)

    print("Writing 3-column header files...")
    write_three_column_csv(train_rows, train_3col)
    write_three_column_csv(val_rows, val_3col)
    write_three_column_csv(test_rows, test_3col)

    # Save species lists too
    train_species_txt = os.path.join(args.output_dir, f"{args.prefix}_train_species.txt")
    val_species_txt = os.path.join(args.output_dir, f"{args.prefix}_val_species.txt")
    test_species_txt = os.path.join(args.output_dir, f"{args.prefix}_test_species.txt")

    with open(train_species_txt, "w", encoding="utf-8") as f:
        for sp in sorted(train_species):
            f.write(sp + "\n")

    with open(val_species_txt, "w", encoding="utf-8") as f:
        for sp in sorted(val_species):
            f.write(sp + "\n")

    with open(test_species_txt, "w", encoding="utf-8") as f:
        for sp in sorted(test_species):
            f.write(sp + "\n")

    print("\nDone.")
    print("Output files created:")
    print(f"  {train_2col}")
    print(f"  {val_2col}")
    print(f"  {test_2col}")
    print(f"  {train_3col}")
    print(f"  {val_3col}")
    print(f"  {test_3col}")


if __name__ == "__main__":
    main()
