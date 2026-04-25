import gzip
import random
import argparse
from Bio import SeqIO
import csv
from pathlib import Path

def extract_sequences_from_fasta_gz(fasta_gz_path, sequence_length=10000):
    """Extract sequences from a fasta.gz file"""
    sequences = []
    with gzip.open(fasta_gz_path, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            seq_str = str(record.seq).upper()
            # Extract non-overlapping chunks of specified length
            for i in range(0, len(seq_str) - sequence_length + 1, sequence_length):
                chunk = seq_str[i:i + sequence_length]
                if len(chunk) == sequence_length and 'N' not in chunk:
                    sequences.append(chunk)
    return sequences

def create_pairs_from_sequences(sequences, species_name):
    """Create pairs of non-overlapping sequences from the same species"""
    pairs = []
    random.shuffle(sequences)
    
    # Create pairs from consecutive sequences with species tracking
    for i in range(0, len(sequences) - 1, 2):
        pairs.append({
            'seq1': sequences[i],
            'seq2': sequences[i + 1],
            'species': species_name
        })
    
    return pairs

def main():
    parser = argparse.ArgumentParser(
        description='Convert FASTA.gz files to CSV pairs for DNABERT-S training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python convert_fasta_to_csv.py -i /data/bacteria/X -o train.csv test.csv
  python convert_fasta_to_csv.py -i ./genomes -o output_train.csv output_test.csv --split 0.8 --length 5000
  python convert_fasta_to_csv.py -i /data/bacteria/X -o train.csv test.csv --seed 123 --track-species
        '''
    )
    
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help='Directory containing .fasta.gz files')
    parser.add_argument('-o', '--output', required=True, nargs=2, metavar=('TRAIN', 'TEST'),
                        help='Output CSV files: train_file.csv test_file.csv')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Train split ratio (default: 0.9 for 90%% train, 10%% test)')
    parser.add_argument('--length', type=int, default=10000,
                        help='Sequence length in bp (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--pattern', type=str, default='*.fasta.gz',
                        help='File pattern to match (default: *.fasta.gz)')
    parser.add_argument('--track-species', action='store_true',
                        help='Add species column to output CSVs (default: False)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Output file for species metadata (optional)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Validate inputs
    data_dir = Path(args.input_dir)
    if not data_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    output_train, output_test = args.output
    
    if not 0 < args.split < 1:
        print(f"Error: Split ratio must be between 0 and 1, got {args.split}")
        return
    
    all_pairs = []
    species_stats = {}
    
    # Process all fasta.gz files
    print(f"Processing files matching '{args.pattern}' in {data_dir}...")
    fasta_files = sorted(data_dir.glob(args.pattern))
    
    if not fasta_files:
        print(f"Error: No files matching '{args.pattern}' found in {data_dir}")
        return
    
    print(f"Found {len(fasta_files)} files")
    
    for fasta_file in fasta_files:
        # Use filename (without extension) as species identifier
        species_name = fasta_file.stem.replace('.fasta', '')
        
        print(f"Processing {fasta_file.name} (species: {species_name})...")
        sequences = extract_sequences_from_fasta_gz(fasta_file, args.length)
        pairs = create_pairs_from_sequences(sequences, species_name)
        all_pairs.extend(pairs)
        
        # Track statistics per species
        species_stats[species_name] = {
            'filename': fasta_file.name,
            'num_sequences': len(sequences),
            'num_pairs': len(pairs)
        }
        
        print(f"  Extracted {len(pairs)} pairs from {len(sequences)} sequences")
    
    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"Total species: {len(species_stats)}")
    
    if len(all_pairs) == 0:
        print("Error: No sequence pairs were extracted. Check your sequence length and input files.")
        return
    
    # Shuffle all pairs
    random.shuffle(all_pairs)
    
    # Split into train and test
    split_idx = int(len(all_pairs) * args.split)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    print(f"Train pairs: {len(train_pairs)} ({args.split*100:.1f}%)")
    print(f"Test pairs: {len(test_pairs)} ({(1-args.split)*100:.1f}%)")
    
    # Write train CSV
    print(f"\nWriting {output_train}...")
    with open(output_train, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header only if tracking species
        if args.track_species:
            writer.writerow(['sequence1', 'sequence2', 'species'])
        for pair in train_pairs:
            if args.track_species:
                writer.writerow([pair['seq1'], pair['seq2'], pair['species']])
            else:
                writer.writerow([pair['seq1'], pair['seq2']])
    
    # Write test CSV
    print(f"Writing {output_test}...")
    with open(output_test, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header only if tracking species
        if args.track_species:
            writer.writerow(['sequence1', 'sequence2', 'species'])
        for pair in test_pairs:
            if args.track_species:
                writer.writerow([pair['seq1'], pair['seq2'], pair['species']])
            else:
                writer.writerow([pair['seq1'], pair['seq2']])
    
    # Write metadata file if requested
    if args.metadata:
        print(f"\nWriting metadata to {args.metadata}...")
        with open(args.metadata, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['species', 'filename', 'num_sequences', 'num_pairs', 'train_pairs', 'test_pairs'])
            
            # Count pairs per species in train/test
            for species_name, stats in sorted(species_stats.items()):
                train_count = sum(1 for p in train_pairs if p['species'] == species_name)
                test_count = sum(1 for p in test_pairs if p['species'] == species_name)
                writer.writerow([
                    species_name,
                    stats['filename'],
                    stats['num_sequences'],
                    stats['num_pairs'],
                    train_count,
                    test_count
                ])
    
    # Print species distribution summary
    print("\n" + "="*60)
    print("Species Distribution Summary:")
    print("="*60)
    print(f"{'Species':<30} {'Pairs':<10} {'Train':<10} {'Test':<10}")
    print("-"*60)
    for species_name, stats in sorted(species_stats.items(), key=lambda x: x[1]['num_pairs'], reverse=True)[:10]:
        train_count = sum(1 for p in train_pairs if p['species'] == species_name)
        test_count = sum(1 for p in test_pairs if p['species'] == species_name)
        print(f"{species_name:<30} {stats['num_pairs']:<10} {train_count:<10} {test_count:<10}")
    
    if len(species_stats) > 10:
        print(f"... and {len(species_stats) - 10} more species")
    
    print("\n✓ Done!")
    print(f"  Train: {output_train}")
    print(f"  Test: {output_test}")
    if args.metadata:
        print(f"  Metadata: {args.metadata}")

if __name__ == '__main__':
    main()
