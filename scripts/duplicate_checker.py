import sys
import argparse

def read_species_list(filename):
    with open(filename, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def main():
    parser = argparse.ArgumentParser(
        description='Check for duplicate species across multiple genome assembly levels'
    )
    parser.add_argument(
        '--complete',
        default='complete_species.txt',
        help='File with complete genome species list (default: complete_species.txt)'
    )
    parser.add_argument(
        '--chromosome',
        default='chromosome_species.txt',
        help='File with chromosome genome species list (default: chromosome_species.txt)'
    )
    parser.add_argument(
        '--scaffold',
        default='scaffold_species.txt',
        help='File with scaffold genome species list (default: scaffold_species.txt)'
    )
    parser.add_argument(
        '--contig',
        default='contig_species.txt',
        help='File with contig genome species list (default: contig_species.txt)'
    )

    args = parser.parse_args()

    # Load all species lists (only if files exist)
    datasets = {}
    
    try:
        datasets['Complete'] = read_species_list(args.complete)
    except FileNotFoundError:
        datasets['Complete'] = set()
        print(f"Warning: {args.complete} not found, skipping complete genomes")
    
    try:
        datasets['Chromosome'] = read_species_list(args.chromosome)
    except FileNotFoundError:
        datasets['Chromosome'] = set()
        print(f"Warning: {args.chromosome} not found, skipping chromosomes")
    
    try:
        datasets['Scaffold'] = read_species_list(args.scaffold)
    except FileNotFoundError:
        datasets['Scaffold'] = set()
        print(f"Warning: {args.scaffold} not found, skipping scaffolds")
    
    try:
        datasets['Contig'] = read_species_list(args.contig)
    except FileNotFoundError:
        datasets['Contig'] = set()
        print(f"Warning: {args.contig} not found, skipping contigs")

    print("\n" + "="*70)
    print("DUPLICATE ANALYSIS ACROSS ALL ASSEMBLY LEVELS")
    print("="*70)
    
    # Print counts
    for name, species_set in datasets.items():
        if species_set:
            print(f"{name}: {len(species_set)} species")
    
    # Find all pairwise overlaps
    print("\n" + "="*70)
    print("PAIRWISE DUPLICATES")
    print("="*70)
    
    dataset_names = [name for name, s in datasets.items() if s]
    found_duplicates = False
    
    for i, name1 in enumerate(dataset_names):
        for name2 in dataset_names[i+1:]:
            overlap = datasets[name1] & datasets[name2]
            if overlap:
                found_duplicates = True
                print(f"\n{name1} ∩ {name2}: {len(overlap)} duplicates")
                for species in sorted(overlap):
                    print(f"  - {species}")
    
    if not found_duplicates:
        print("\nNo pairwise duplicates found!")
    
    # Find species that appear in multiple datasets
    print("\n" + "="*70)
    print("SPECIES IN MULTIPLE DATASETS")
    print("="*70)
    
    all_species = set()
    for species_set in datasets.values():
        all_species.update(species_set)
    
    multi_dataset_species = []
    for species in sorted(all_species):
        locations = [name for name, s in datasets.items() if species in s]
        if len(locations) > 1:
            multi_dataset_species.append((species, locations))
    
    if multi_dataset_species:
        print(f"\nFound {len(multi_dataset_species)} species in multiple datasets:\n")
        for species, locations in multi_dataset_species:
            print(f"  {species}")
            print(f"    Found in: {', '.join(locations)}")
    else:
        print("\nNo species found in multiple datasets!")
    
    # Print unique counts
    print("\n" + "="*70)
    print("UNIQUE SPECIES PER DATASET")
    print("="*70)
    
    for name, species_set in datasets.items():
        if species_set:
            others = set()
            for other_name, other_set in datasets.items():
                if other_name != name:
                    others.update(other_set)
            unique = species_set - others
            print(f"\nOnly in {name}: {len(unique)} species")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if multi_dataset_species:
        print(f"\n⚠️  Found {len(multi_dataset_species)} species with multiple assembly levels!")
        print("\nQuality hierarchy (keep highest quality):")
        print("  1. Complete genome (best)")
        print("  2. Chromosome")
        print("  3. Scaffold")
        print("  4. Contig (lowest)")
        print("\nOptions:")
        print("  A. Keep only the highest quality version per species")
        print("  B. Keep all versions for more training data per species")
        print("  C. Merge sequences from all versions into single species dataset")
    else:
        print("\n✓ No duplicates found - all species are unique across datasets!")
        print("You can safely merge all datasets.")

if __name__ == '__main__':
    main()
