#!/usr/bin/env python3
"""
Generate DNABERT-S Chunk Embeddings (No Averaging)

This script:
1. Reads ALL sequences from a single FASTA file
2. Chunks each sequence
3. Generates embeddings for each chunk
4. Saves chunk embeddings with ground truth labels
5. No averaging - preserves individual chunks for clustering

Usage:
    python generate_chunk_embeddings.py --fasta secret_genome.fasta --output chunk_embeddings.npz
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import os
import warnings
warnings.filterwarnings('ignore')

class ChunkEmbeddingGenerator:
    """Generate DNABERT-S embeddings for genome chunks"""

    def __init__(self, model_name="zhihan1996/DNABERT-S", chunk_size=2500, overlap=0):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step_size = chunk_size if overlap == 0 else chunk_size - overlap
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"{'='*60}")
        print(f"DNABERT-S CHUNK EMBEDDING GENERATOR")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Chunk size: {chunk_size}bp")
        print(f"Overlap: {overlap}bp")
        print(f"Step size: {self.step_size}bp")

        print(f"\nLoading DNABERT-S model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!\n")

    def chunk_sequence(self, sequence, sequence_name):
        """Break sequence into chunks - KEEPS ALL CHUNKS including short ones"""
        print(f"  Chunking {sequence_name}...")
        print(f"    Sequence length: {len(sequence):,} bp")

        chunks = []
        chunk_metadata = []

        for i in range(0, len(sequence), self.step_size):
            chunk = sequence[i:i+self.chunk_size]

            # Clean chunk - remove non-ATCG characters
            clean_chunk = ''.join([c for c in chunk.upper() if c in 'ATCG'])

            # MODIFIED: Keep ALL chunks (even short ones) if they have valid bases
            if len(clean_chunk) > 0:
                chunks.append(clean_chunk)
                chunk_metadata.append({
                    'sequence_name': sequence_name,
                    'chunk_start': i,
                    'chunk_end': i + len(chunk),
                    'chunk_length': len(clean_chunk),
                    'clean_chunk_length': len(clean_chunk)
                })

        print(f"    Created {len(chunks)} chunks (including short end chunks)")
        return chunks, chunk_metadata

    def get_embedding_for_chunk(self, chunk):
        """Generate DNABERT-S embedding for a single chunk"""
        with torch.no_grad():
            inputs = self.tokenizer(chunk, return_tensors='pt',
                                  truncation=True, max_length=1024)["input_ids"]
            inputs = inputs.to(self.device)
            hidden_states = self.model(inputs)[0]
            embedding = torch.mean(hidden_states[0], dim=0).cpu().numpy()
            return embedding

    def process_fasta_file(self, fasta_file):
        """Process ALL sequences in a FASTA file"""
        print(f"\n{'='*60}")
        print(f"Processing FASTA file: {fasta_file}")
        print(f"{'='*60}\n")

        all_chunk_embeddings = []
        all_chunk_metadata = []
        sequence_count = 0

        # Read ALL sequences from the FASTA file
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_count += 1
            sequence_name = record.id
            sequence = str(record.seq).upper()

            print(f"\n[Sequence {sequence_count}] {sequence_name}")
            print(f"  Length: {len(sequence):,} bp")

            # Chunk this sequence
            chunks, metadata = self.chunk_sequence(sequence, sequence_name)

            # Generate embeddings for each chunk
            print(f"  Generating embeddings for {len(chunks)} chunks...")

            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                embedding = self.get_embedding_for_chunk(chunk)
                all_chunk_embeddings.append(embedding)

                # Add chunk index and embedding dimension to metadata
                meta['chunk_index'] = len(all_chunk_embeddings) - 1
                meta['embedding_dim'] = len(embedding)
                all_chunk_metadata.append(meta)

                if (i + 1) % 500 == 0:
                    print(f"    Progress: {i+1}/{len(chunks)} chunks")

            print(f"  ✓ Completed {len(chunks)} chunks")

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total sequences processed: {sequence_count}")
        print(f"Total chunks generated: {len(all_chunk_embeddings)}")
        print(f"Embedding dimension: {all_chunk_embeddings[0].shape[0]}")

        chunk_embeddings = np.array(all_chunk_embeddings)
        metadata_df = pd.DataFrame(all_chunk_metadata)

        return chunk_embeddings, metadata_df, sequence_count

    def save_embeddings(self, chunk_embeddings, metadata_df, output_file):
        """Save chunk embeddings and metadata"""
        print(f"\n{'='*60}")
        print(f"SAVING CHUNK EMBEDDINGS")
        print(f"{'='*60}")

        # Save embeddings as .npz
        np.savez_compressed(
            output_file,
            embeddings=chunk_embeddings,
            model_name=self.model_name,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        print(f"✓ Chunk embeddings saved to: {output_file}")
        print(f"  Shape: {chunk_embeddings.shape}")
        print(f"  Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

        # Save metadata as CSV
        metadata_file = output_file.replace('.npz', '_metadata.csv')
        metadata_df.to_csv(metadata_file, index=False)
        print(f"✓ Metadata saved to: {metadata_file}")

        # Create summary
        summary_file = output_file.replace('.npz', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CHUNK EMBEDDING GENERATION SUMMARY\n")
            f.write("="*60 + "\n\n")

            f.write("MODEL:\n")
            f.write(f"  {self.model_name}\n")
            f.write(f"  Chunk size: {self.chunk_size} bp\n")
            f.write(f"  Overlap: {self.overlap} bp\n")
            f.write(f"  Embedding dimension: {chunk_embeddings.shape[1]}\n\n")

            f.write("DATA:\n")
            f.write(f"  Total chunks: {len(chunk_embeddings):,}\n")
            f.write(f"  Unique sequences: {metadata_df['sequence_name'].nunique()}\n\n")

            f.write("CHUNKS PER SEQUENCE:\n")
            for seq_name, group in metadata_df.groupby('sequence_name'):
                f.write(f"  {seq_name}: {len(group):,} chunks\n")

            f.write(f"\n{'='*60}\n")
            f.write("NEXT STEPS:\n")
            f.write("  1. Cluster the chunks:\n")
            f.write(f"     python cluster_chunks.py --embeddings {output_file}\n\n")
            f.write("  2. Evaluate clustering:\n")
            f.write(f"     python evaluate_clustering.py --embeddings {output_file} --labels {metadata_file}\n")

        print(f"✓ Summary saved to: {summary_file}")
        print(f"\n{'='*60}")
        print("READY FOR CLUSTERING!")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Generate DNABERT-S chunk embeddings (no averaging)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
  # Step 1: Concatenate genomes
  cat genome1.fasta genome2.fasta genome3.fasta > secret_genome.fasta

  # Step 2: Generate chunk embeddings
  python generate_chunk_embeddings.py --fasta secret_genome.fasta --output chunks.npz

  # Step 3: Cluster (use separate script)
  python cluster_chunks.py --embeddings chunks.npz
        """
    )

    parser.add_argument("--fasta", required=True,
                       help="Input FASTA file (can contain multiple sequences)")
    parser.add_argument("--output", default="chunk_embeddings.npz",
                       help="Output file for embeddings (default: chunk_embeddings.npz)")
    parser.add_argument("--model", default="zhihan1996/DNABERT-S",
                       help="DNABERT model name")
    parser.add_argument("--chunk-size", type=int, default=2500,
                       help="Chunk size in bp (default: 2500)")
    parser.add_argument("--overlap", type=int, default=0,
                       help="Overlap between chunks in bp (default: 0)")

    args = parser.parse_args()

    # Initialize generator
    generator = ChunkEmbeddingGenerator(
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

    # Process FASTA file (all sequences)
    chunk_embeddings, metadata_df, seq_count = generator.process_fasta_file(args.fasta)

    # Save embeddings
    generator.save_embeddings(chunk_embeddings, metadata_df, args.output)

    print(f"\n✓ Generated embeddings for {len(chunk_embeddings):,} chunks from {seq_count} sequences")
    print(f"✓ Ready for clustering experiment!")

if __name__ == "__main__":
    main()
