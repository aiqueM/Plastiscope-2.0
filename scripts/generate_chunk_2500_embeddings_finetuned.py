#!/usr/bin/env python3
"""
Generate Chunk Embeddings using a fine-tuned DNABERT model saved in ./best/

This script:
1. Reads ALL sequences from a single FASTA file
2. Chunks each sequence
3. Generates embeddings for each chunk (mean pooling across tokens)
4. Saves chunk embeddings + metadata for clustering

Usage:
  python generate_chunk_2500_embeddings_finetuned.py \
    --fasta combined.fasta \
    --best-dir ./best \
    --base-repo zhihan1996/DNABERT-2-117M \
    --output finetuned_chunks.npz
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file


class ChunkEmbeddingGenerator:
    """Generate embeddings for genome chunks using fine-tuned weights"""

    def __init__(self, base_repo, best_dir, chunk_size=2500, overlap=0, max_length=1024):
        self.base_repo = base_repo
        self.best_dir = best_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step_size = chunk_size if overlap == 0 else chunk_size - overlap
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"{'='*60}")
        print("FINE-TUNED DNABERT CHUNK EMBEDDING GENERATOR")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Base repo (code/arch): {self.base_repo}")
        print(f"Fine-tuned weights dir: {self.best_dir}")
        print(f"Chunk size: {self.chunk_size} bp")
        print(f"Overlap: {self.overlap} bp")
        print(f"Step size: {self.step_size} bp")
        print(f"Tokenizer max_length: {self.max_length}")
        print()

        # 1) Tokenizer from base repo (guarantees compatible vocab/tokenization)
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_repo, trust_remote_code=True)

        # 2) Model architecture + custom code from base repo
        print("Loading base model (architecture + custom code)...")
        self.model = AutoModel.from_pretrained(self.base_repo, trust_remote_code=True)

        # 3) Load fine-tuned weights from best_dir/model.safetensors
        weights_path = os.path.join(self.best_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Could not find {weights_path}")

        print(f"Loading fine-tuned weights: {weights_path}")
        state_dict = load_file(weights_path)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded into model")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) > 0:
            print("  (Unexpected keys are usually projection-head / contrastive head weights — OK)")

        self.model.to(self.device)
        self.model.eval()
        print("Model ready!\n")

    def chunk_sequence(self, sequence, sequence_name):
        """Break sequence into chunks (keeps short end chunks)"""
        chunks = []
        chunk_metadata = []

        for i in range(0, len(sequence), self.step_size):
            chunk = sequence[i:i + self.chunk_size]
            clean_chunk = "".join([c for c in chunk.upper() if c in "ATCG"])

            if len(clean_chunk) > 0:
                chunks.append(clean_chunk)
                chunk_metadata.append({
                    "sequence_name": sequence_name,
                    "chunk_start": i,
                    "chunk_end": i + len(chunk),
                    "chunk_length": len(clean_chunk),
                })

        return chunks, chunk_metadata

    def get_embedding_for_chunk(self, chunk):
        """Generate embedding for a single chunk (mean pool over tokens)"""
        with torch.no_grad():
            toks = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}

            out = self.model(**toks)
            last_hidden = out[0]  # [1, seq_len, hidden_dim]

            # Mean pooling over tokens (same spirit as your DNABERT-S script)
            emb = last_hidden[0].mean(dim=0).cpu().numpy()
            return emb

    def process_fasta_file(self, fasta_file):
        all_chunk_embeddings = []
        all_chunk_metadata = []
        sequence_count = 0

        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_count += 1
            sequence_name = record.id
            sequence = str(record.seq).upper()

            print(f"[Sequence {sequence_count}] {sequence_name} | {len(sequence):,} bp")
            chunks, metadata = self.chunk_sequence(sequence, sequence_name)
            print(f"  Chunks: {len(chunks)}")

            for idx, (chunk, meta) in enumerate(zip(chunks, metadata)):
                emb = self.get_embedding_for_chunk(chunk)
                all_chunk_embeddings.append(emb)

                meta["chunk_index"] = len(all_chunk_embeddings) - 1
                meta["embedding_dim"] = int(emb.shape[0])
                all_chunk_metadata.append(meta)

                if (idx + 1) % 500 == 0:
                    print(f"    Progress: {idx+1}/{len(chunks)}")

            print("  ✓ Done\n")

        chunk_embeddings = np.array(all_chunk_embeddings)
        metadata_df = pd.DataFrame(all_chunk_metadata)
        return chunk_embeddings, metadata_df, sequence_count

    def save_embeddings(self, chunk_embeddings, metadata_df, output_file):
        np.savez_compressed(
            output_file,
            embeddings=chunk_embeddings,
            base_repo=self.base_repo,
            best_dir=self.best_dir,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            max_length=self.max_length,
        )

        metadata_file = output_file.replace(".npz", "_metadata.csv")
        metadata_df.to_csv(metadata_file, index=False)

        print(f"✓ Saved embeddings: {output_file} | shape={chunk_embeddings.shape}")
        print(f"✓ Saved metadata:   {metadata_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True, help="Input FASTA (can contain multiple sequences)")
    p.add_argument("--output", default="finetuned_chunk_embeddings.npz", help="Output .npz")
    p.add_argument("--best-dir", required=True, help="Path to fine-tuned best/ folder")
    p.add_argument("--base-repo", default="zhihan1996/DNABERT-2-117M",
                   help="Base repo to provide architecture + custom code")
    p.add_argument("--chunk-size", type=int, default=2500)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--max-length", type=int, default=1024, help="Tokenizer max_length (tokens)")
    p.add_argument(
    "--tokenizers-parallelism",
    choices=["true", "false"],
    default="true",
    help="Enable/disable HuggingFace tokenizers multithreading"
    )
    args = p.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    print(f"TOKENIZERS_PARALLELISM={os.environ['TOKENIZERS_PARALLELISM']}")

    gen = ChunkEmbeddingGenerator(
        base_repo=args.base_repo,
        best_dir=args.best_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_length=args.max_length
    )

    embs, meta, nseq = gen.process_fasta_file(args.fasta)
    gen.save_embeddings(embs, meta, args.output)

    print(f"\n✓ Generated {len(embs):,} embeddings from {nseq} sequences")


if __name__ == "__main__":
    main()

