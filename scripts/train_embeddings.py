#!/usr/bin/env python3
"""
CLI Entry Point: Train or Evolve 523-Word Visual-Semantic Skip-Gram Embeddings.

Supports:
1. Training from scratch on a text corpus with Semantic Negative Sampling and
   Degree-Penalized Positive Pair Weighting (`--mode scratch`).
2. Inspecting and verifying the pre-trained 523-word Skip-Gram checkpoint (`--mode inspect`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visual_semantic_planner import (
    VOCAB_523,
    build_my_embeddings,
    find_similar_words,
    process_text_network_restricted,
    solve_analogy,
    train_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or inspect 523-word Visual-Semantic Skip-Gram Embeddings."
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "scratch"],
        default="inspect",
        help="Run mode: 'inspect' pre-trained checkpoint or train from 'scratch'.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="",
        help="Path to text corpus file for --mode scratch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "models" / "best_skipgram_523words.pth"),
        help="Path to load/save model checkpoint.",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--num-negative", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "scratch":
        if not args.corpus or not Path(args.corpus).exists():
            parser.error("--corpus must point to an existing text file when --mode=scratch.")
        network_data = process_text_network_restricted(args.corpus, target_nodes=VOCAB_523, verbose=True)
        res = train_embeddings(
            network_data=network_data,
            embedding_dim=args.embedding_dim,
            context_size=args.context_size,
            num_negative=args.num_negative,
            learning_rate=args.lr,
            epochs=args.epochs,
            save_path=args.checkpoint,
        )
        print(f"Trained {len(res['nodes'])} embeddings of dim {res['embeddings'].shape[1]} -> {args.checkpoint}")
    else:
        vocab, embeddings = build_my_embeddings(args.checkpoint)
        nodes = [w for w, _ in sorted(vocab.items(), key=lambda x: x[1])]
        print("=" * 72)
        print("SKIP-GRAM 523-WORD SEMANTIC MANIFOLD SUMMARY")
        print("=" * 72)
        print(f"Vocabulary Size : {len(vocab)}")
        print(f"Embedding Shape : {embeddings.shape}")

        for probe in ("airplane", "dinosaur", "maple_tree", "crab", "sweet_pepper"):
            neighbors = find_similar_words(probe, nodes, embeddings, top_k=5)
            formatted = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
            print(f"Top-5 Neighbors for '{probe:12s}': {formatted}")

        for a, b, c in [("man", "woman", "boy"), ("bear", "lion", "tiger"), ("maple_tree", "oak_tree", "palm_tree")]:
            ans = solve_analogy(a, b, c, nodes, embeddings, top_k=3)
            formatted = ", ".join(f"{w} ({s:.3f})" for w, s in ans)
            print(f"Analogy {a} : {b} :: {c} : ? -> {formatted}")


if __name__ == "__main__":
    main()
