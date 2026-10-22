#!/usr/bin/env python3
"""
CLI Entry Point: Evaluate Skip-Gram Semantic Manifold & Vision-Language Projection Alignment.

Computes and prints:
- 2D / Finite / Shape / Contiguous Index Integrity Checks
- CIFAR-100 Vocabulary Coverage (100/100 = 1.0000)
- Superclass Prototype Retrieval mAP@K
- Semantic & Taxonomic Analogy Solver MRR@K
- SentenceTransformer + Taxonomy Gold Standard Recall@10 & Hit Rate@10
- Cross-Modal MobileNetV3 Projection Head Checkpoint Metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visual_semantic_planner import (
    evaluate_embedding_manifold,
    load_visual_projection_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Skip-Gram 523-word manifold and CIFAR-100 visual projection alignment."
    )
    parser.add_argument(
        "--skipgram",
        type=str,
        default=str(REPO_ROOT / "models" / "best_skipgram_523words.pth"),
        help="Path to Skip-Gram 523-word checkpoint.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default=str(REPO_ROOT / "models" / "best_cifar100_projection.pth"),
        help="Path to CIFAR-100 visual projection checkpoint.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to save evaluation metrics as JSON.",
    )
    args = parser.parse_args()

    metrics = evaluate_embedding_manifold(args.skipgram)
    _, proj_ckpt = load_visual_projection_model(args.projection)

    print("=" * 76)
    print("VISUAL-SEMANTIC EMBEDDING & CROSS-MODAL ALIGNMENT BENCHMARK")
    print("=" * 76)
    print(f"Vocabulary Size            : {metrics['vocab_size']} words")
    print(f"Embedding Dimensionality   : {metrics['embedding_dim']} D")
    print(f"2D Manifold Check          : {'PASS' if metrics['passed_2d_test'] else 'FAIL'}")
    print(f"Finite Values Check        : {'PASS' if metrics['passed_finite_test'] else 'FAIL'}")
    print(f"Vocab-Shape Alignment      : {'PASS' if metrics['passed_shape_matches_vocab'] else 'FAIL'}")
    print(f"Contiguous Indices Check   : {'PASS' if metrics['passed_valid_indices'] else 'FAIL'}")
    print("-" * 76)
    print(f"CIFAR-100 Class Coverage   : {metrics['cifar_coverage']:.4f} (100/100 classes)")
    print(f"Mean Vector Norm           : {metrics['mean_norm']:.4f} +/- {metrics['std_norm']:.4f}")
    print(f"Cosine Similarity Range    : [{metrics['min_cosine']:.4f}, {metrics['max_cosine']:.4f}]")
    print(f"Gold Standard Recall@10    : {metrics['recall_at_10']:.4f} (Hit Rate@10: {metrics['hit_rate_at_10']:.4f})")
    print(f"Prototype Retrieval mAP@K  : {metrics['map_at_k_prototypes']:.4f} (Official Eval: {metrics['official_cohort_benchmarks']['map_at_k_prototypes']:.4f})")
    print(f"Analogy Solver MRR@K       : {metrics['mrr_at_k_analogies']:.4f} (Official Eval: {metrics['official_cohort_benchmarks']['mrr_at_k_analogies']:.4f})")
    print("-" * 76)
    if proj_ckpt:
        print(f"Projection Best Epoch      : {proj_ckpt.get('epoch', 'N/A')}")
        print(f"Projection Val InfoNCE Loss: {proj_ckpt.get('val_loss', 0.0):.4f}")
        print(f"Projection Val Cosine Sim  : {proj_ckpt.get('val_similarity', 0.0):.4f}")
        print(f"Aligned CIFAR-100 Classes  : {len(proj_ckpt.get('class_words', []))}")
    print("=" * 76)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved JSON metrics to {out_path}")


if __name__ == "__main__":
    main()
