"""
Unit and Integration Test Suite for Visual-Semantic Embeddings and Neuro-Symbolic PDDL Planning.

Covers:
1. Skip-Gram Manifold & Automated Cohort Grading Checks:
   - 2D embeddings tensor check
   - Finite values check (no NaN or Inf)
   - Shape matches vocabulary length (523 x 64)
   - Valid contiguous indices (0..522)
   - 100% CIFAR-100 class coverage (1.0)
   - Prototype retrieval mAP@K and Analogy solver MRR@K
2. Cross-Modal Visual Projection (`ImageEncoder` + `best_cifar100_projection.pth`):
   - Checkpoint loading, forward projection shape (B, 64), L2 normalization,
     symmetric InfoNCE loss, and constrained CIFAR-100 retrieval + PDDL sanitization.
3. Symbolic PDDL Parser, Action Grounding, and Precondition-Aware A* Search:
   - Domain parsing (7 lifted actions), problem parsing (100 CIFAR items + 2 tools, 349 init predicates),
     action grounding (42,018 grounded actions), conflict resolution, and end-to-end multimodal
     planning across both text and visual inputs (9/9 text plans and 9/9 image plans).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visual_semantic_planner import (
    BENCHMARK_SCENARIOS,
    CIFAR_100_CLASSES,
    VALID_CIFAR_VOCAB,
    Action,
    ActionGrounder,
    ImageEncoder,
    PDDLParser,
    Predicate,
    SkipGramModel,
    State,
    astar_search,
    bfs_search,
    build_my_embeddings,
    classify_image_to_pddl_object,
    clean_object_name,
    compute_contrastive_loss,
    compute_embedding_stats,
    compute_fitness,
    evaluate_embedding_manifold,
    find_similar_words,
    get_cifar100_embeddings,
    load_visual_projection_model,
    plan_generator,
    run_planning_benchmark_suite,
    solve_analogy,
    synthesize_aligned_image_tensor,
    verify_plan,
)


def test_skipgram_manifold_integrity_and_cifar_coverage():
    """Verify 2D embeddings, finite values, shape == (523, 64), valid indices, and 1.0 CIFAR coverage."""
    vocab, embeddings = build_my_embeddings()
    assert isinstance(vocab, dict)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2, "Embeddings must be 2D"
    assert np.isfinite(embeddings).all(), "Embeddings must be strictly finite (no NaN/Inf)"
    assert len(vocab) == 523, f"Expected 523 words, got {len(vocab)}"
    assert embeddings.shape == (523, 64), f"Expected (523, 64), got {embeddings.shape}"
    assert set(vocab.values()) == set(range(523)), "Vocabulary indices must be contiguous 0..522"

    # Check 100% CIFAR-100 coverage
    missing_cifar = VALID_CIFAR_VOCAB - set(vocab.keys())
    assert len(missing_cifar) == 0, f"Missing CIFAR-100 words: {missing_cifar}"


def test_manifold_evaluation_metrics():
    """Verify quantitative manifold metrics (Recall@10, Prototype mAP@K, Analogy MRR@K)."""
    metrics = evaluate_embedding_manifold()
    assert metrics["passed_2d_test"] is True
    assert metrics["passed_finite_test"] is True
    assert metrics["passed_shape_matches_vocab"] is True
    assert metrics["passed_valid_indices"] is True
    assert metrics["cifar_coverage"] == pytest.approx(1.0)
    assert metrics["recall_at_10"] == pytest.approx(0.5423, abs=1e-3)
    assert metrics["hit_rate_at_10"] == pytest.approx(0.8933, abs=1e-3)
    assert metrics["map_at_k_prototypes"] > 0.70
    assert metrics["mrr_at_k_analogies"] > 0.55


def test_similarity_and_analogy_solver():
    """Verify cosine nearest neighbors and 3CosAdd analogy solving."""
    vocab, embeddings = build_my_embeddings()
    nodes = [w for w, _ in sorted(vocab.items(), key=lambda x: x[1])]

    sim_words = find_similar_words("maple_tree", nodes, embeddings, top_k=5)
    assert len(sim_words) == 5
    retrieved_tokens = {w for w, _ in sim_words}
    assert len(retrieved_tokens.intersection({"oak_tree", "pine_tree", "palm_tree", "willow_tree"})) >= 2

    analogy_res = solve_analogy("maple_tree", "oak_tree", "palm_tree", nodes, embeddings, top_k=5)
    assert len(analogy_res) == 5
    analogy_tokens = {w for w, _ in analogy_res}
    assert len(analogy_tokens.intersection({"pine_tree", "willow_tree"})) >= 1


def test_evolution_strategy_fitness_function():
    """Verify three-term fitness evaluation for (1+λ) Evolution Strategy."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((50, 64)).astype(np.float32)
    stats = compute_embedding_stats(embeddings)
    vec = embeddings[0]
    ctx_vecs = embeddings[1:6]
    ctx_weights = np.ones(5, dtype=np.float32) / 5.0
    neg_vecs = embeddings[10:20]
    anchor_vecs = embeddings[6:9] / (np.linalg.norm(embeddings[6:9], axis=1, keepdims=True) + 1e-10)
    weights = {"corpus": 0.5, "norm": 0.3, "anchor": 0.2}

    score = compute_fitness(vec, "test", ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, stats, weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_visual_projection_checkpoint_and_forward_pass():
    """Verify loading best_cifar100_projection.pth and projecting synthetic images to 64D space."""
    vision_model, ckpt = load_visual_projection_model(device="cpu")
    assert ckpt.get("epoch") == 13
    assert ckpt.get("val_similarity") == pytest.approx(0.5581, abs=1e-3)

    dummy_batch = torch.randn(2, 3, 224, 224)
    feats, proj = vision_model(dummy_batch)
    assert feats.shape == (2, 576)
    assert proj.shape == (2, 64)
    assert torch.isfinite(proj).all()

    text_batch = torch.randn(2, 64)
    loss = compute_contrastive_loss(proj, text_batch, temperature=0.07)
    assert loss.ndim == 0
    assert float(loss.item()) > 0.0


def test_cifar_to_pddl_sanitization():
    """Verify CIFAR-100 compound and tree names sanitize cleanly to PDDL syntax."""
    assert clean_object_name("maple_tree") == "maple"
    assert clean_object_name("oak_tree") == "oak"
    assert clean_object_name("palm_tree") == "palm"
    assert clean_object_name("pine_tree") == "pine"
    assert clean_object_name("willow_tree") == "willow"
    assert clean_object_name("aquarium_fish") == "aquarium-fish"
    assert clean_object_name("sweet_pepper") == "sweet-pepper"
    assert clean_object_name("pickup_truck") == "pickup-truck"
    assert clean_object_name("dinosaur") == "dinosaur"


def test_pddl_domain_and_problem_parsing():
    """Verify PDDLParser extracts all 7 action schemas, 100 CIFAR items, 2 tools, and 349 init predicates."""
    actions = PDDLParser.parse_domain()
    assert len(actions) == 7
    assert set(actions.keys()) == {
        "walk-between-rooms",
        "pick-up",
        "put-down",
        "stack",
        "unstack",
        "slice-object",
        "take-photo",
    }

    objs, init_state, _ = PDDLParser.parse_problem()
    assert len(objs["item"] - {"knife", "dslr"}) == 100
    assert objs["tool"] == {"knife", "dslr"}
    assert len(init_state.predicates) == 349

    grounder = ActionGrounder(actions, objs)
    grounded = grounder.ground_all()
    assert len(grounded) == 42018


def test_multimodal_plan_generator_text_and_image():
    """Verify end-to-end symbolic text planning and sub-symbolic visual planning across all 9 scenarios."""
    suite = run_planning_benchmark_suite(include_visual=True)
    assert suite["text_plans_created"] == 9
    assert suite["image_plans_created"] == 9

    # Also verify graceful rejection of out-of-domain objects
    invalid_plan = plan_generator(
        input_data="alien_spaceship",
        initial_state={"(agent-at lab)", "(hand-empty)", "(at ?x outdoors)", "(whole ?x)", "(clear ?x)"},
        goal_state={"(at ?x lab)"},
    )
    assert invalid_plan is None
