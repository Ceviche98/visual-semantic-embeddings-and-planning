#!/usr/bin/env python3
"""
CLI Entry Point: Run Neuro-Symbolic Multimodal PDDL Planner (Text & Visual Inputs).

Supports running a custom object/goal scenario or executing the full 18-plan
benchmark suite (9 symbolic text-based plans + 9 sub-symbolic image-based plans).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visual_semantic_planner import (
    plan_generator,
    run_planning_benchmark_suite,
    synthesize_aligned_image_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute Neuro-Symbolic Multimodal PDDL Planning from Text or Visual Input."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the full 18-scenario multimodal benchmark suite (9 text + 9 visual plans).",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "image"],
        default="text",
        help="Input modality for single-scenario planning.",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="dinosaur",
        help="Target CIFAR-100 object name (e.g., 'dinosaur', 'crab', 'sweet_pepper').",
    )
    args = parser.parse_args()

    if args.benchmark:
        suite = run_planning_benchmark_suite(include_visual=True)
        print("=" * 80)
        print("NEURO-SYMBOLIC MULTIMODAL PDDL BENCHMARK SUITE")
        print("=" * 80)
        print(f"Text-Based Plans Created  : {suite['text_plans_created']}/{suite['text_plans_total']}")
        print(f"Image-Based Plans Created : {suite['image_plans_created']}/{suite['image_plans_total']}")
        print("-" * 80)
        for t_res, i_res in zip(suite["text_results"], suite["image_results"]):
            status_t = "PASS" if t_res["success"] else "FAIL"
            status_i = "PASS" if i_res["success"] else "FAIL"
            print(f"[{status_t} Text | {status_i} Image] {t_res['scenario']} ({t_res['steps']} steps)")
            print(f"    Plan: {' -> '.join(t_res['plan'] or [])}")
        print("=" * 80)
        return

    initial_state = {
        "(agent-at lab)",
        "(hand-empty)",
        "(at ?x outdoors)",
        "(whole ?x)",
        "(clear ?x)",
    }
    goal_state = {"(at ?x lab)"}

    if args.modality == "text":
        input_data = args.object
    else:
        input_data = synthesize_aligned_image_tensor(args.object)

    plan = plan_generator(
        input_data=input_data,
        initial_state=initial_state,
        goal_state=goal_state,
        verbose=True,
    )
    if plan is not None:
        print(f"Synthesized Plan ({len(plan)} steps): {plan}")
    else:
        print("No valid plan found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
