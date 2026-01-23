#!/usr/bin/env python3
"""
RULER NIAH benchmark comparing baseline vs store-enabled RLM.

Usage:
    uv run python examples/niah_benchmark.py
    uv run python examples/niah_benchmark.py --num-items 5
    uv run python examples/niah_benchmark.py --store-only
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from datasets import load_dataset
    from datasets.utils.logging import disable_progress_bar
except ImportError:
    raise SystemExit("Missing: datasets. Install with: uv pip install datasets")


@dataclass
class NIAHResult:
    name: str
    num_items: int
    correct: int
    total_time: float
    rlm_calls: int
    batch_calls: int
    store_events: int

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.num_items)

    @property
    def avg_time(self) -> float:
        return self.total_time / max(1, self.num_items)


def extract_text(result: object) -> str:
    if isinstance(result, str):
        return result
    response = getattr(result, "response", None)
    if isinstance(response, str):
        return response
    raise TypeError(f"Unexpected completion type: {type(result)}")


def ruler_match(response: str, answers: list[str]) -> bool:
    return all(answer in response for answer in answers)


def parse_log_metrics(log_file: str) -> dict:
    """Extract metrics from a JSONL log file."""
    metrics = {"rlm_calls": 0, "batch_calls": 0, "store_events": 0}

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("type") == "iteration":
                    for block in entry.get("code_blocks", []):
                        result = block.get("result", {})
                        metrics["rlm_calls"] += len(result.get("rlm_calls", []))
                        metrics["store_events"] += len(result.get("store_events", []))
                        for _ in result.get("batch_calls", []):
                            metrics["batch_calls"] += 1
            except json.JSONDecodeError:
                continue
    return metrics


def run_niah(
    backend: str,
    backend_kwargs: dict,
    store_prompt: bool,
    samples: list,
    log_dir: str,
    max_iterations: int,
    verbose: bool,
) -> NIAHResult:
    """Run NIAH benchmark with given configuration."""
    from rlm import RLM
    from rlm.logger import RLMLogger

    name = "store" if store_prompt else "baseline"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=log_dir, file_name=f"niah_{name}")

    ROOT_PROMPT = (
        "You are using RLM. The benchmark prompt is in `context`. "
        "Use the REPL to search/slice context. "
        "When you find the answer, output: FINAL(your_answer)"
    )

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=max_iterations,
        logger=logger,
        verbose=verbose,
        store_prompt=store_prompt,
    )

    correct = 0
    total_time = 0.0

    for idx, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        extra = sample.get("extra_info") or {}
        ground_truth = extra.get("ground_truth") or {}
        answers = ground_truth.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]

        print(f"  [{idx+1}/{len(samples)}] Expected: {answers}")

        start = time.perf_counter()
        result = rlm.completion(prompt, root_prompt=ROOT_PROMPT)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        response = extract_text(result).strip()
        is_correct = ruler_match(response, answers)
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}/{len(samples)}] {status} ({elapsed:.1f}s) Got: {response[:100]}...")

    metrics = parse_log_metrics(logger.log_file_path)

    return NIAHResult(
        name=name,
        num_items=len(samples),
        correct=correct,
        total_time=total_time,
        rlm_calls=metrics["rlm_calls"],
        batch_calls=metrics["batch_calls"],
        store_events=metrics["store_events"],
    )


def print_results(results: list[NIAHResult]):
    """Pretty print benchmark results."""
    print("\n")
    print("=" * 80)
    print("                      NIAH BENCHMARK RESULTS")
    print("=" * 80)
    print()

    # Header
    print(f"{'Metric':<25} ", end="")
    for r in results:
        print(f"{r.name.upper():<20} ", end="")
    print()
    print("-" * 80)

    # Metrics
    rows = [
        ("Accuracy", [f"{r.accuracy:.0%}" for r in results]),
        ("Correct / Total", [f"{r.correct}/{r.num_items}" for r in results]),
        ("Total Time (s)", [f"{r.total_time:.1f}" for r in results]),
        ("Avg Time (s)", [f"{r.avg_time:.1f}" for r in results]),
        ("LLM Sub-calls", [str(r.rlm_calls) for r in results]),
        ("Batch Calls", [str(r.batch_calls) for r in results]),
        ("Store Events", [str(r.store_events) for r in results]),
    ]

    for label, values in rows:
        print(f"{label:<25} ", end="")
        for v in values:
            print(f"{v:<20} ", end="")
        print()

    print("-" * 80)

    # Analysis
    if len(results) == 2:
        baseline, store = results[0], results[1]
        print("\nANALYSIS:")

        if store.batch_calls > baseline.batch_calls:
            print(f"  ✓ Store mode used batching ({store.batch_calls} batch calls)")
        else:
            print(f"  ✗ Store mode did not use batching")

        if store.total_time < baseline.total_time:
            speedup = baseline.total_time / store.total_time
            print(f"  ✓ Store mode was {speedup:.1f}x faster")
        else:
            slowdown = store.total_time / baseline.total_time
            print(f"  ✗ Store mode was {slowdown:.1f}x slower")

        if store.accuracy >= baseline.accuracy:
            print(f"  ✓ Store accuracy: {store.accuracy:.0%} (baseline: {baseline.accuracy:.0%})")
        else:
            print(f"  ✗ Store accuracy dropped: {store.accuracy:.0%} vs {baseline.accuracy:.0%}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="NIAH Benchmark: Baseline vs Store")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--num-items", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category", default="niah_single_1")
    parser.add_argument("--context-lengths", default="", help="e.g., '4096,8192'")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--log-dir", default="visualizer/public/logs")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--store-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    disable_progress_bar()

    # Load dataset
    print(f"Loading RULER dataset (category={args.category})...")
    dataset = load_dataset("tonychenxyz/ruler-full", "plain", split="validation")

    # Filter
    context_lengths = [int(x) for x in args.context_lengths.split(",") if x.strip()] if args.context_lengths else None
    filtered = []
    for sample in dataset:
        category = sample.get("category", "")
        if args.category not in category:
            continue
        if context_lengths:
            if not any(str(length) in category for length in context_lengths):
                continue
        filtered.append(sample)

    if not filtered:
        raise ValueError(f"No samples matched category={args.category}")

    # Select samples
    rng = random.Random(args.seed)
    rng.shuffle(filtered)
    samples = filtered[:args.num_items]

    print(f"Selected {len(samples)} samples")

    # Setup backend
    backend_kwargs = {"model_name": args.model, "max_tokens": args.max_tokens}
    if args.backend == "vllm":
        backend_kwargs["base_url"] = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")
    elif args.backend == "openai":
        backend_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    results = []

    # Run baseline
    if not args.store_only:
        print("\n" + "=" * 60)
        print("Running BASELINE (store_prompt=False)")
        print("=" * 60)
        results.append(run_niah(
            args.backend, backend_kwargs, False,
            samples, args.log_dir, args.max_iterations, args.verbose
        ))

    # Run store-enabled
    if not args.baseline_only:
        print("\n" + "=" * 60)
        print("Running STORE-ENABLED (store_prompt=True)")
        print("=" * 60)
        results.append(run_niah(
            args.backend, backend_kwargs, True,
            samples, args.log_dir, args.max_iterations, args.verbose
        ))

    # Print comparison
    print_results(results)


if __name__ == "__main__":
    main()
