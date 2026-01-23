"""
Simple benchmark comparing RLM with and without shared store.

Usage:
    python examples/store_benchmark.py --backend vllm --model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    python examples/store_benchmark.py --backend openai --model "gpt-4o-mini"
    python examples/store_benchmark.py --mock  # Test without real LLM

This generates N facts with hidden "needles" and asks the model to find them,
comparing behavior with store_mode='none' vs store_mode='shared'.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required for --mock mode


def generate_facts(
    n: int = 20,
    needle_positions: list[int] | None = None,
) -> tuple[str, list[int], str]:
    """
    Generate N facts requiring semantic reasoning (not keyword matching).

    Task: Find projects led by women - requires understanding names/gender.

    Returns:
        Tuple of (facts_text, needle_positions, question)
    """
    if needle_positions is None:
        needle_positions = sorted(random.sample(range(n), k=min(4, n)))

    # Facts with female project leaders (the needles)
    female_led = [
        "Project Aurora was spearheaded by Dr. Sarah Chen, achieving 40% cost reduction.",
        "The Mercury initiative, directed by Maria González, launched ahead of schedule.",
        "Dr. Emily Watson led the Quantum team to a breakthrough in error correction.",
        "The Atlas modernization, overseen by Jennifer Park, reduced latency by 60%.",
        "Rachel Morrison's leadership of Project Helix resulted in 3 new patents.",
    ]

    # Facts with male project leaders (distractors - can't just search for "project")
    male_led = [
        "Project Titan was managed by James Wilson, delivering $2M in savings.",
        "The Orion expansion, led by Michael Brown, added 50 new markets.",
        "Dr. Robert Kim directed the Fusion project to successful completion.",
        "The Neptune upgrade, overseen by David Chen, improved throughput 3x.",
        "Marcus Johnson's Apex initiative streamlined operations significantly.",
        "Project Horizon, managed by Christopher Lee, met all Q3 targets.",
    ]

    # Neutral facts (no project leader)
    neutral = [
        "Revenue increased by 15% in Q3 compared to the previous quarter.",
        "The new office building has 12 floors and a rooftop garden.",
        "Employee satisfaction scores improved by 8 points this year.",
        "Customer support resolved 95% of tickets within 24 hours.",
        "R&D spending accounts for 18% of the annual budget.",
        "The security audit found no critical vulnerabilities.",
        "Quarterly earnings exceeded analyst expectations by 3%.",
        "Energy consumption was reduced by 20% through efficiency upgrades.",
    ]

    facts = []
    f_idx, m_idx, n_idx = 0, 0, 0

    for i in range(n):
        if i in needle_positions:
            facts.append(f"Fact {i + 1}: {female_led[f_idx % len(female_led)]}")
            f_idx += 1
        elif random.random() < 0.4:
            facts.append(f"Fact {i + 1}: {male_led[m_idx % len(male_led)]}")
            m_idx += 1
        else:
            facts.append(f"Fact {i + 1}: {neutral[n_idx % len(neutral)]}")
            n_idx += 1

    question = "Which facts describe projects led by women? List each fact number, project name, and the woman's name."
    return "\n".join(facts), needle_positions, question


def run_benchmark(
    backend: str,
    backend_kwargs: dict,
    store_mode: str,
    prompt_preset: str,
    context: str,
    question: str,
    log_dir: str,
) -> dict:
    """
    Run RLM and return metrics.

    Returns dict with:
        - response: final answer
        - wall_time: total execution time
        - log_path: path to JSONL log
    """
    from rlm import RLM
    from rlm.logger import RLMLogger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    suffix = "shared" if store_mode == "shared" else "baseline"
    logger = RLMLogger(log_dir=str(log_dir), file_name=f"benchmark_{suffix}")

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=10,
        logger=logger,
        verbose=True,
        store_mode=store_mode,
        prompt_preset=prompt_preset,
    )

    prompt = f"{context}\n\n---\n\nQuestion: {question}"

    start = time.perf_counter()
    result = rlm.completion(prompt)
    wall_time = time.perf_counter() - start

    return {
        "response": result.response,
        "wall_time": wall_time,
        "log_path": logger.log_file_path,
    }


def parse_jsonl_metrics(log_file: str) -> dict:
    """
    Extract metrics from a JSONL log file.

    Returns dict with:
        - total_rlm_calls: number of LLM subcalls
        - total_batch_calls: number of batch operations
        - total_store_events: number of store operations
        - batch_prompts_total: total prompts across all batches
    """
    metrics = {
        "total_rlm_calls": 0,
        "total_batch_calls": 0,
        "total_store_events": 0,
        "batch_prompts_total": 0,
    }

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)

                # Check for iteration data
                if "code_blocks" in entry:
                    for block in entry.get("code_blocks", []):
                        result = block.get("result", {})
                        metrics["total_rlm_calls"] += len(result.get("rlm_calls", []))
                        metrics["total_store_events"] += len(result.get("store_events", []))

                        for batch in result.get("batch_calls", []):
                            metrics["total_batch_calls"] += 1
                            metrics["batch_prompts_total"] += batch.get("prompts_count", 0)
            except json.JSONDecodeError:
                continue

    return metrics


def print_comparison(baseline: dict, shared: dict) -> None:
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'Shared Store':<20}")
    print("-" * 70)

    print(f"{'Wall Time (s)':<30} {baseline['wall_time']:<20.2f} {shared['wall_time']:<20.2f}")
    print(f"{'LLM Subcalls':<30} {baseline['metrics']['total_rlm_calls']:<20} {shared['metrics']['total_rlm_calls']:<20}")
    print(f"{'Batch Calls':<30} {baseline['metrics']['total_batch_calls']:<20} {shared['metrics']['total_batch_calls']:<20}")
    print(f"{'Batch Prompts Total':<30} {baseline['metrics']['batch_prompts_total']:<20} {shared['metrics']['batch_prompts_total']:<20}")
    print(f"{'Store Events':<30} {baseline['metrics']['total_store_events']:<20} {shared['metrics']['total_store_events']:<20}")

    print("\n" + "-" * 70)
    print("RESPONSES")
    print("-" * 70)

    print(f"\nBaseline Response:\n{baseline['response'][:500]}...")
    print(f"\nShared Store Response:\n{shared['response'][:500]}...")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    if shared["metrics"]["total_batch_calls"] > baseline["metrics"]["total_batch_calls"]:
        print("Model used store.llm_map() for batching with store_mode='shared'")
    elif shared["metrics"]["total_batch_calls"] == 0:
        print("Model did NOT use store.llm_map() - may need prompt tuning")

    if shared["wall_time"] < baseline["wall_time"]:
        speedup = baseline["wall_time"] / shared["wall_time"]
        print(f"Shared store was {speedup:.1f}x faster")
    else:
        print("No speedup observed (model may not have used batching)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RLM store/batching")
    parser.add_argument("--backend", default="vllm", help="LLM backend (openai, vllm, etc.)")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", help="Model name")
    parser.add_argument("--api-key", default=None, help="API key (or use env var)")
    parser.add_argument("--base-url", default=None, help="Base URL for vLLM/compatible APIs")
    parser.add_argument("--num-facts", type=int, default=20, help="Number of facts to generate")
    parser.add_argument("--log-dir", default="visualizer/public/logs", help="Log directory")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--mock", action="store_true", help="Run with mock data (no LLM)")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline")
    parser.add_argument("--store-only", action="store_true", help="Only run store-enabled")
    parser.add_argument("--prompt-preset", default="default", choices=["default", "legacy"])
    args = parser.parse_args()

    # Generate test data
    print("Generating test data...")
    context, needle_positions, question = generate_facts(n=args.num_facts)

    print(f"Generated {args.num_facts} facts with needles at positions: {[p + 1 for p in needle_positions]}")
    print(f"Question: {question}\n")

    if args.mock:
        print("Mock mode - skipping actual LLM calls")
        print("To run with a real LLM, use --backend and --model flags")
        return

    # Setup backend kwargs
    backend_kwargs = {"model_name": args.model, "max_tokens": args.max_tokens}

    if args.api_key:
        backend_kwargs["api_key"] = args.api_key
    elif args.backend == "openai":
        backend_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    elif args.backend == "vllm":
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")

    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    elif args.backend == "vllm":
        # Default vLLM local server URL
        backend_kwargs["base_url"] = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    results = {}

    # Run baseline (store_mode="none")
    if not args.store_only:
        print("\n" + "=" * 70)
        print("Running BASELINE (store_mode='none')")
        print("=" * 70)

        baseline_result = run_benchmark(
            backend=args.backend,
            backend_kwargs=backend_kwargs,
            store_mode="none",
            prompt_preset=args.prompt_preset,
            context=context,
            question=question,
            log_dir=args.log_dir,
        )
        baseline_result["metrics"] = parse_jsonl_metrics(baseline_result["log_path"])
        results["baseline"] = baseline_result

    # Run shared store mode (store_mode="shared")
    if not args.baseline_only:
        print("\n" + "=" * 70)
        print("Running SHARED STORE (store_mode='shared')")
        print("=" * 70)

        shared_result = run_benchmark(
            backend=args.backend,
            backend_kwargs=backend_kwargs,
            store_mode="shared",
            prompt_preset=args.prompt_preset,
            context=context,
            question=question,
            log_dir=args.log_dir,
        )
        shared_result["metrics"] = parse_jsonl_metrics(shared_result["log_path"])
        results["shared"] = shared_result

    # Print comparison
    if "baseline" in results and "shared" in results:
        print_comparison(results["baseline"], results["shared"])
    elif "baseline" in results:
        print(f"\nBaseline results: {json.dumps(results['baseline']['metrics'], indent=2)}")
    elif "shared" in results:
        print(f"\nShared store results: {json.dumps(results['shared']['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
