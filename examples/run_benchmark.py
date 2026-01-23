#!/usr/bin/env python3
"""
Benchmark comparing RLM baseline (store_mode='none') vs shared store mode (store_mode='shared').

Usage:
    python examples/run_benchmark.py
    python examples/run_benchmark.py --num-facts 50
    python examples/run_benchmark.py --store-only
    python examples/run_benchmark.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class BenchmarkResult:
    name: str
    wall_time: float
    iterations: int
    rlm_calls: int
    batch_calls: int
    store_events: int
    answer: str
    correct_facts: set[int]
    found_facts: set[int]

    @property
    def accuracy(self) -> float:
        if not self.correct_facts:
            return 0.0
        return len(self.found_facts & self.correct_facts) / len(self.correct_facts)

    @property
    def precision(self) -> float:
        if not self.found_facts:
            return 0.0
        return len(self.found_facts & self.correct_facts) / len(self.found_facts)


def generate_task(n: int = 20, seed: int = None) -> tuple[str, set[int], str]:
    """Generate facts requiring semantic reasoning."""
    if seed is not None:
        random.seed(seed)

    needle_positions = set(sorted(random.sample(range(n), k=min(4, n))))

    female_led = [
        ("Aurora", "Dr. Sarah Chen", "achieving 40% cost reduction"),
        ("Mercury", "Maria González", "launched ahead of schedule"),
        ("Quantum", "Dr. Emily Watson", "breakthrough in error correction"),
        ("Atlas", "Jennifer Park", "reduced latency by 60%"),
        ("Helix", "Rachel Morrison", "resulted in 3 new patents"),
    ]

    male_led = [
        ("Titan", "James Wilson", "delivering $2M in savings"),
        ("Orion", "Michael Brown", "added 50 new markets"),
        ("Fusion", "Dr. Robert Kim", "successful completion"),
        ("Neptune", "David Chen", "improved throughput 3x"),
        ("Apex", "Marcus Johnson", "streamlined operations"),
        ("Horizon", "Christopher Lee", "met all Q3 targets"),
    ]

    neutral = [
        "Revenue increased by 15% in Q3.",
        "Employee satisfaction improved by 8 points.",
        "Customer support resolved 95% of tickets within 24 hours.",
        "R&D spending accounts for 18% of the annual budget.",
        "The security audit found no critical vulnerabilities.",
        "Quarterly earnings exceeded expectations by 3%.",
        "Energy consumption was reduced by 20%.",
        "The data warehouse processes 500TB daily.",
    ]

    facts = []
    f_idx, m_idx, n_idx = 0, 0, 0

    for i in range(n):
        if i in needle_positions:
            proj, leader, achievement = female_led[f_idx % len(female_led)]
            facts.append(f"Fact {i+1}: Project {proj} was led by {leader}, {achievement}.")
            f_idx += 1
        elif random.random() < 0.35:
            proj, leader, achievement = male_led[m_idx % len(male_led)]
            facts.append(f"Fact {i+1}: Project {proj} was managed by {leader}, {achievement}.")
            m_idx += 1
        else:
            facts.append(f"Fact {i+1}: {neutral[n_idx % len(neutral)]}")
            n_idx += 1

    context = "\n".join(facts)
    question = "Which facts describe projects led by women? Return ONLY the fact numbers as a comma-separated list, e.g., '3, 7, 12'."
    correct = {i + 1 for i in needle_positions}

    return context, correct, question


def extract_fact_numbers(answer: str) -> set[int]:
    """Extract fact numbers from answer text.

    Prioritizes FINAL_VAR/FINAL patterns, then looks for explicit answer lines.
    Falls back to extracting all numbers only if no structured answer found.
    """
    import re

    # First, look for FINAL_VAR(...) or FINAL(...) patterns
    final_match = re.search(r'FINAL(?:_VAR)?\(([^)]+)\)', answer)
    if final_match:
        content = final_match.group(1)
        numbers = re.findall(r'\b(\d+)\b', content)
        return {int(n) for n in numbers if 1 <= int(n) <= 100}

    # Look for explicit answer lines like "Thus, the fact numbers are: 1, 4, 8, 9."
    answer_patterns = [
        r'fact numbers[^:]*:\s*([\d,\s]+)',
        r'facts describing[^:]*:\s*([\d,\s]+)',
        r'answer[^:]*:\s*([\d,\s]+)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            content = match.group(1)
            numbers = re.findall(r'\b(\d+)\b', content)
            if numbers:
                return {int(n) for n in numbers if 1 <= int(n) <= 100}

    # Fallback: extract all numbers (original behavior)
    numbers = re.findall(r'\b(\d+)\b', answer)
    return {int(n) for n in numbers if 1 <= int(n) <= 100}


def parse_log_metrics(log_dir: str) -> dict:
    """Extract metrics from JSONL logs."""
    metrics = {"rlm_calls": 0, "batch_calls": 0, "store_events": 0, "iterations": 0}

    log_path = Path(log_dir)
    for jsonl_file in log_path.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "iteration":
                        metrics["iterations"] += 1
                        for block in entry.get("code_blocks", []):
                            result = block.get("result", {})
                            metrics["rlm_calls"] += len(result.get("rlm_calls", []))
                            metrics["store_events"] += len(result.get("store_events", []))
                            for batch in result.get("batch_calls", []):
                                metrics["batch_calls"] += 1
                except json.JSONDecodeError:
                    continue
    return metrics


def run_single(
    backend: str,
    backend_kwargs: dict,
    store_mode: str,
    context: str | dict,
    question: str,
    correct_facts: set[int],
    log_base: str,
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        context: Either a string (single context) or dict (file-based context)
    """
    from rlm import RLM
    from rlm.logger import RLMLogger

    name = "shared" if store_mode == "shared" else "baseline"
    log_dir = f"{log_base}/{name}_{int(time.time())}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = RLMLogger(log_dir=log_dir)

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=10,
        logger=logger,
        verbose=True,
        store_mode=store_mode,
    )

    start = time.perf_counter()
    result = rlm.completion(context, root_prompt=question)
    wall_time = time.perf_counter() - start

    metrics = parse_log_metrics(log_dir)
    found_facts = extract_fact_numbers(result.response)

    return BenchmarkResult(
        name=name,
        wall_time=wall_time,
        iterations=metrics["iterations"],
        rlm_calls=metrics["rlm_calls"],
        batch_calls=metrics["batch_calls"],
        store_events=metrics["store_events"],
        answer=result.response,
        correct_facts=correct_facts,
        found_facts=found_facts,
    )


def print_results(results: list[BenchmarkResult], correct_facts: set[int]):
    """Pretty print benchmark results."""
    print("\n")
    print("=" * 80)
    print("                         BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nCorrect answer: Facts {sorted(correct_facts)}")
    print()

    # Header
    print(f"{'Metric':<25} ", end="")
    for r in results:
        print(f"{r.name.upper():<20} ", end="")
    print()
    print("-" * 80)

    # Metrics
    rows = [
        ("Wall Time (s)", [f"{r.wall_time:.2f}" for r in results]),
        ("Iterations", [str(r.iterations) for r in results]),
        ("LLM Sub-calls", [str(r.rlm_calls) for r in results]),
        ("Batch Calls", [str(r.batch_calls) for r in results]),
        ("Store Events", [str(r.store_events) for r in results]),
        ("Facts Found", [str(sorted(r.found_facts)) for r in results]),
        ("Recall", [f"{r.accuracy:.0%}" for r in results]),
        ("Precision", [f"{r.precision:.0%}" for r in results]),
    ]

    for label, values in rows:
        print(f"{label:<25} ", end="")
        for v in values:
            print(f"{v:<20} ", end="")
        print()

    print("-" * 80)

    # Analysis
    print("\nANALYSIS:")

    if len(results) == 2:
        baseline, store = results[0], results[1]

        if store.batch_calls > baseline.batch_calls:
            print(f"  ✓ Store mode used batching ({store.batch_calls} batch calls)")
        else:
            print(f"  ✗ Store mode did not use batching")

        if store.wall_time < baseline.wall_time:
            speedup = baseline.wall_time / store.wall_time
            print(f"  ✓ Store mode was {speedup:.1f}x faster")
        else:
            slowdown = store.wall_time / baseline.wall_time
            print(f"  ✗ Store mode was {slowdown:.1f}x slower")

        if store.accuracy >= baseline.accuracy:
            print(f"  ✓ Store mode accuracy: {store.accuracy:.0%} (baseline: {baseline.accuracy:.0%})")
        else:
            print(f"  ✗ Store mode accuracy dropped: {store.accuracy:.0%} vs {baseline.accuracy:.0%}")

    print()
    print("=" * 80)


def llm_judge_score(answer: str, correct_facts: set[int], judge_client) -> dict:
    """Use LLM-as-a-judge to evaluate the answer quality.

    Returns dict with:
        - score: 0-10 rating
        - correct_identified: list of correctly identified facts
        - false_positives: list of incorrectly identified facts
        - reasoning: judge's explanation
    """
    prompt = f"""You are evaluating an AI's answer to the question: "Which facts describe projects led by women?"

The CORRECT answer is: Facts {sorted(correct_facts)}

The AI's answer was:
{answer}

Please evaluate:
1. Which correct facts did the AI identify? (should be from {sorted(correct_facts)})
2. Which facts did the AI incorrectly identify as led by women (false positives)?
3. Overall score from 0-10 (10 = perfect, identified all correct facts with no false positives)

Respond in JSON format:
{{"score": <0-10>, "correct_identified": [<list of fact numbers>], "false_positives": [<list of fact numbers>], "reasoning": "<brief explanation>"}}
"""
    response = judge_client.completion([{"role": "user", "content": prompt}])

    # Parse JSON from response
    import re
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback if parsing fails
    return {"score": -1, "correct_identified": [], "false_positives": [], "reasoning": "Failed to parse judge response"}


def main():
    parser = argparse.ArgumentParser(description="RLM Store Benchmark")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--num-facts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--store-only", action="store_true")
    parser.add_argument("--use-files", action="store_true", help="Load context as chunked files instead of single string")
    parser.add_argument("--chunk-size", type=int, default=5, help="Facts per chunk when using --use-files")
    parser.add_argument("--judge", action="store_true", help="Use LLM-as-a-judge for evaluation")
    parser.add_argument("--judge-model", default=None, help="Model for LLM judge (defaults to same as --model)")
    parser.add_argument("--judge-backend", default=None, help="Backend for LLM judge (defaults to same as --backend)")
    args = parser.parse_args()

    # Generate task
    print(f"Generating {args.num_facts} facts (seed={args.seed})...")
    context_str, correct_facts, question = generate_task(args.num_facts, args.seed)
    print(f"Correct answers: Facts {sorted(correct_facts)}")
    print(f"Question: {question}\n")

    # Convert to file-based context if requested
    if args.use_files:
        facts = context_str.strip().split('\n')
        chunk_size = args.chunk_size
        context = {}
        for i in range(0, len(facts), chunk_size):
            chunk_facts = facts[i:i + chunk_size]
            chunk_name = f"chunk_{i // chunk_size + 1}.txt"
            context[chunk_name] = '\n'.join(chunk_facts)
        context["metadata.json"] = {
            "total_facts": len(facts),
            "chunks": len(context) - 1,
            "task": "Find facts describing projects led by women"
        }
        print(f"Using file-based context: {list(context.keys())}\n")
    else:
        context = context_str

    # Setup backend
    backend_kwargs = {"model_name": args.model}
    if args.backend == "vllm":
        backend_kwargs["base_url"] = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")
    elif args.backend == "openai":
        backend_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    # Setup judge client if requested
    judge_client = None
    if args.judge:
        from rlm.clients import get_client
        judge_backend = args.judge_backend or args.backend
        judge_model = args.judge_model or args.model
        judge_kwargs = {"model_name": judge_model}
        if judge_backend == "vllm":
            judge_kwargs["base_url"] = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            judge_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")
        elif judge_backend == "openai":
            judge_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        judge_client = get_client(judge_backend, judge_kwargs)
        print(f"Using LLM judge: {judge_model}\n")

    results = []

    # Run baseline (store_mode="none")
    if not args.store_only:
        print("\n" + "=" * 60)
        print("Running BASELINE (store_mode='none')")
        print("=" * 60)
        results.append(run_single(
            args.backend, backend_kwargs, "none",
            context, question, correct_facts, args.log_dir
        ))

    # Run shared store mode (store_mode="shared")
    if not args.baseline_only:
        print("\n" + "=" * 60)
        print("Running SHARED STORE (store_mode='shared')")
        print("=" * 60)
        results.append(run_single(
            args.backend, backend_kwargs, "shared",
            context, question, correct_facts, args.log_dir
        ))

    # LLM-as-a-judge evaluation
    judge_scores = {}
    if judge_client and results:
        print("\n" + "=" * 60)
        print("Running LLM-as-a-Judge Evaluation")
        print("=" * 60)
        for r in results:
            print(f"\nJudging {r.name}...")
            score = llm_judge_score(r.answer, correct_facts, judge_client)
            judge_scores[r.name] = score
            print(f"  Score: {score['score']}/10")
            print(f"  Correct identified: {score['correct_identified']}")
            print(f"  False positives: {score['false_positives']}")
            print(f"  Reasoning: {score['reasoning']}")

    # Print comparison
    print_results(results, correct_facts)

    # Print judge summary if available
    if judge_scores:
        print("\nLLM JUDGE SUMMARY:")
        print("-" * 40)
        for name, score in judge_scores.items():
            print(f"  {name.upper()}: {score['score']}/10")


if __name__ == "__main__":
    main()
