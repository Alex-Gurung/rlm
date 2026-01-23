#!/usr/bin/env python3
"""
OOLONG benchmark comparing baseline vs store-enabled RLM.

Usage:
    uv run python examples/oolong_benchmark.py
    uv run python examples/oolong_benchmark.py --num-items 5
    uv run python examples/oolong_benchmark.py --store-only
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
class OolongResult:
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


def normalize_text(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    boxed = re.search(r"\\\\boxed\\{([^}]+)\\}", text)
    if boxed:
        text = boxed.group(1)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def maybe_parse_list(text: str) -> list[str] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return [normalize_text(str(item)) for item in parsed]
    return None


def extract_context(sample: dict) -> str:
    if "prompt" in sample and isinstance(sample["prompt"], str):
        return sample["prompt"]
    if "input" in sample and isinstance(sample["input"], str):
        return sample["input"]
    if "context" in sample:
        return sample["context"]
    if "context_window_text" in sample:
        return sample["context_window_text"]
    if "query" in sample and "context" in sample:
        return sample["context"]
    raise ValueError(f"Unable to extract context from sample keys: {sorted(sample.keys())}")


def extract_question(sample: dict) -> str:
    for key in ("question", "query", "prompt"):
        if key in sample and isinstance(sample[key], str):
            return sample[key]
    raise ValueError(f"Unable to extract question from sample keys: {sorted(sample.keys())}")


def extract_answers(sample: dict) -> list[str]:
    for key in ("answers", "answer", "output", "label", "ground_truth"):
        if key in sample:
            value = sample[key]
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]
    if "extra_info" in sample and isinstance(sample["extra_info"], dict):
        extra = sample["extra_info"]
        if "answers" in extra:
            answers = extra["answers"]
            if isinstance(answers, list):
                return [str(item) for item in answers]
            return [str(answers)]
    raise ValueError("Unable to extract answers from sample")


def is_correct(response: str, answers: list[str], allow_substring: bool) -> bool:
    normalized_response = normalize_text(response)
    normalized_answers = [normalize_text(ans) for ans in answers]

    if any(ans == normalized_response for ans in normalized_answers):
        return True

    response_list = maybe_parse_list(response)
    if response_list is not None:
        return sorted(response_list) == sorted(normalized_answers)

    if allow_substring:
        return all(ans in normalized_response for ans in normalized_answers)

    return False


def parse_log_metrics(log_file: str) -> dict:
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


def run_oolong(
    backend: str,
    backend_kwargs: dict,
    store_enabled: bool,
    prompt_preset: str,
    samples: list[dict],
    log_dir: str,
    max_iterations: int,
    verbose: bool,
    allow_substring: bool,
    subagent_hint: bool,
    icl: bool,
) -> OolongResult:
    from rlm import RLM
    from rlm.logger import RLMLogger

    name = "store" if store_enabled else "baseline"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=log_dir, file_name=f"oolong_{name}")

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=max_iterations,
        logger=logger,
        verbose=verbose,
        store_mode="shared" if store_enabled else "none",
        prompt_preset=prompt_preset,
    )

    correct = 0
    total_time = 0.0

    for idx, sample in enumerate(samples):
        prompt = extract_context(sample)
        question = extract_question(sample)
        root_prompt = build_root_prompt(question, subagent_hint=subagent_hint, icl=icl)
        answers = extract_answers(sample)

        print(f"  [{idx + 1}/{len(samples)}] Expected: {answers}")
        start = time.perf_counter()
        result = rlm.completion(prompt, root_prompt=root_prompt)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        response = result.response.strip()
        ok = is_correct(response, answers, allow_substring)
        correct += int(ok)
        status = "✓" if ok else "✗"
        print(f"  [{idx + 1}/{len(samples)}] {status} ({elapsed:.1f}s) Got: {response[:120]}...")

    metrics = parse_log_metrics(logger.log_file_path)

    return OolongResult(
        name=name,
        num_items=len(samples),
        correct=correct,
        total_time=total_time,
        rlm_calls=metrics["rlm_calls"],
        batch_calls=metrics["batch_calls"],
        store_events=metrics["store_events"],
    )


def print_results(results: list[OolongResult]) -> None:
    print("\n")
    print("=" * 80)
    print("                      OOLONG BENCHMARK RESULTS")
    print("=" * 80)
    print()

    print(f"{'Metric':<25} ", end="")
    for r in results:
        print(f"{r.name.upper():<20} ", end="")
    print()
    print("-" * 80)

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
    if len(results) == 2:
        baseline, store = results[0], results[1]
        print("\nANALYSIS:")

        if store.batch_calls > baseline.batch_calls:
            print(f"  ✓ Store mode used batching ({store.batch_calls} batch calls)")
        else:
            print("  ✗ Store mode did not use batching")

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


def build_root_prompt(question: str, subagent_hint: bool, icl: bool) -> str:
    root_prompt = (
        f"Question: {question}\n\n"
        "Guidance for OOLONG-style transcripts:\n"
        "- The answer is almost never stated as an explicit total; you must compute it from the transcript.\n"
        "- The context uses episode delimiters like [START OF EPISODE] ... [END OF EPISODE].\n"
        "- The instruction preface also contains these delimiter strings; ignore any 'episode' segment that is very short or lacks dialogue.\n"
        "- Dialogue lines are formatted as '<Speaker>: <utterance>' (e.g., 'Matt: ...'); do NOT require a literal 'Name:' prefix.\n"
        "- Extract occurrences directly from speaker lines that mention the target, then aggregate in Python.\n"
        "- Do NOT count raw keyword frequencies (e.g., every 'magic' or 'spell'); count only explicit casting events.\n"
        "- Prefer patterns like \"Name: I cast <Spell>\" or \"<Name> casts <Spell>\"; ignore generic mentions.\n"
        "- Do NOT stop at a small sample; process all real episodes after filtering placeholders.\n"
        "- Do NOT guess. If you compute a value into a variable, use FINAL_VAR(variable_name) (not FINAL_VAR(expression)).\n"
        "- If you ask a sub-agent for JSON, insist on raw JSON only; if parsing fails, re-ask with stricter formatting.\n"
        "- Regex-only counting is too noisy here; you MUST use llm_query to confirm whether a line is a true spell-cast.\n"
        "- If you extract candidate lines with regex, you MUST validate ambiguous cases (e.g., 'cast' in non-spell context) with a sub-agent.\n"
        "- You will likely need llm_query over episode chunks to distinguish real spell-casts from chatter; do that early (iteration 0).\n"
        "- If you do not use llm_query at least once, your answer will be considered incorrect."
    )
    if subagent_hint:
        root_prompt += (
            "\n\nSub-agent hint: Use llm_query on small chunks and ask for a strict, machine-readable output "
            "(e.g., JSON list of matches with speaker and line). Aggregate across chunks in Python."
        )
    if icl:
        root_prompt += (
            "\n\nExample (pattern to follow):\n"
            "```repl\n"
            "episodes = re.findall(r\"\\[START OF EPISODE\\].*?\\[END OF EPISODE\\]\", context, re.DOTALL)\n"
            "episodes = [ep for ep in episodes if len(ep) > 1000 and ':' in ep]\n"
            "all_spells = []\n"
            "for ep in episodes[:2]:\n"
            "    result = llm_query(\"You are a sub-agent. Task: list every explicit spell-cast event in this transcript chunk. "
            "Return raw JSON array of {caster, spell, line}. Only include true spell casts. Context: \" + ep)\n"
            "    all_spells.extend(parse_json(result))\n"
            "count = len(all_spells)\n"
            "```\n"
            "FINAL_VAR(count)"
        )
    return root_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="OOLONG Benchmark: Baseline vs Store")
    parser.add_argument("--dataset", default="oolongbench/oolong-real")
    parser.add_argument("--config", default="dnd")
    parser.add_argument("--split", default="test")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--num-items", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--log-dir", default="visualizer/public/logs")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--store-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--substring-eval", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--subagent-hint", action="store_true")
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--prompt-preset", default="default", choices=["default", "legacy"])
    args = parser.parse_args()

    disable_progress_bar()

    print(f"Loading OOLONG dataset {args.dataset} ({args.split})...")
    if args.config:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: args.num_items]
    samples = [dataset[i] for i in indices]

    backend_kwargs = {"model_name": args.model, "max_tokens": args.max_tokens}
    if args.backend == "vllm":
        backend_kwargs["base_url"] = args.base_url or os.getenv(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")
    elif args.backend == "openai":
        backend_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    results: list[OolongResult] = []

    if not args.store_only:
        print("\n" + "=" * 60)
        print("Running BASELINE (store_mode='none')")
        print("=" * 60)
        results.append(
            run_oolong(
                args.backend,
                backend_kwargs,
                False,
                args.prompt_preset,
                samples,
                args.log_dir,
                args.max_iterations,
                args.verbose,
                args.substring_eval,
                args.subagent_hint,
                args.icl,
            )
        )

    if not args.baseline_only:
        print("\n" + "=" * 60)
        print("Running STORE-ENABLED (store_mode='shared')")
        print("=" * 60)
        results.append(
            run_oolong(
                args.backend,
                backend_kwargs,
                True,
                args.prompt_preset,
                samples,
                args.log_dir,
                args.max_iterations,
                args.verbose,
                args.substring_eval,
                args.subagent_hint,
                args.icl,
            )
        )

    print_results(results)


if __name__ == "__main__":
    main()
