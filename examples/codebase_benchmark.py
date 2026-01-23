#!/usr/bin/env python3
"""
Codebase understanding benchmark using OpenRLHF as the target.

Tests the RLM's ability to explore a real codebase and answer fact-finding questions.
Uses LLM-as-a-judge for evaluation since answers are open-ended.

Usage:
    python examples/codebase_benchmark.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    python examples/codebase_benchmark.py --question 0  # Run specific question
    python examples/codebase_benchmark.py --list  # List all questions
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Fact-finding questions about OpenRLHF codebase
# Each has a question and key facts the answer should contain
# Questions are designed to require cross-file analysis and specific detail finding
QUESTIONS = [
    # --- EASY: Single file, straightforward ---
    {
        "id": "training_methods",
        "question": "What different training methods/algorithms does OpenRLHF support? List them with a brief description of each.",
        "key_facts": ["PPO", "DPO", "KTO", "SFT", "reward model", "knowledge distillation"],
        "difficulty": "easy",
        "files_hint": "cli/train_*.py",
    },
    # --- MEDIUM: Requires reading specific implementation ---
    {
        "id": "advantage_estimators",
        "question": "What are ALL the advantage estimation methods supported in PPO training? For each method, explain when gamma is forced to 1.0 and why.",
        "key_facts": ["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo", "gamma", "1.0"],
        "difficulty": "medium",
        "files_hint": "trainer/ppo_utils/experience_maker.py, cli/train_ppo_ray.py",
    },
    {
        "id": "kto_vs_dpo",
        "question": "Compare the KTO loss and DPO loss implementations. What additional parameters does KTO require that DPO doesn't? What is the role of the KL term in KTO?",
        "key_facts": ["desirable_weight", "undesirable_weight", "world_size", "KL", "all_reduce", "policy_KL_logps", "reference_KL_logps"],
        "difficulty": "medium",
        "files_hint": "models/loss.py",
    },
    {
        "id": "reward_clipping",
        "question": "What is the default reward clip range in PPO training? Where in the code is reward clipping applied, and what function performs it?",
        "key_facts": ["-10", "10", "reward_clip_range", "compute_reward", "clamp"],
        "difficulty": "medium",
        "files_hint": "cli/train_ppo_ray.py, models/utils.py",
    },
    # --- HARD: Cross-file, requires synthesizing multiple sources ---
    {
        "id": "mc_reward_computation",
        "question": "Explain the Marginal Contribution (MC) reward computation system. What are the key parameters in MCConfig (group_size, quota, n_trials) and what do they control? What is the default streaming_batch_size?",
        "key_facts": ["group_size", "quota", "n_trials", "aggregation_builder", "reward_fn", "streaming_batch_size", "0", "64"],
        "difficulty": "hard",
        "files_hint": "trainer/ppo_utils/mc/config.py, trainer/ppo_utils/mc/computation.py",
    },
    {
        "id": "experience_data_flow",
        "question": "Trace the data flow in PPO experience generation: What class generates experiences? What fields does the Experience dataclass contain? How does make_experience_batch combine multiple experiences?",
        "key_facts": ["RemoteExperienceMaker", "Experience", "sequences", "action_log_probs", "values", "returns", "advantages", "attention_mask", "concat_experiences"],
        "difficulty": "hard",
        "files_hint": "trainer/ppo_utils/experience_maker.py, trainer/ppo_utils/replay_buffer.py",
    },
    {
        "id": "label_smoothing_ipo",
        "question": "The DPO loss supports both standard DPO and IPO variants. What is the exact formula difference? What paper is cited for IPO in the code comments? What does label_smoothing do?",
        "key_facts": ["ipo", "label_smoothing", "logsigmoid", "(logits - 1/(2*beta))^2", "2310.12036", "cdpo"],
        "difficulty": "hard",
        "files_hint": "models/loss.py",
    },
    {
        "id": "vllm_dispatch",
        "question": "How does the experience maker dispatch prompts to vLLM engines? What is the role of _dispatch_prompts_to_vllm? How are responses collected and processed into Experience objects?",
        "key_facts": ["_dispatch_prompts_to_vllm", "batch_vllm_engine_call", "_process_response_into_experience", "ray", "async"],
        "difficulty": "hard",
        "files_hint": "trainer/ppo_utils/experience_maker.py, trainer/ray/vllm_engine.py",
    },
    # --- VERY HARD: Requires deep cross-file understanding ---
    {
        "id": "gae_implementation",
        "question": "Find the exact GAE (Generalized Advantage Estimation) implementation. What are the variable names used for the discount factor and GAE lambda? Show the core loop formula with the actual variable names from the code.",
        "key_facts": ["gamma", "lambd", "lastgaelam", "delta", "nextvalues", "rewards[:, t]", "values[:, t]"],
        "difficulty": "very_hard",
        "files_hint": "trainer/ppo_utils/experience_maker.py",
    },
    {
        "id": "distributed_kl_sync",
        "question": "In KTO training, how is the KL divergence synchronized across distributed workers? What PyTorch distributed operation is used? Why is the result clamped to min=0?",
        "key_facts": ["dist.all_reduce", "ReduceOp.SUM", "world_size", "clamp(min=0)", "detach"],
        "difficulty": "very_hard",
        "files_hint": "models/loss.py, trainer/kto_trainer.py",
    },
    {
        "id": "buffer_balancing",
        "question": "How does balance_experiences work to distribute experiences across data parallel workers? What sorting is applied? What is the purpose of reordering by sequence length?",
        "key_facts": ["balance_experiences", "sort", "total_length", "data parallel", "world_size", "rank"],
        "difficulty": "very_hard",
        "files_hint": "trainer/ppo_utils/replay_buffer.py",
    },
    {
        "id": "full_ppo_pipeline",
        "question": "Describe the complete PPO training loop: What are the 4 main Ray actor groups created? How do they interact during one training iteration? What is the role of each group?",
        "key_facts": ["actor", "critic", "reward", "reference", "RayActorGroup", "vllm_engines", "generate", "compute_reward", "update"],
        "difficulty": "very_hard",
        "files_hint": "cli/train_ppo_ray.py, trainer/ppo_trainer.py, trainer/ray/launcher.py",
    },
]


@dataclass
class BenchmarkResult:
    question_id: str
    question: str
    answer: str
    wall_time: float
    iterations: int
    llm_calls: int
    judge_score: float | None = None
    judge_reasoning: str | None = None
    key_facts_found: list[str] | None = None


def load_codebase_as_files(base_path: str, extensions: list[str] = [".py"]) -> dict[str, str]:
    """Load codebase files into a dict for file-based context."""
    files = {}
    base = Path(base_path)

    for ext in extensions:
        for filepath in base.rglob(f"*{ext}"):
            # Get relative path from base
            rel_path = filepath.relative_to(base)
            try:
                content = filepath.read_text()
                # Skip very large files
                if len(content) < 50000:
                    files[str(rel_path)] = content
            except Exception:
                pass

    return files


def run_question(
    question_data: dict,
    backend: str,
    backend_kwargs: dict,
    codebase_path: str,
    log_dir: str,
    store_mode: str = "shared",
    prompt_preset: str = "default",
) -> BenchmarkResult:
    """Run a single question through the RLM."""
    from rlm import RLM
    from rlm.logger import RLMLogger

    # Load codebase as files
    files = load_codebase_as_files(codebase_path)

    # Add metadata
    files["_metadata.json"] = json.dumps({
        "codebase": "OpenRLHF",
        "description": "Open-source RLHF training framework",
        "total_files": len(files),
        "hint": f"Relevant files might be in: {question_data.get('files_hint', 'unknown')}",
    }, indent=2)

    # Setup logging
    question_id = question_data["id"]
    run_log_dir = f"{log_dir}/{question_id}_{int(time.time())}"
    Path(run_log_dir).mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=run_log_dir)

    # Create RLM
    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=15,
        logger=logger,
        verbose=True,
        store_mode=store_mode,
        prompt_preset=prompt_preset,
    )

    # Run completion
    start = time.perf_counter()
    result = rlm.completion(files, root_prompt=question_data["question"])
    wall_time = time.perf_counter() - start

    # Parse metrics from log
    metrics = parse_log_metrics(run_log_dir)

    return BenchmarkResult(
        question_id=question_id,
        question=question_data["question"],
        answer=result.response,
        wall_time=wall_time,
        iterations=metrics["iterations"],
        llm_calls=metrics["llm_calls"],
    )


def parse_log_metrics(log_dir: str) -> dict:
    """Extract metrics from JSONL logs."""
    metrics = {"llm_calls": 0, "iterations": 0}

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
                            metrics["llm_calls"] += len(result.get("rlm_calls", []))
                except json.JSONDecodeError:
                    continue
    return metrics


def llm_judge(
    question: str,
    answer: str,
    key_facts: list[str],
    judge_client,
) -> dict:
    """Use LLM-as-a-judge to evaluate answer quality."""
    prompt = f"""You are evaluating an AI's answer about a codebase (OpenRLHF - an RLHF training framework).

QUESTION: {question}

KEY FACTS the answer should mention (not all required, but more is better):
{json.dumps(key_facts, indent=2)}

AI'S ANSWER:
{answer}

Please evaluate:
1. Accuracy: Is the information correct? (0-10)
2. Completeness: How many key facts were covered? (0-10)
3. Clarity: Is the answer clear and well-organized? (0-10)
4. Which key facts were found in the answer?

Respond in JSON:
{{"accuracy": <0-10>, "completeness": <0-10>, "clarity": <0-10>, "overall": <0-10>, "key_facts_found": [<list of found facts>], "reasoning": "<brief explanation>"}}
"""

    response = judge_client.completion([{"role": "user", "content": prompt}])

    # Parse JSON
    import re
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"overall": -1, "reasoning": "Failed to parse judge response", "key_facts_found": []}


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("                    CODEBASE UNDERSTANDING BENCHMARK")
    print("=" * 80)

    print(f"\n{'Question':<20} {'Time':<10} {'Iters':<8} {'LLM Calls':<12} {'Judge':<8}")
    print("-" * 80)

    for r in results:
        judge_str = f"{r.judge_score:.1f}/10" if r.judge_score is not None else "N/A"
        print(f"{r.question_id:<20} {r.wall_time:<10.1f}s {r.iterations:<8} {r.llm_calls:<12} {judge_str:<8}")

    print("-" * 80)

    # Summary stats
    total_time = sum(r.wall_time for r in results)
    avg_score = sum(r.judge_score for r in results if r.judge_score is not None) / len([r for r in results if r.judge_score is not None]) if any(r.judge_score is not None for r in results) else 0

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Average judge score: {avg_score:.1f}/10")

    # Show answers
    print("\n" + "=" * 80)
    print("ANSWERS")
    print("=" * 80)

    for r in results:
        print(f"\n### {r.question_id}")
        print(f"Q: {r.question}")
        print(f"A: {r.answer[:500]}..." if len(r.answer) > 500 else f"A: {r.answer}")
        if r.judge_reasoning:
            print(f"Judge: {r.judge_reasoning}")
        if r.key_facts_found:
            print(f"Key facts found: {r.key_facts_found}")


def main():
    parser = argparse.ArgumentParser(description="Codebase Understanding Benchmark")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--codebase", default="/mnt/disk/OpenRLHF/openrlhf")
    parser.add_argument("--log-dir", default="./logs/codebase")
    parser.add_argument("--question", type=int, default=None, help="Run specific question by index")
    parser.add_argument("--list", action="store_true", help="List all questions")
    parser.add_argument("--judge", action="store_true", help="Use LLM-as-a-judge")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--store-mode", default="shared", choices=["shared", "none"])
    parser.add_argument("--prompt-preset", default="default", choices=["default", "legacy"])
    args = parser.parse_args()

    if args.list:
        print("Available questions:")
        for i, q in enumerate(QUESTIONS):
            print(f"  {i}: [{q['difficulty']}] {q['id']}: {q['question'][:60]}...")
        return

    # Setup backend
    backend_kwargs = {"model_name": args.model}
    if args.backend == "vllm":
        backend_kwargs["base_url"] = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")
    elif args.backend == "openai":
        backend_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    # Setup judge
    judge_client = None
    if args.judge:
        from rlm.clients import get_client
        judge_model = args.judge_model or args.model
        judge_kwargs = {"model_name": judge_model}
        if args.backend == "vllm":
            judge_kwargs["base_url"] = backend_kwargs.get("base_url")
            judge_kwargs["api_key"] = backend_kwargs.get("api_key")
        judge_client = get_client(args.backend, judge_kwargs)
        print(f"Using LLM judge: {judge_model}\n")

    # Select questions
    if args.question is not None:
        questions = [QUESTIONS[args.question]]
    else:
        questions = QUESTIONS[:3]  # Default to first 3 (easier ones)

    print(f"Running {len(questions)} question(s) on {args.codebase}")
    print(f"Model: {args.model}")
    print(f"Store mode: {args.store_mode}")
    print(f"Prompt preset: {args.prompt_preset}\n")

    results = []
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q['id']} ({q['difficulty']})")
        print(f"{'='*60}")
        print(f"{q['question']}\n")

        result = run_question(
            q, args.backend, backend_kwargs,
            args.codebase, args.log_dir, args.store_mode, args.prompt_preset
        )

        # Judge if requested
        if judge_client:
            print("\nJudging answer...")
            judge_result = llm_judge(q["question"], result.answer, q["key_facts"], judge_client)
            result.judge_score = judge_result.get("overall", -1)
            result.judge_reasoning = judge_result.get("reasoning", "")
            result.key_facts_found = judge_result.get("key_facts_found", [])
            print(f"Score: {result.judge_score}/10 - {result.judge_reasoning}")

        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
