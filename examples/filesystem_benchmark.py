#!/usr/bin/env python3
"""
Filesystem benchmark for RLM behavior on codebases.

This is a lightweight, CodeQA-like benchmark that:
- Serializes a codebase into context
- Generates file-path retrieval tasks from regex patterns
- Evaluates baseline vs store_prompt suggestion
- Logs metrics (subcalls, batch calls, store events, worker spawns)

Usage:
  uv run python examples/filesystem_benchmark.py --repo /mnt/disk/OpenRLHF --backend vllm --model "Qwen/..." --base-url http://localhost:8000/v1
  uv run python examples/filesystem_benchmark.py --repo /mnt/disk/OpenRLHF --store-only
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rlm import RLM
from rlm.logger import RLMLogger

# File extensions to include
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp"}
DOC_EXTENSIONS = {".md", ".txt", ".rst"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}
ALL_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | CONFIG_EXTENSIONS

PATTERN_CANDIDATES: list[tuple[str, str]] = [
    ("PPO trainer", r"PPO|ppo"),
    ("reward model", r"reward_model|reward model|rm_model|rewardmodel"),
    ("rollout", r"rollout"),
    ("actor", r"\bactor\b|Actor"),
    ("critic", r"\bcritic\b|Critic"),
    ("optimizer", r"optimizer"),
    ("train step", r"train_step|training step|train_step\("),
    ("kl penalty", r"kl_penalty|KL"),
    ("checkpoint", r"checkpoint|ckpt"),
    ("dataset", r"dataset|data_loader|dataloader"),
]


@dataclass
class Task:
    label: str
    pattern: str
    expected_paths: list[str]


@dataclass
class RunResult:
    name: str
    num_tasks: int
    avg_f1: float
    avg_precision: float
    avg_recall: float
    wall_time: float
    log_path: str
    metrics: dict


def serialize_directory(
    root_path: str,
    max_files: int = 50,
    max_file_size: int = 10000,
    extensions: set[str] = ALL_EXTENSIONS,
) -> tuple[str, dict, list[str]]:
    """Serialize a directory into context string for RLM."""
    root = Path(root_path).resolve()
    files_content: list[str] = []
    file_list: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        rel_path = path.relative_to(root)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        if any(part in {"node_modules", "__pycache__", ".venv", "venv", "dist", "build"} for part in rel_path.parts):
            continue
        file_list.append(rel_path)
        if len(file_list) >= max_files:
            break

    for rel_path in file_list:
        full_path = root / rel_path
        try:
            content = full_path.read_text(errors="replace")
            if len(content) > max_file_size:
                content = content[:max_file_size] + f"\n... [TRUNCATED at {max_file_size} chars]"
            files_content.append(f"{'='*60}\n=== FILE: {rel_path} ===\n{'='*60}\n{content}\n")
        except Exception as e:
            files_content.append(f"=== FILE: {rel_path} ===\n[ERROR reading file: {e}]\n")

    header = f"""CODEBASE SNAPSHOT
Root: {root}
Total files: {len(file_list)}
Extensions: {', '.join(sorted(extensions))}

FILE LISTING:
{chr(10).join(f'  - {p}' for p in file_list)}

{'='*60}
FILE CONTENTS
{'='*60}
"""
    context = header + "\n".join(files_content)

    metadata = {
        "root": str(root),
        "file_count": len(file_list),
        "files": [str(p) for p in file_list],
        "total_chars": len(context),
    }

    return context, metadata, [str(p) for p in file_list]


def build_tasks(file_list: list[str], root_path: str, num_tasks: int, min_matches: int, max_matches: int) -> list[Task]:
    root = Path(root_path).resolve()
    tasks: list[Task] = []

    # Preload file contents for selected files only
    contents: dict[str, str] = {}
    for rel_path in file_list:
        full_path = root / rel_path
        try:
            contents[rel_path] = full_path.read_text(errors="replace")
        except Exception:
            contents[rel_path] = ""

    random.shuffle(PATTERN_CANDIDATES)

    for label, pattern in PATTERN_CANDIDATES:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        matches = [path for path, text in contents.items() if regex.search(text)]
        if min_matches <= len(matches) <= max_matches:
            tasks.append(Task(label=label, pattern=pattern, expected_paths=sorted(matches)))
        if len(tasks) >= num_tasks:
            break

    if not tasks:
        raise SystemExit("No suitable tasks found. Try increasing max_files or adjusting pattern range.")

    return tasks


def build_root_prompt(tasks: list[Task], store_enabled: bool) -> str:
    task_lines = []
    for i, task in enumerate(tasks, start=1):
        task_lines.append(
            f"{i}) Find file paths that contain the pattern for '{task.label}'. "
            f"Pattern: /{task.pattern}/. Answer with one path per line."
        )

    prefix = (
        "You are analyzing a codebase snapshot. Use the REPL to inspect the context. "
        "You may use rlm_worker() to parallelize across directories and apply_commit() to share findings. "
        if store_enabled
        else "You are analyzing a codebase snapshot. Use the REPL to inspect the context. "
    )

    return (
        prefix
        + "The context format is:\n"
        "- FILE LISTING: lines between 'FILE LISTING:' and the FILE CONTENTS header\n"
        "- FILE CONTENTS header: a line containing 'FILE CONTENTS' between two lines of '='\n"
        "- FILE CONTENTS blocks: each file starts with a line like '=== FILE: <path> ==='\n"
        "Prefer extracting paths from the FILE CONTENTS blocks (more reliable than parsing the listing). "
        "After executing any code, return answers to the questions below as plain text (no code blocks).\n"
        "If you build intermediate dicts, use stable keys like Q1/Q2/Q3 (avoid using raw pattern strings).\n"
        "Helpers available: extract_file_blocks(context), find_matches(blocks, patterns), format_answers(matches, keys).\n"
        "safe_append(d, key, value) is also available to avoid KeyError when accumulating lists.\n\n"
        "Questions:\n" + "\n".join(task_lines) + "\n\n"
        "Answer format: Provide a section per question labeled 'Q1', 'Q2', etc., followed by file paths."
    )


def extract_first_error(log_file: str) -> str | None:
    """Return the first stderr exception line found in the log, if any."""
    try:
        with open(log_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "iteration":
                    continue
                for block in entry.get("code_blocks", []):
                    result = block.get("result", {})
                    stderr = result.get("stderr", "")
                    if stderr:
                        for sline in stderr.splitlines():
                            sline = sline.strip()
                            if sline:
                                return sline
    except FileNotFoundError:
        return None
    return None


def extract_paths_from_response(response: str, file_list: Iterable[str]) -> set[str]:
    found = set()
    for path in file_list:
        if path in response:
            found.add(path)
    return found


def score_task(response: str, task: Task, file_list: list[str]) -> tuple[float, float, float]:
    predicted = extract_paths_from_response(response, file_list)
    expected = set(task.expected_paths)

    if not expected and not predicted:
        return 1.0, 1.0, 1.0

    tp = len(predicted & expected)
    precision = tp / max(1, len(predicted))
    recall = tp / max(1, len(expected))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def parse_log_metrics(log_file: str) -> dict:
    metrics = {
        "iterations": 0,
        "sub_calls": 0,
        "store_events": 0,
        "batch_calls": 0,
        "commit_events": 0,
        "worker_spawns": 0,
        "worker_completes": 0,
    }

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "iteration":
                metrics["iterations"] += 1
                for block in entry.get("code_blocks", []):
                    result = block.get("result", {})
                    metrics["sub_calls"] += len(result.get("rlm_calls", []))
                    metrics["store_events"] += len(result.get("store_events", []))
                    metrics["batch_calls"] += len(result.get("batch_calls", []))
                    metrics["commit_events"] += len(result.get("commit_events", []))
            elif entry.get("type") == "worker_spawn":
                metrics["worker_spawns"] += 1
            elif entry.get("type") == "worker_complete":
                metrics["worker_completes"] += 1
    return metrics


def run_benchmark(
    name: str,
    backend: str,
    backend_kwargs: dict,
    context: str,
    tasks: list[Task],
    file_list: list[str],
    log_dir: str,
    max_iterations: int,
    store_prompt: bool,
    environment_kwargs: dict | None = None,
) -> RunResult:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=log_dir, file_name=f"filesystem_{name}")

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        environment_kwargs=environment_kwargs or {},
        max_depth=3,
        max_iterations=max_iterations,
        logger=logger,
        verbose=True,
        store_mode="shared" if store_prompt else "none",
        task_name=f"filesystem_benchmark/{name}",
    )

    root_prompt = build_root_prompt(tasks, store_enabled=store_prompt)

    start = time.perf_counter()
    result = rlm.completion(context, root_prompt=root_prompt)
    wall_time = time.perf_counter() - start

    retry_log_path = None
    err_line = extract_first_error(logger.log_file_path)
    if err_line:
        retry_logger = RLMLogger(log_dir=log_dir, file_name=f"filesystem_{name}_retry")
        retry_rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment="local",
            environment_kwargs=environment_kwargs or {},
            max_depth=3,
            max_iterations=max_iterations,
            logger=retry_logger,
            verbose=True,
            store_mode="shared" if store_prompt else "none",
            task_name=f"filesystem_benchmark/{name}",
        )
        retry_prompt = (
            root_prompt
            + f"\n\nNote: Your previous code error was: {err_line}. Fix the error and retry."
        )
        retry_start = time.perf_counter()
        result = retry_rlm.completion(context, root_prompt=retry_prompt)
        wall_time += time.perf_counter() - retry_start
        retry_log_path = retry_logger.log_file_path

    response = result.response or ""
    precisions = []
    recalls = []
    f1s = []
    for task in tasks:
        precision, recall, f1 = score_task(response, task, file_list)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = parse_log_metrics(retry_log_path or logger.log_file_path)
    metrics["retried"] = 1 if retry_log_path else 0

    return RunResult(
        name=name,
        num_tasks=len(tasks),
        avg_f1=sum(f1s) / len(f1s),
        avg_precision=sum(precisions) / len(precisions),
        avg_recall=sum(recalls) / len(recalls),
        wall_time=wall_time,
        log_path=retry_log_path or logger.log_file_path,
        metrics=metrics,
    )


def print_report(results: list[RunResult], tasks: list[Task]) -> None:
    print("\n" + "=" * 80)
    print("FILESYSTEM BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Tasks: {len(tasks)}")
    for i, task in enumerate(tasks, start=1):
        print(f"  Q{i}: label='{task.label}' pattern=/{task.pattern}/ matches={len(task.expected_paths)}")

    print("\n" + "-" * 80)
    print(f"{'Metric':<25} " + " ".join(f"{r.name.upper():<18}" for r in results))
    print("-" * 80)
    rows = [
        ("Avg F1", [f"{r.avg_f1:.2f}" for r in results]),
        ("Avg Precision", [f"{r.avg_precision:.2f}" for r in results]),
        ("Avg Recall", [f"{r.avg_recall:.2f}" for r in results]),
        ("Wall Time (s)", [f"{r.wall_time:.2f}" for r in results]),
        ("Iterations", [str(r.metrics["iterations"]) for r in results]),
        ("Subcalls", [str(r.metrics["sub_calls"]) for r in results]),
        ("Batch Calls", [str(r.metrics["batch_calls"]) for r in results]),
        ("Store Events", [str(r.metrics["store_events"]) for r in results]),
        ("Commit Events", [str(r.metrics["commit_events"]) for r in results]),
        ("Worker Spawns", [str(r.metrics["worker_spawns"]) for r in results]),
        ("Worker Completes", [str(r.metrics["worker_completes"]) for r in results]),
        ("Retried", [str(r.metrics.get("retried", 0)) for r in results]),
    ]

    for label, values in rows:
        print(f"{label:<25} " + " ".join(f"{v:<18}" for v in values))

    print("\nLogs:")
    for r in results:
        print(f"  {r.name}: {r.log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filesystem benchmark for RLM")
    parser.add_argument("--repo", default="/mnt/disk/OpenRLHF", help="Repository path to analyze (read-only)")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--max-file-size", type=int, default=8000)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--min-matches", type=int, default=1)
    parser.add_argument("--max-matches", type=int, default=8)
    parser.add_argument("--log-dir", default="visualizer/public/logs")
    parser.add_argument("--store-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)

    backend_kwargs = {"model_name": args.model}
    if args.backend == "vllm":
        backend_kwargs["base_url"] = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")

    context, metadata, file_list = serialize_directory(
        args.repo, max_files=args.max_files, max_file_size=args.max_file_size
    )
    print(f"Serializing repo: {metadata['root']}")
    print(f"  Files: {metadata['file_count']}")
    print(f"  Context size: {metadata['total_chars']:,} chars")

    tasks = build_tasks(
        file_list=file_list,
        root_path=args.repo,
        num_tasks=args.num_tasks,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
    )

    results: list[RunResult] = []
    setup_code = (
        "import re\n"
        "def safe_append(d, key, value):\n"
        "    if key not in d:\n"
        "        d[key] = []\n"
        "    d[key].append(value)\n"
        "\n"
        "def extract_file_blocks(context):\n"
        "    blocks = []\n"
        "    current_path = None\n"
        "    current_content = []\n"
        "    for line in context.splitlines():\n"
        "        line_stripped = line.strip()\n"
        "        if line_stripped.startswith('=== FILE: ') and line_stripped.endswith(' ==='):\n"
        "            if current_path is not None:\n"
        "                blocks.append((current_path, '\\n'.join(current_content)))\n"
        "            current_path = line_stripped[len('=== FILE: '):-len(' ===')].strip()\n"
        "            current_content = []\n"
        "        elif current_path is not None:\n"
        "            current_content.append(line)\n"
        "    if current_path is not None:\n"
        "        blocks.append((current_path, '\\n'.join(current_content)))\n"
        "    return blocks\n"
        "\n"
        "def find_matches(blocks, patterns):\n"
        "    compiled = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}\n"
        "    results = {k: [] for k in patterns.keys()}\n"
        "    for path, content in blocks:\n"
        "        for key, cregex in compiled.items():\n"
        "            if cregex.search(content):\n"
        "                results[key].append(path)\n"
        "    return results\n"
        "\n"
        "def format_answers(matches, keys):\n"
        "    parts = []\n"
        "    for key in keys:\n"
        "        parts.append(f\"{key}\\n\" + \"\\n\".join(matches.get(key, [])))\n"
        "    return \"\\n\\n\".join(parts)\n"
    )
    if not args.store_only:
        results.append(
            run_benchmark(
                name="baseline",
                backend=args.backend,
                backend_kwargs=backend_kwargs,
                context=context,
                tasks=tasks,
                file_list=file_list,
                log_dir=args.log_dir,
                max_iterations=args.max_iterations,
                store_prompt=False,
                environment_kwargs={"setup_code": setup_code},
            )
        )
    if not args.baseline_only:
        results.append(
            run_benchmark(
                name="store",
                backend=args.backend,
                backend_kwargs=backend_kwargs,
                context=context,
                tasks=tasks,
                file_list=file_list,
                log_dir=args.log_dir,
                max_iterations=args.max_iterations,
                store_prompt=True,
                environment_kwargs={"setup_code": setup_code},
            )
        )

    print_report(results, tasks)


if __name__ == "__main__":
    main()
