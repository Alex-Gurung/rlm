#!/usr/bin/env python3
"""
Synthetic commit-protocol task.

Runs a tiny, deterministic analysis that should trigger:
- rlm_worker() subcalls
- apply_commit() merges
- store events + commit events in logs

Usage:
  uv run python examples/commit_synthetic_task.py
  uv run python examples/commit_synthetic_task.py --base-url http://localhost:8000/v1 --model-name Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
from rlm.utils.prompts_commit import COMMIT_PROTOCOL_PROMPT_ADDON


def build_context() -> dict[str, Any]:
    return {
        "docs": [
            {
                "doc_id": "doc_0",
                "lines": [
                    "1 SUCCESS: Phoenix launched on time.",
                    "2 SUCCESS: 92% uptime in the first month.",
                    "3 NOTE: Minor bugs fixed quickly.",
                ],
            },
            {
                "doc_id": "doc_1",
                "lines": [
                    "1 FAIL: Phoenix missed the Q2 deadline.",
                    "2 FAIL: Budget overrun of 40%.",
                ],
            },
            {
                "doc_id": "doc_2",
                "lines": [
                    "1 SUCCESS: Customer satisfaction averaged 4.6/5.",
                    "2 SUCCESS: CEO called Phoenix a success.",
                ],
            },
            {
                "doc_id": "doc_3",
                "lines": [
                    "1 SUCCESS: Audit states goals were met.",
                    "2 SUCCESS: Reliability targets achieved in Q4.",
                ],
            },
        ]
    }


def build_root_prompt() -> str:
    return """
You must use the REPL and the commit protocol. Do not answer directly.
Your next action MUST be a ```repl``` block.

Goal: decide whether H_SUCCESS or H_FAILURE is better supported.

Brief method notes (for correct usage):
- store.create(type, description, content, ...) -> returns an id string you should save and reuse.
- store.card_view(query) -> small list of ids + descriptions (safe to pass to workers).
- rlm_worker(prompt, store_cards=...) -> returns a JSON commit dict as text.
- apply_commit(commit, batch_prefix=...) -> merges commit into store and records events.
- In REPL blocks, use FINAL(result_dict) to emit the final answer. FINAL_VAR is for assistant text, not for REPL code.

Mini unrelated example of worker output (format only, not about this task):
```json
{
  "commit_id": "worker_example",
  "creates": [
    {"type": "evidence", "id": "e1", "description": "Sentence about topic A", "content": {"quote": "..."}},
    {"type": "summary", "id": "s1", "description": "One-line summary", "content": "A implies B", "parents": ["e1"]}
  ],
  "links": [
    {"type": "supports", "src": "e1", "dst": "hypothesis/H"}
  ],
  "proposes_updates": []
}
```

Steps (do all of this in REPL):
1) Create two hypothesis objects in `store`:
   - H_SUCCESS: "Phoenix succeeded"
   - H_FAILURE: "Phoenix failed"
2) For each doc in context["docs"], call rlm_worker once with a prompt that:
   - Treats lines containing "SUCCESS" as supporting H_SUCCESS
   - Treats lines containing "FAIL" as contradicting H_SUCCESS (i.e., supporting H_FAILURE)
   - Returns a JSON commit containing:
     * Evidence objects for each relevant line
     * One doc_summary object whose parents are the evidence IDs
     * Link objects: supports/contradicts from evidence -> hypothesis id
   - Use backrefs: {"source_id": doc_id, "start": line_num, "end": line_num, "unit": "lines"}
   - Use hypothesis IDs from store.card_view("type=hypothesis")
3) apply_commit(...) for each worker commit with batch_prefix=doc_id
4) Count supports vs contradicts by scanning store.view("type=link") and store.get(id)
5) FINAL_VAR a dict like {"winner": "...", "supports": N, "contradicts": M}

Return only that FINAL output.
""".strip()


def parse_result(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(text)
        if isinstance(value, dict):
            return value
    except (ValueError, SyntaxError):
        pass
    final_match = re.search(r"^\\s*FINAL\\((.+)\\)\\s*$", text, flags=re.MULTILINE | re.DOTALL)
    if final_match:
        inner = final_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        try:
            value = ast.literal_eval(inner)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            pass
    final_var_match = re.search(r"FINAL_VAR\\((\\{.*?\\})\\)", text, flags=re.DOTALL)
    if final_var_match:
        inner = final_var_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        try:
            value = ast.literal_eval(inner)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            pass
    return None


def resolve_log_dir(arg_value: str | None) -> str:
    if arg_value:
        return arg_value

    # Prefer the local worktree visualizer (matches `npm run dev` in this repo).
    repo_root = Path(__file__).resolve().parents[1]
    candidate_local = repo_root / "visualizer" / "public" / "logs"
    candidate_parent = repo_root.parent / "visualizer" / "public" / "logs"

    if candidate_local.exists():
        return str(candidate_local)
    if candidate_parent.exists():
        return str(candidate_parent)
    return str(candidate_local)


def count_commit_events(log_path: str) -> int:
    count = 0
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("type") != "iteration":
                    continue
                for block in entry.get("code_blocks", []):
                    result = block.get("result", {})
                    count += len(result.get("commit_events", []))
    except Exception:
        return 0
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic commit-protocol test")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    context = build_context()
    root_prompt = build_root_prompt()

    log_dir = os.path.abspath(resolve_log_dir(args.log_dir))
    logger = RLMLogger(log_dir)

    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "base_url": args.base_url,
            "model_name": args.model_name,
            "api_key": os.getenv("OPENAI_API_KEY", "local"),
            "max_tokens": args.max_tokens,
        },
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        custom_system_prompt=RLM_SYSTEM_PROMPT + COMMIT_PROTOCOL_PROMPT_ADDON,
        logger=logger,
        store_mode="shared",
    )

    print("=" * 72)
    print("Commit Protocol Synthetic Test")
    print("=" * 72)
    print(f"Model: {args.model_name}")
    print(f"Base URL: {args.base_url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Log dir: {log_dir}")
    print(f"Log file: {logger.log_file_path}")
    print("Expected winner: H_SUCCESS (supports=3 docs, contradicts=1 doc)")
    print("=" * 72)

    result = rlm.completion(context, root_prompt=root_prompt)
    print("Raw final answer:")
    print(result.response)
    print("=" * 72)

    parsed = parse_result(result.response)
    if not parsed:
        print("Could not parse final answer as dict. Check raw output above.")
        print(f"Commit events recorded: {count_commit_events(logger.log_file_path)}")
        return

    winner = parsed.get("winner")
    supports = parsed.get("supports")
    contradicts = parsed.get("contradicts")

    print("Parsed result:")
    print(f"  winner      : {winner}")
    print(f"  supports    : {supports}")
    print(f"  contradicts : {contradicts}")
    print("=" * 72)

    expected = {"winner": "H_SUCCESS", "supports": 3, "contradicts": 1}
    ok = winner == expected["winner"] and supports == expected["supports"] and contradicts == expected["contradicts"]
    if ok:
        print("PASS")
    else:
        print("FAIL")
        print(f"Expected: {expected}")
    print(f"Commit events recorded: {count_commit_events(logger.log_file_path)}")


if __name__ == "__main__":
    main()
