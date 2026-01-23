#!/usr/bin/env python3
"""
Hierarchical file mapping demo using RLM with commit protocol.

This demonstrates how to use RLM to build a structured map of a codebase,
using the same pattern as LongBench-v2 CodeQA: serialize files into context,
then let the model analyze and commit structured findings.

Usage:
    uv run python examples/filemap_demo.py
    uv run python examples/filemap_demo.py --path ./rlm/core
    uv run python examples/filemap_demo.py --path . --max-files 20
"""

import argparse
import os
from pathlib import Path

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


def serialize_directory(
    root_path: str,
    max_files: int = 50,
    max_file_size: int = 10000,
    extensions: set[str] = ALL_EXTENSIONS,
) -> tuple[str, dict]:
    """
    Serialize a directory into a context string for RLM.

    Returns:
        (context_string, metadata_dict)
    """
    root = Path(root_path).resolve()
    files_content = []
    file_list = []

    # Walk directory and collect files
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        # Skip hidden files and common excludes
        rel_path = path.relative_to(root)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        if any(part in {"node_modules", "__pycache__", ".venv", "venv", "dist", "build"} for part in rel_path.parts):
            continue

        file_list.append(rel_path)
        if len(file_list) >= max_files:
            break

    # Read file contents
    for rel_path in file_list:
        full_path = root / rel_path
        try:
            content = full_path.read_text(errors="replace")
            if len(content) > max_file_size:
                content = content[:max_file_size] + f"\n... [TRUNCATED at {max_file_size} chars]"
            files_content.append(f"{'='*60}\n=== FILE: {rel_path} ===\n{'='*60}\n{content}\n")
        except Exception as e:
            files_content.append(f"=== FILE: {rel_path} ===\n[ERROR reading file: {e}]\n")

    # Build context string
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

    return context, metadata


# Root prompt for hierarchical mapping task - encourages recursive worker spawning
FILEMAP_ROOT_PROMPT = """
Your task is to analyze this codebase and build a hierarchical map using RECURSIVE WORKERS.

This codebase has nested directories like:
- trainer/ppo_utils/mc/  (3 levels deep)
- trainer/ray/
- utils/deepspeed/

You MUST use rlm_worker() to delegate - DO NOT try to analyze everything yourself.

## Step 1: Identify directories

```repl
import re
# Extract file paths from listing
file_list_section = context.split("FILE CONTENTS")[0]
files = re.findall(r"- (.+)", file_list_section)
print(f"Total files: {len(files)}")

# Find top-level directories
top_dirs = sorted(set(f.split("/")[0] for f in files if "/" in f))
print(f"Top-level directories to analyze: {top_dirs}")
```

## Step 2: Spawn a worker for EACH directory

```repl
# Helper to extract directory content
def get_dir_files(dir_name):
    pattern = rf"=== FILE: {re.escape(dir_name)}/[^=]*?(?==== FILE:|\\Z)"
    matches = re.findall(pattern, context, re.DOTALL)
    return "\\n".join(matches)[:12000]

# IMPORTANT: Spawn worker for the first directory
first_dir = top_dirs[0] if top_dirs else "cli"
dir_content = get_dir_files(first_dir)

worker_prompt = f'''You are analyzing the "{first_dir}" directory of a codebase.

YOUR TASK: Return a JSON commit describing what you find.

FILES IN THIS DIRECTORY:
{dir_content}

INSTRUCTIONS:
1. If this directory has subdirectories, YOU MUST spawn rlm_worker() for each one
2. Analyze the files and identify the module's purpose
3. Return a commit like this:

{{
    "commit_id": "map_{first_dir}",
    "creates": [
        {{"type": "module", "id": "{first_dir}", "description": "...", "content": {{"purpose": "...", "key_files": [...]}}}}
    ],
    "links": []
}}

Return ONLY the JSON commit object as your final answer.
'''

print(f"Spawning worker for: {first_dir}")
result = rlm_worker(worker_prompt, store_cards=store.card_view())
print(f"Worker returned: {type(result)}")
apply_commit(result)
print(f"Commit applied. Store now has {len(store.view())} objects")
```

## Step 3: Repeat for remaining directories

```repl
# Spawn workers for remaining directories
for dir_name in top_dirs[1:4]:  # Do a few more
    dir_content = get_dir_files(dir_name)
    if not dir_content.strip():
        continue

    worker_prompt = f'''Analyze the "{dir_name}" directory. Return a JSON commit.

FILES:
{dir_content[:10000]}

Return a commit with:
- commit_id: "map_{dir_name}"
- creates: [{{"type": "module", "id": "{dir_name}", "description": "...", "content": {{...}}}}]

If you see subdirectories, spawn rlm_worker() for them too!
'''

    print(f"Spawning worker for: {dir_name}")
    result = rlm_worker(worker_prompt, store_cards=store.card_view())
    apply_commit(result)
    print(f"  -> Store now has {len(store.view())} objects")
```

## Step 4: View collected results and create summary

```repl
# See what we've mapped
print("=== MAPPED MODULES ===")
for obj in store.view():
    print(f"  [{obj.get('type')}] {obj.get('id')}: {obj.get('description', '')[:60]}")

# Create root summary
modules = [o for o in store.view() if o.get('type') == 'module']
summary_commit = {
    "commit_id": "codebase_summary",
    "creates": [{
        "type": "codebase",
        "id": "openrlhf",
        "description": "OpenRLHF - Open-source RLHF framework",
        "content": {"modules": [m.get('id') for m in modules], "total_modules": len(modules)}
    }],
    "links": [{"type": "contains", "src": "openrlhf", "dst": m.get('id')} for m in modules]
}
apply_commit(summary_commit)
```

```repl
summary = f"Hierarchically mapped {len(store.view())} objects: {[o.get('id') for o in store.view()]}"
print(summary)
```
FINAL_VAR(summary)

## REMEMBER
- Use rlm_worker() for EACH directory - this is the whole point!
- Workers can spawn their own workers for subdirectories (we have depth=3)
- Always apply_commit() to merge worker results
- The store accumulates findings from all workers
"""


def run_filemap(
    backend: str,
    backend_kwargs: dict,
    target_path: str,
    max_files: int,
    max_iterations: int,
    verbose: bool,
):
    """Run the file mapping task."""

    print(f"\n{'='*60}")
    print("HIERARCHICAL FILE MAPPING DEMO")
    print(f"{'='*60}\n")

    # Serialize the directory
    print(f"Serializing directory: {target_path}")
    context, metadata = serialize_directory(target_path, max_files=max_files)
    print(f"  Files: {metadata['file_count']}")
    print(f"  Context size: {metadata['total_chars']:,} chars")
    print(f"  Files: {metadata['files'][:10]}{'...' if len(metadata['files']) > 10 else ''}")

    # Create RLM with higher depth for recursive worker spawning
    logger = RLMLogger(log_dir="visualizer/public/logs", file_name="filemap_hierarchical")

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=3,  # Allow root -> worker -> sub-worker
        max_iterations=max_iterations,
        verbose=verbose,
        store_mode="shared",
        logger=logger,
    )

    print(f"\nRunning RLM analysis...")
    result = rlm.completion(context, root_prompt=FILEMAP_ROOT_PROMPT)

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"\nFinal answer:\n{result.response[:2000]}")
    print(f"\nLog file: {logger.log_file_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Hierarchical file mapping with RLM")
    parser.add_argument("--path", default="/mnt/disk/OpenRLHF/openrlhf", help="Directory to analyze")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-files", type=int, default=50)  # OpenRLHF has ~68 Python files
    parser.add_argument("--max-iterations", type=int, default=8)  # More iterations for hierarchical exploration
    parser.add_argument("--max-tokens", type=int, default=6000)  # More tokens for complex reasoning
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    backend_kwargs = {"model_name": args.model, "max_tokens": args.max_tokens}
    if args.backend == "vllm":
        backend_kwargs["base_url"] = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        backend_kwargs["api_key"] = os.getenv("VLLM_API_KEY", "dummy")

    run_filemap(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        target_path=args.path,
        max_files=args.max_files,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
