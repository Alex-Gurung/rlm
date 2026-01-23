#!/usr/bin/env python3
"""
Demo: Commit protocol for structured sub-LLM outputs.

Shows how workers return JSON commits that are merged into the global store.

Usage:
    uv run python examples/commit_demo.py
    uv run python examples/commit_demo.py --backend openai
"""

import argparse
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rlm.core.commit import Commit, CommitCreate, MergeResult, apply_commit, parse_commit
from rlm.core.store import Store


def demo_parse_commit():
    """Demonstrate parsing commits from various text formats."""
    print("=" * 60)
    print("DEMO: Parsing commits from text")
    print("=" * 60)

    # 1. Pure JSON
    json_text = '{"commit_id": "pure_json", "creates": [{"type": "note", "id": "n1", "description": "Found X", "content": "Evidence text"}]}'
    commit = parse_commit(json_text)
    print(f"\n1. Pure JSON -> commit_id={commit.commit_id}, creates={len(commit.creates)}")

    # 2. Fenced JSON (as LLMs often return)
    fenced_text = """
I analyzed the chunk and found relevant evidence.

```json
{
  "commit_id": "worker_chunk_7",
  "creates": [
    {"type": "evidence", "id": "e1", "description": "Quote about X", "content": {"quote": "...", "page": 42}}
  ],
  "links": [
    {"type": "supports", "src": "e1", "dst": "hypothesis/H"}
  ]
}
```
"""
    commit = parse_commit(fenced_text)
    print(f"2. Fenced JSON -> commit_id={commit.commit_id}, creates={len(commit.creates)}, links={len(commit.links)}")

    # 3. Embedded JSON (fallback parsing)
    embedded_text = """
After careful analysis, I found one relevant piece of evidence.

{"commit_id": "embedded", "creates": [{"type": "claim", "id": "c1", "description": "Main claim", "content": "..."}], "proposes_updates": []}

That concludes my analysis.
"""
    commit = parse_commit(embedded_text)
    print(f"3. Embedded JSON -> commit_id={commit.commit_id}, creates={len(commit.creates)}")

    # 4. Invalid text (graceful error handling)
    invalid_text = "This is not JSON at all, just plain text."
    commit = parse_commit(invalid_text, fallback_id="fallback_worker")
    print(f"4. Invalid text -> commit_id={commit.commit_id}, error={commit.error is not None}")


def demo_apply_commit():
    """Demonstrate applying commits to the store."""
    print("\n" + "=" * 60)
    print("DEMO: Applying commits to store")
    print("=" * 60)

    store = Store()

    # Create a commit with multiple operations
    commit = Commit(
        commit_id="worker_doc_42",
        creates=[
            CommitCreate(
                type="evidence",
                id="e1",
                description="Quote from doc 42 supporting hypothesis",
                content={"quote": "The data clearly shows...", "page": 15},
                tags=["evidence", "doc:42"],
            ),
            CommitCreate(
                type="summary",
                id="s1",
                description="Summary of doc 42 findings",
                content="Document 42 contains three key pieces of evidence...",
                tags=["summary"],
            ),
        ],
    )

    # Apply with batch prefix (simulating a batch operation)
    result = apply_commit(store, commit, batch_prefix="batch_wave0")

    print(f"\nApplied commit: {result.commit_id}")
    print(f"  Created objects: {len(result.created_ids)}")
    print(f"  Links created: {result.links_created}")
    print(f"  Proposals stored: {result.proposals_stored}")
    print(f"  Success: {result.success}")

    # View the store contents
    print("\nStore contents after commit:")
    for item in store.view():
        print(f"  - [{item['type']}] {item['description'][:50]}...")


def demo_store_card_view():
    """Demonstrate store card views for worker context."""
    print("\n" + "=" * 60)
    print("DEMO: Store card views for worker context")
    print("=" * 60)

    store = Store()

    # Create some objects
    store.create(type="hypothesis", description="Hypothesis H: X causes Y", content={"status": "pending"}, tags=["primary"])
    store.create(type="hypothesis", description="Hypothesis H2: A leads to B", content={"status": "confirmed"}, tags=["secondary"])
    store.create(type="note", description="General observation about the data", content="...", tags=["note"])

    # Get card view (lightweight metadata only)
    cards = store.card_view("type=hypothesis")

    print(f"\nCard view (type=hypothesis): {len(cards)} cards")
    for card in cards:
        print(f"  - id={card['id']}, type={card['type']}, desc={card['description'][:30]}...")

    # Cards are suitable for passing to workers
    print("\nCards contain only metadata, safe to pass to sub-LLMs without leaking full content.")


def demo_full_workflow():
    """Demonstrate full workflow: chunk -> worker commits -> merge -> view."""
    print("\n" + "=" * 60)
    print("DEMO: Full workflow simulation")
    print("=" * 60)

    store = Store()

    # 1. Create a hypothesis to search for
    hyp_id = store.create(
        type="hypothesis",
        description="Hypothesis: The phoenix project succeeded",
        content={"question": "Did the phoenix project achieve its goals?"},
        tags=["primary"],
    )
    print(f"\n1. Created hypothesis: {hyp_id}")

    # 2. Simulate worker responses (as if from llm_query or rlm_worker)
    worker_responses = [
        """```json
{
  "commit_id": "worker_chunk_0",
  "creates": [
    {"type": "evidence", "id": "e1", "description": "Phoenix launch successful", "content": {"quote": "Project launched on time", "chunk": 0}}
  ],
  "links": [{"type": "supports", "src": "e1", "dst": "HYPOTHESIS"}]
}
```""",
        """```json
{
  "commit_id": "worker_chunk_1",
  "creates": [
    {"type": "evidence", "id": "e1", "description": "Funding issues noted", "content": {"quote": "Budget exceeded by 20%", "chunk": 1}}
  ],
  "links": [{"type": "contradicts", "src": "e1", "dst": "HYPOTHESIS"}]
}
```""",
        """```json
{
  "commit_id": "worker_chunk_2",
  "creates": [],
  "links": []
}
```""",
    ]

    # 3. Parse and merge each worker commit
    print("\n2. Processing worker responses:")
    all_results: list[MergeResult] = []
    for i, response in enumerate(worker_responses):
        commit = parse_commit(response, fallback_id=f"worker_{i}")
        result = apply_commit(store, commit, batch_prefix="batch_0")
        all_results.append(result)
        print(f"   Worker {i}: created={len(result.created_ids)}, links={result.links_created}, success={result.success}")

    # 4. View aggregated results
    print("\n3. Final store contents:")
    for item in store.view():
        print(f"   [{item['type']:15}] {item['description'][:45]}...")

    # 5. Summary
    total_created = sum(len(r.created_ids) for r in all_results)
    total_links = sum(r.links_created for r in all_results)
    all_success = all(r.success for r in all_results)
    print(f"\n4. Summary: {total_created} objects created, {total_links} links, all_success={all_success}")


def main():
    parser = argparse.ArgumentParser(description="Commit Protocol Demo")
    parser.add_argument("--demo", choices=["parse", "apply", "cards", "workflow", "all"], default="all")
    args = parser.parse_args()

    if args.demo in ("parse", "all"):
        demo_parse_commit()

    if args.demo in ("apply", "all"):
        demo_apply_commit()

    if args.demo in ("cards", "all"):
        demo_store_card_view()

    if args.demo in ("workflow", "all"):
        demo_full_workflow()


if __name__ == "__main__":
    main()
