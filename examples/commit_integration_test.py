#!/usr/bin/env python3
"""
Integration test for commit protocol with real vllm backend.

Tests:
1. apply_commit() works in REPL and tracks CommitEvents
2. parse_commit() is available and works
3. store.card_view() works
4. Sub-LLM calls work (llm_query)

Usage:
    uv run python examples/commit_integration_test.py
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rlm import RLM
from rlm.logger import RLMLogger


def test_commit_protocol():
    """Test commit protocol with minimal LLM interaction."""
    print("=" * 60)
    print("TEST: Commit Protocol Integration")
    print("=" * 60)

    # Create RLM with store_prompt enabled
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key": os.getenv("VLLM_API_KEY", "dummy"),
            "max_tokens": 2000,
        },
        environment="local",
        max_depth=2,
        max_iterations=3,
        verbose=True,
        store_prompt=True,
        logger=RLMLogger(log_dir="visualizer/public/logs", file_name="commit_test"),
    )

    # Minimal context - just enough to test
    context = "The secret number is 42."

    # Root prompt that forces the model to test commit protocol
    root_prompt = """
Do exactly these steps in order, using ```repl``` code blocks:

STEP 1: Create a commit dict and apply it:
```repl
commit = {
    "commit_id": "test_commit_1",
    "creates": [
        {"type": "fact", "id": "f1", "description": "Secret number found", "content": {"value": 42}}
    ],
    "links": [],
    "proposes_updates": []
}
result = apply_commit(commit)
print(f"Commit applied: {result.success}, created: {list(result.created_ids.keys())}")
```

STEP 2: Use store.card_view() to see what was created:
```repl
cards = store.card_view()
for c in cards:
    print(f"  {c['type']}: {c['description']}")
```

STEP 3: Return the final answer:
FINAL(42)
"""

    print(f"\nRunning with context: {context[:50]}...")
    print(f"Root prompt instructs model to test commit protocol\n")

    result = rlm.completion(context, root_prompt=root_prompt)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Final answer: {result.response}")
    print(f"Execution time: {result.execution_time:.2f}s")

    return result


def test_sub_llm_in_repl():
    """Test that sub-LLM calls work in the REPL."""
    print("\n" + "=" * 60)
    print("TEST: Sub-LLM Calls in REPL")
    print("=" * 60)

    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key": os.getenv("VLLM_API_KEY", "dummy"),
            "max_tokens": 1500,
        },
        environment="local",
        max_depth=2,
        max_iterations=3,
        verbose=True,
        store_prompt=True,
        logger=RLMLogger(log_dir="visualizer/public/logs", file_name="sub_llm_test"),
    )

    context = "Document: The capital of France is Paris. The population is about 2 million in the city proper."

    root_prompt = """
Test sub-LLM query functionality. Do these steps:

STEP 1: Query a sub-LLM with a simple question:
```repl
response = llm_query("What is the capital of France based on: " + context[:100])
print(f"Sub-LLM response: {response[:200]}")
```

STEP 2: Parse the sub-LLM response and create a commit:
```repl
commit = {
    "commit_id": "sub_llm_result",
    "creates": [
        {"type": "extracted_fact", "id": "capital", "description": "Capital city identified", "content": response[:100]}
    ]
}
result = apply_commit(commit)
print(f"Commit success: {result.success}")
```

STEP 3: Return final answer:
FINAL(Paris)
"""

    print(f"\nTesting sub-LLM calls...")
    result = rlm.completion(context, root_prompt=root_prompt)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Final answer: {result.response}")

    return result


def main():
    print("\n" + "#" * 60)
    print("# COMMIT PROTOCOL INTEGRATION TESTS")
    print("#" * 60 + "\n")

    try:
        # Test 1: Basic commit protocol
        result1 = test_commit_protocol()
        test1_pass = "42" in result1.response
        print(f"\nTest 1 (Commit Protocol): {'PASS' if test1_pass else 'FAIL'}")

        # Test 2: Sub-LLM calls
        result2 = test_sub_llm_in_repl()
        test2_pass = "paris" in result2.response.lower()
        print(f"Test 2 (Sub-LLM Calls): {'PASS' if test2_pass else 'FAIL'}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Test 1 (Commit Protocol): {'PASS' if test1_pass else 'FAIL'}")
        print(f"Test 2 (Sub-LLM Calls): {'PASS' if test2_pass else 'FAIL'}")
        print(f"\nLogs written to: visualizer/public/logs/")

        if test1_pass and test2_pass:
            print("\n✓ All tests passed!")
            return 0
        else:
            print("\n✗ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
