# RLM Store & Commit: Improvement Plan

## Executive Summary

After reviewing the codebase, I've identified several issues with the current commit protocol and store architecture. This plan proposes concrete fixes organized by priority.

---

## Issues Identified

### Issue 1: Workers Are Isolated - No Sibling Visibility

**Problem:** Each worker has its own `Store` instance. Worker A cannot see commits from Worker B. The only coordination mechanism is `store_cards`, which is a read-only snapshot passed at spawn time.

**Impact:**
- Workers can't build on each other's findings
- Parent must manually coordinate all information sharing
- Later workers in a sequential loop don't see earlier workers' results

**Current behavior:**
```python
cards = store.card_view("type=hypothesis")  # Snapshot at time T0
for doc in docs:
    commit = rlm_worker(prompt, store_cards=cards)  # All workers see same T0 snapshot
    apply_commit(commit)  # Parent store updated, but next worker won't see it
```

---

### Issue 2: No Parallel Worker Spawning

**Problem:** `rlm_worker()` is synchronous. Spawning 10 workers takes 10x the time of one worker.

**Impact:** Major performance bottleneck for fan-out tasks like OOLONG.

**Missing API:**
```python
# This doesn't exist yet:
commits = rlm_worker_batched([
    {"prompt": "Analyze doc 0", "store_cards": cards},
    {"prompt": "Analyze doc 1", "store_cards": cards},
    {"prompt": "Analyze doc 2", "store_cards": cards},
])
for commit in commits:
    apply_commit(commit)
```

---

### Issue 3: Commit Protocol is Mandatory for Workers

**Problem:** `rlm_worker()` always parses response as JSON commit via `parse_commit()`. If parsing fails, you get an error commit with `error` field set.

**Impact:**
- Can't use `rlm_worker()` for simple text-answer tasks
- Forces commit overhead even when not needed
- No way to get raw worker response

**Current code (local_repl.py:367):**
```python
commit = parse_commit(result.response, fallback_id=f"worker_{self.depth}")
return commit.to_dict()  # Always returns commit dict
```

---

### Issue 4: `proposes_updates` Are Never Applied

**Problem:** The `proposes_updates` field in commits is stored as `type="update_proposal"` objects but never actually applied to target objects.

**Impact:** Feature is half-implemented and not useful in practice.

**Current code (commit.py:321-336):**
```python
# Stores proposal but doesn't apply it
store.create(
    type="update_proposal",
    description=f"Proposal for {update.target_id}",
    content={...},
)
```

---

### Issue 5: ID Resolution Edge Cases

**Problem:** Workers receive hypothesis IDs from `store_cards` (e.g., `"abc123"`). When creating links, they reference these global IDs. But `apply_commit` tries to resolve IDs through `created_ids` mapping first, which can cause confusion.

**Impact:** Links may not resolve correctly if local and global IDs collide.

---

### Issue 6: No Return Type Flexibility

**Problem:** Workers can only return commits. No way to specify:
- Return plain text (like `llm_query`)
- Return commit with specific schema validation
- Return streaming/partial results

---

### Issue 7: Depth Parameter Naming Confusion

**Problem:** RLM has `depth` (my depth), environment has `depth` (parent's depth + 1). The spawning logic is correct but confusing to read.

```python
# local_repl.py:344 - This looks wrong but is actually correct
worker_rlm = RLM(
    depth=self.depth,  # Worker RLM depth = this env's depth (which is parent RLM depth + 1)
    ...
)
```

---

## Proposed Solutions

### Solution 1: Add `store_cards` Refresh Option

**Approach:** Allow workers to request fresh cards from parent.

**Option A - Pass callback:**
```python
def _rlm_worker(self, prompt, store_cards=None, refresh_cards_fn=None):
    # Worker can call refresh_cards_fn() to get updated cards
```

**Option B - Add `store_cards="live"` mode:**
```python
commit = rlm_worker(prompt, store_cards="live")  # Worker receives latest cards
```

**Option C (Recommended) - Update cards between sequential workers:**
```python
for doc in docs:
    cards = store.card_view("type=hypothesis")  # Fresh cards each iteration
    commit = rlm_worker(prompt, store_cards=cards)
    apply_commit(commit)
```

This is already possible - just update the examples/prompts to show this pattern.

---

### Solution 2: Add `rlm_worker_batched()`

**Approach:** Parallel worker spawning using ThreadPoolExecutor.

```python
def _rlm_worker_batched(
    self,
    tasks: list[dict],  # Each dict has: prompt, store_cards (optional), worker_config (optional)
    max_parallel: int = 8,
) -> list[dict]:  # Returns list of commit dicts
    """
    Spawn multiple workers in parallel.

    Args:
        tasks: List of task dicts, each with:
            - prompt: str (required)
            - store_cards: list[dict] (optional, defaults to current store.card_view())
            - worker_config: dict (optional)
        max_parallel: Max concurrent workers

    Returns:
        List of commit dicts in same order as tasks
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx = {}
        for idx, task in enumerate(tasks):
            future = executor.submit(
                self._rlm_worker,
                task["prompt"],
                task.get("store_cards"),
                task.get("worker_config"),
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
```

**Usage:**
```python
tasks = [
    {"prompt": f"Analyze {doc['doc_id']}", "store_cards": cards}
    for doc in context["docs"]
]
commits = rlm_worker_batched(tasks)
for i, commit in enumerate(commits):
    apply_commit(commit, batch_prefix=f"doc_{i}")
```

---

### Solution 3: Add `raw=True` Option to `rlm_worker()`

**Approach:** Allow returning raw text instead of forcing commit parsing.

```python
def _rlm_worker(
    self,
    prompt: str,
    store_cards: list[dict] | None = None,
    worker_config: dict | None = None,
    raw: bool = False,  # NEW: Return raw response string
) -> dict | str:
    ...
    result = worker_rlm.completion(worker_context, root_prompt=worker_prompt)

    if raw:
        return result.response  # Return raw string

    commit = parse_commit(result.response, fallback_id=f"worker_{self.depth}")
    return commit.to_dict()
```

**Usage:**
```python
# For simple questions - get text back
answer = rlm_worker("What is the capital of France?", raw=True)

# For structured analysis - get commit back
commit = rlm_worker("Analyze this document and return evidence", store_cards=cards)
```

---

### Solution 4: Implement `apply_updates` Option

**Approach:** Add flag to `apply_commit` to actually apply proposed updates.

```python
def apply_commit(
    store: "Store",
    commit: Commit,
    batch_prefix: str = "",
    apply_updates: bool = False,  # NEW: Actually apply proposes_updates
) -> MergeResult:
    ...
    if apply_updates:
        for update in commit.proposes_updates:
            target = store.get(update.target_id)
            if target:
                # Apply patch to target.content
                if isinstance(target.content, dict) and isinstance(update.patch, dict):
                    target.content.update(update.patch)
                if update.description_update:
                    target.description = update.description_update
```

**Alternative:** Keep proposals as review queue, add `store.apply_proposal(proposal_id)` method.

---

### Solution 5: Clarify ID Resolution Rules

**Approach:** Document and enforce clear ID resolution order:

1. Check `created_ids` mapping (local IDs from this commit)
2. Check store directly (global IDs)
3. If not found, skip (don't fail)

Add warning when ambiguous:
```python
if local_id in created_ids and store.get(local_id):
    warnings.append(f"Ambiguous ID '{local_id}' - using local mapping")
```

---

### Solution 6: Rename Depth Parameters

**Approach:** Rename for clarity:

```python
# In RLM.__init__:
rlm_depth: int = 0  # This RLM's depth in the hierarchy

# In LocalREPL.__init__:
env_depth: int = 1  # Environment depth (always rlm_depth + 1)
```

Or add comments explaining the relationship.

---

## Implementation Priority

### P0 - Critical (Do First)

1. **Add `rlm_worker_batched()`** - Major performance win for fan-out tasks
2. **Add `raw=True` option** - Allows simpler worker tasks
3. **Update examples** to show `store_cards` refresh pattern

### P1 - Important

4. **Implement `apply_updates` or proposal review** - Complete the feature
5. **Add ID resolution warnings** - Help debug linking issues
6. **Add comprehensive integration tests** for worker batching

### P2 - Nice to Have

7. **Rename depth parameters** for clarity
8. **Add `store_cards="live"` mode** - Auto-refresh cards
9. **Add worker streaming/progress** - See partial results
10. **Add commit schema validation** - Ensure workers return expected structure

---

## Testing Plan

### Unit Tests to Add

```python
# test_commit.py additions:

def test_rlm_worker_batched_parallel():
    """Multiple workers run in parallel and return commits."""

def test_rlm_worker_raw_mode():
    """raw=True returns string instead of commit dict."""

def test_apply_commit_with_updates():
    """proposes_updates actually modify target objects."""

def test_id_resolution_priority():
    """Local IDs take precedence over global IDs."""
```

### Integration Tests

```python
# test_commit_integration.py additions:

def test_worker_batch_fan_out():
    """Spawn 4 workers in parallel, merge all commits, verify store state."""

def test_sequential_workers_with_card_refresh():
    """Later workers see earlier workers' commits via fresh cards."""
```

---

## Migration Notes

All changes should be **backward compatible**:

- `rlm_worker()` continues to work exactly as before
- `apply_commit()` continues to work exactly as before
- New parameters have sensible defaults
- No breaking changes to commit JSON schema
