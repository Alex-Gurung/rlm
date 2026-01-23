# RLM Store & Commit Architecture

## Store Modes

RLM supports two store modes for worker coordination:

1. **Isolated Stores (default)**: Each worker has its own store. Workers communicate via commit protocol (serialize findings, merge after completion).
2. **Shared Store (explicit opt-in)**: Parallel workers share a single append-only store. Workers can see each other's findings in real-time.

---

## Shared Store Architecture (explicit opt-in)

When `environment_kwargs={"shared_store": True}` is set for local environments, all workers share a single `SharedStore`:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Root RLM (depth=0, max_depth=3, store_prompt=True)                               │
│                                                                                  │
│ REPL globals:                                                                    │
│   context, store (WorkerStoreProxy), rlm_worker, rlm_worker_batched              │
│                                                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────────┐ │
│ │ Parallel Worker Fan-Out                                                      │ │
│ │                                                                              │ │
│ │   tasks = [{"prompt": f"Analyze {doc}"} for doc in docs]                     │ │
│ │   commits = rlm_worker_batched(tasks, max_parallel=8)                        │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│         │                     │                     │                            │
│         ▼                     ▼                     ▼                            │
│   ┌───────────┐         ┌───────────┐         ┌───────────┐                      │
│   │ Worker A  │         │ Worker B  │         │ Worker C  │                      │
│   │ depth=1   │         │ depth=1   │         │ depth=1   │                      │
│   │           │         │           │         │           │                      │
│   │ Proxy to  │         │ Proxy to  │         │ Proxy to  │   All workers share  │
│   │ shared    │◄───────►│ shared    │◄───────►│ shared    │   one SharedStore    │
│   │ store     │         │ store     │         │ store     │                      │
│   │           │         │           │         │           │                      │
│   │ Can see:  │         │ Can see:  │         │ Can see:  │                      │
│   │ • Own     │         │ • Own     │         │ • Own     │                      │
│   │ • A's     │         │ • A's     │         │ • A's     │                      │
│   │ • B's     │         │ • B's     │         │ • B's     │                      │
│   │ • C's     │         │ • C's     │         │ • C's     │                      │
│   └─────┬─────┘         └─────┬─────┘         └─────┬─────┘                      │
│         │                     │                     │                            │
│         └─────────────────────┼─────────────────────┘                            │
│                               ▼                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │                        SharedStore (append-only)                         │   │
│   │                                                                          │   │
│   │  Event Log:                                                              │   │
│   │    seq=0: CREATE worker_A_000000 type=evidence "Found X"                 │   │
│   │    seq=1: CREATE worker_B_000000 type=note "Analyzing..."                │   │
│   │    seq=2: CREATE worker_C_000000 type=evidence "Found Y"                 │   │
│   │    seq=3: CREATE worker_A_000001 type=summary "X implies..."             │   │
│   │    ...                                                                   │   │
│   │                                                                          │   │
│   │  Materialized View:                                                      │   │
│   │    Objects queryable by type, worker, description, tags                  │   │
│   │                                                                          │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   Workers can query each other's findings during execution:                      │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │   # In Worker B's code:                                                  │   │
│   │   others = store.view_others("type=evidence")  # See A's and C's finds   │   │
│   │   for obj in others:                                                     │   │
│   │       print(f"Worker {obj['worker']} found: {obj['description']}")       │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Shared Store Data Flow

```
Worker A                    Worker B                    Worker C
    │                           │                           │
    │   store.create(...)       │   store.view_others()     │   store.create(...)
    │           │               │           │               │           │
    │           ▼               │           │               │           ▼
    │   ┌───────────────────────┴───────────┴───────────────┴───────────────────┐
    │   │                                                                       │
    └──►│                     SharedStore (thread-safe)                         │◄──┘
        │                                                                       │
        │  _events: [StoreEvent, ...]          # Append-only log               │
        │  _objects: {id: SharedStoreObject}   # Materialized view             │
        │  _lock: RLock                        # Thread safety                 │
        │                                                                       │
        │  IDs: {worker_id}_{seq:06d}          # No collisions                 │
        │                                                                       │
        └───────────────────────────────────────────────────────────────────────┘
```

### Key Shared Store APIs

```python
# WorkerStoreProxy (what workers see as `store`)

store.create(type, description, content, ...)  # Attributed to this worker
store.view("type=evidence")                     # See ALL workers' objects
store.view_others("type=evidence")              # See only OTHER workers' objects
store.invalidate(target_id, reason)             # Soft-delete an object
store.card_view(query)                          # Lightweight cards for context

# SharedStore (underlying implementation)

shared.create(worker_id, type, description, content, ...)
shared.query(type=None, worker=None, exclude_worker=None, ...)
shared.invalidate(worker_id, target_id, reason)
shared.events_since(seq)                        # For future remote sync
```

---

## Isolated Store Architecture (default)

## Overview Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Root RLM (depth=0, max_depth=3)                                                  │
│                                                                                  │
│ REPL globals:                                                                    │
│   context, store, llm_query, llm_query_batched, rlm_worker, apply_commit         │
│                                                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────────┐ │
│ │ Worker Fan-Out (sequential with commit protocol)                             │ │
│ │                                                                              │ │
│ │   cards = store.card_view("type=hypothesis")  # snapshot for workers         │ │
│ │                                                                              │ │
│ │   for doc in docs:                                                           │ │
│ │       commit = rlm_worker(f"Analyze {doc}", store_cards=cards)               │ │
│ │       apply_commit(commit, batch_prefix=doc.id)                              │ │
│ │                                                                              │ │
│ │   # Or use rlm_worker_batched for parallel (requires store_prompt=True):     │ │
│ │   # tasks = [{"prompt": f"Analyze {doc}"} for doc in docs]                   │ │
│ │   # commits = rlm_worker_batched(tasks, max_parallel=8)                      │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│         │                     │                     │                            │
│         ▼                     ▼                     ▼                            │
│   ┌───────────┐         ┌───────────┐         ┌───────────┐                      │
│   │ Worker A  │         │ Worker B  │         │ Worker C  │                      │
│   │ depth=1   │         │ depth=1   │         │ depth=1   │                      │
│   │           │         │           │         │           │                      │
│   │ Own REPL: │         │ Own REPL: │         │ Own REPL: │   Isolated stores    │
│   │ • context │         │ • context │         │ • context │   Read-only cards    │
│   │ • store   │         │ • store   │         │ • store   │                      │
│   │ • llm_query         │ • llm_query         │ • llm_query                      │
│   │ • rlm_worker        │ • rlm_worker        │ • rlm_worker ──┐                 │
│   │           │         │           │         │           │    │                 │
│   │ Returns:  │         │ Returns:  │         │           │    ▼                 │
│   │ {commit}  │         │ {commit}  │         │           │ ┌─────────┐          │
│   └─────┬─────┘         └─────┬─────┘         │           │ │Sub-Wrkr │          │
│         │                     │               │           │ │depth=2  │          │
│         │                     │               │           │ └────┬────┘          │
│         │                     │               │  apply_commit    │               │
│         │                     │               │  internally      │               │
│         │                     │               │           │◄─────┘               │
│         │                     │               │ Returns:  │                      │
│         │                     │               │ {commit}  │                      │
│         │                     │               └─────┬─────┘                      │
│         │                     │                     │                            │
│         ▼                     ▼                     ▼                            │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │                    Root Store (after merging)                            │   │
│   │                                                                          │   │
│   │  apply_commit(A, "doc_0")  →  doc_0/worker_A/e1, doc_0/worker_A/s1      │   │
│   │  apply_commit(B, "doc_1")  →  doc_1/worker_B/e1, doc_1/worker_B/s1      │   │
│   │  apply_commit(C, "doc_2")  →  doc_2/worker_C/e1, ... (incl sub-worker)  │   │
│   │                                                                          │   │
│   │  Objects: hypotheses, evidence, summaries, links                         │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ Root Synthesizes Final Answer                                            │   │
│   │                                                                          │   │
│   │   links = store.view("type=link")                                        │   │
│   │   supports = len([l for l in links if "supports" in l["description"]])   │   │
│   │   contradicts = len([l for l in links if "contradicts" in ...])          │   │
│   │   winner = "H_SUCCESS" if supports > contradicts else "H_FAILURE"        │   │
│   │                                                                          │   │
│   │   FINAL_VAR({"winner": winner, "supports": supports, ...})               │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Current Data Flow

```
┌─────────────────┐     store_cards      ┌─────────────────┐
│   Parent Store  │ ──────────────────►  │  Worker Store   │
│   (mutable)     │   (read-only         │  (isolated)     │
│                 │    snapshot)         │                 │
└────────┬────────┘                      └────────┬────────┘
         │                                        │
         │         apply_commit(commit)           │
         │◄───────────────────────────────────────┘
         │            (merge back)
         ▼
┌─────────────────┐
│  Parent Store   │  Now contains namespaced objects:
│  (updated)      │  batch_prefix/commit_id/local_id
└─────────────────┘
```

## Commit Structure

```python
{
    "commit_id": "worker_doc_42",

    # Create new objects
    "creates": [
        {
            "type": "evidence",       # note|claim|summary|evidence|link|...
            "id": "e1",               # local ID, namespaced on merge
            "description": "Quote supporting hypothesis",
            "content": {"quote": "...", "page": 15},
            "backrefs": [{"source_id": "doc_42", "start": 100, "end": 200, "unit": "chars"}],
            "parents": ["BATCH:parent"],  # reference batch parent
            "tags": ["evidence", "doc:42"]
        }
    ],

    # Create relationships between objects
    "links": [
        {
            "type": "supports",       # supports|contradicts|refines
            "src": "e1",              # local or global ID
            "dst": "abc123"           # usually global ID from store_cards
        }
    ],

    # Propose updates (stored as proposals, not applied)
    "proposes_updates": [
        {
            "target_id": "abc123",
            "patch": {"evidence_count": 5},
            "description_update": "Updated with new evidence"
        }
    ],

    # Set if parsing failed
    "error": null
}
```

## Key Files

| File | Purpose |
|------|---------|
| `rlm/core/rlm.py` | Main RLM orchestrator, spawns environment per completion |
| `rlm/core/store.py` | `Store`, `SpanRef`, `StoreObject` classes (isolated store) |
| `rlm/core/shared_store.py` | `SharedStore`, `WorkerStoreProxy`, `StoreEvent` (shared store) |
| `rlm/core/commit.py` | `Commit`, `parse_commit()`, `apply_commit()` |
| `rlm/environments/local_repl.py` | `LocalREPL` with `rlm_worker()`, `rlm_worker_batched()` |
| `rlm/core/types.py` | `REPLResult`, `CommitEvent`, `RLMChatCompletion` |
| `rlm/utils/prompts.py` | `COMMIT_PROTOCOL_PROMPT_ADDON`, `SHARED_STORE_PROMPT_ADDON` |
