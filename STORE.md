# Shared Store & Batching

Shared, append-only store for coordinating parallel RLM workers and batching sub-LLM calls.

## What It Is

The REPL environment can expose a `store` object (a `WorkerStoreProxy`) that writes to a
shared `SharedStore`. This lets parallel workers see each other's findings in real time,
while keeping writes append-only and attributable by worker ID.

Key capabilities:
- **Hierarchical storage**: notes, claims, summaries with parent/child relationships
- **Provenance tracking**: attach `SpanRef` backrefs to source spans
- **Discovery**: `store.search()` and `store.summary()` for lightweight lookup
- **Batching**: `store.llm_map()` runs parallel sub-LLM calls and stores results

## Files (Core)

| File | Description |
|------|-------------|
| `rlm/core/shared_store.py` | `SharedStore`, `WorkerStoreProxy`, append-only event log |
| `rlm/core/store.py` | `Store`, `SpanRef`, `StoreObject` utilities |
| `rlm/environments/local_repl.py` | Injects `store` + commit helpers into REPL |
| `rlm/core/rlm.py` | `store_mode` wiring and prompt add-ons |
| `rlm/utils/prompts.py` | `STORE_PROMPT_ADDON`, `COMMIT_PROTOCOL_PROMPT_ADDON` |

## APIs Available in REPL

### Create / Query

```python
obj_id = store.create(
    type="note",
    description="Short description",
    content={"any": "json-serializable data"},
    backrefs=[SpanRef("context_0", 100, 200)],  # optional provenance
    parents=["parent_id"],  # optional parent objects
    tags=["important"],  # optional tags for filtering
)

items = store.view("type=note")          # metadata only
others = store.view_others("type=note")  # exclude your worker's objects
hits = store.search('desc~"ppo"')        # search by metadata
summary = store.summary()                # top types/tags + sample matches
```

### Batch Sub-LLM Calls

```python
batch_id = store.llm_map(
    tasks=[
        {"name": "chunk_1", "prompt": "Analyze chunk 1"},
        {"name": "chunk_2", "prompt": "Analyze chunk 2"},
    ],
)

results = store.children(batch_id)
for r in results:
    print(r.description, r.content)
```

## Usage

### Shared Store (default)

```python
from rlm import RLM

rlm = RLM(
    backend="vllm",
    backend_kwargs={"model_name": "..."},
    store_mode="shared",  # default
)
```

### Disable Store (baseline)

```python
rlm = RLM(
    backend="vllm",
    backend_kwargs={"model_name": "..."},
    store_mode="none",  # no store, no rlm_worker helpers
)
```

**Prompting note:** The shared-store prompt is *suggestive*, not directive. Models are encouraged
to use the store when helpful, but nothing forces it.

## Logging

Store events are captured in `REPLResult.store_events`:

```json
{"op": "create", "id": "worker_A_000001", "type": "note", "description": "...", "ts": 1234.5}
```

Batch calls are captured in `REPLResult.batch_calls`:

```json
{"prompts_count": 8, "model": "qwen", "execution_time": 2.3, "ts": 1234.5}
```

## Design Notes

- **Per-completion store**: every root completion gets a fresh store, even in `persistent=True` mode.
- **Shared across workers**: nested workers see the same store via `WorkerStoreProxy`.
- **Store optional**: `store_mode="none"` disables store and commit helpers.

## TODO / Future Plans

- Benchmark shared-store vs baseline on OOLONG / fan-out tasks
- Interactive store/tree visualization in the visualizer
- Remote environment support
- `store.llm_reduce()` for aggregation
- Store export/import for debugging
