# RLM Store & Batching Extensions

Hierarchical store with provenance tracking and hyper-parallel batching for RLM.

## What Was Built

A `store` object is now available in the REPL environment that provides:
- **Hierarchical storage**: Create structured notes, claims, summaries with parent/child relationships
- **Provenance tracking**: Link stored objects back to source spans in context
- **Parallel batching**: `store.llm_map()` runs multiple LLM queries concurrently and stores results

## Files Changed

| File | Description |
|------|-------------|
| `rlm/core/types.py` | Added `store_events`, `batch_calls` fields to `REPLResult` |
| `rlm/core/store.py` | NEW - `SpanRef`, `StoreObject`, `Store` class with `llm_map()` |
| `rlm/environments/local_repl.py` | Injects `store` and `SpanRef` as REPL globals |
| `rlm/utils/prompts.py` | Added `SMALL_MODEL_PROMPT_ADDON` constant |
| `rlm/core/rlm.py` | Added `store_prompt: bool = False` parameter |

## APIs Available in REPL

### Hierarchical Storage

```python
# Create an object
id = store.create(
    type="note",           # "note"|"claim"|"summary"|"batch_node"|"result"
    description="Short description for navigation",
    content={"any": "json-serializable data"},
    backrefs=[SpanRef("context_0", 100, 200)],  # optional provenance
    parents=["parent_id"],  # optional parent objects
    tags=["important"],     # optional tags for filtering
)

# Retrieve objects
obj = store.get(id)                    # Full object
children = store.children(id)          # Child objects
parents = store.parents(id)            # Parent objects

# Query/navigate (returns metadata only: {id, type, description, tags})
store.view()                           # All objects (newest first)
store.view("type=note")                # Filter by type
store.view("tag=important")            # Filter by tag
store.view("parent=abc123")            # Filter by parent
store.view('desc~"keyword"')           # Description contains
```

### Parallel Batching

```python
# Run multiple LLM queries in parallel, store results automatically
batch_id = store.llm_map(
    tasks=[
        {"name": "task1", "prompt": "Analyze chunk 1...", "description": "Chunk 1 analysis"},
        {"name": "task2", "prompt": "Analyze chunk 2...", "description": "Chunk 2 analysis"},
        # ... more tasks
    ],
    parent="optional_parent_id",  # Link batch to a parent object
    model="optional_model_name",  # Use specific model
)

# Results are stored as children of the batch node
results = store.children(batch_id)
for r in results:
    print(r.description, r.content)
```

### Provenance with SpanRef

```python
# Reference a span in context
ref = SpanRef(
    source_id="context_0",  # or file path like "/path/to/file.txt"
    start=100,
    end=200,
    unit="chars",  # or "lines"
)

# Attach to created objects
store.create("claim", "Found answer", "42", backrefs=[ref])
```

## Usage

### Enable Store Instructions in Prompt

```python
from rlm import RLM

rlm = RLM(
    backend="vllm",
    backend_kwargs={"model_name": "..."},
    store_prompt=True,  # Appends store/batching instructions to system prompt
)

result = rlm.completion("Analyze these 20 documents and find...")
```

When `store_prompt=True`, the model receives **directive** instructions requiring it to:
- Use `store.llm_map()` for 3+ similar queries - **NEVER** use loops with `llm_query()`
- Includes a concrete example matching common fan-out patterns
- Results are auto-stored and accessible via `store.children(batch_id)`

### Logging

Events are logged in JSONL via `REPLResult`:

```python
# store_events: operations on the store
{"op": "create", "id": "abc123", "type": "note", "description": "...", "ts": 1234.5}

# batch_calls: llm_map executions
{"prompts_count": 8, "model": "qwen", "execution_time": 2.3, "ts": 1234.5}
```

## Design Decisions

- **Environments**: LocalREPL only (remote envs deferred - need script serialization)
- **Persistence**: In-memory per `completion()` call (when `persistent=False`). If `persistent=True`, the store persists across completion calls for that REPL instance.
- **Prompts**: `store_prompt=True` is opt-in; default behavior unchanged

## TODO / Future Plans

- [ ] Benchmark: Compare `store_prompt=True` vs `False` on OOLONG / fan-out tasks
- [ ] Visualizer: Add a hierarchy/tree view (store/batch event lists are already visible)
- [ ] Remote environment support
- [ ] `store.llm_reduce()` - aggregate child results with LLM
- [ ] Persistence across `completion()` calls
- [ ] Store export/import for debugging
