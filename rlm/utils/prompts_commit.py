"""Commit-protocol prompt addon (only when commit protocol is enabled)."""

import textwrap


COMMIT_PROTOCOL_PROMPT_ADDON = textwrap.dedent("""
## Commit Protocol (for structured worker outputs)

When enabled, workers return structured commits instead of plain answers. This is an **opt-in feature** for advanced workflows.

### What is a Commit?

A commit is a structured JSON output that workers produce, containing:
- Objects to create (findings, evidence, summaries)
- Links between objects (supports, contradicts, refines)
- Proposed updates to existing objects

### Commit Structure

Workers should output JSON with this structure:
```json
{
  "commit_id": "unique_id",
  "creates": [
    {"type": "finding", "id": "f1", "description": "Key insight about X", "content": {"detail": "..."}}
  ],
  "links": [
    {"type": "supports", "src": "f1", "dst": "hypothesis_1"}
  ],
  "proposes_updates": [
    {"target": "existing_obj_id", "field": "content", "value": {...}}
  ]
}
```

### Fields

- `commit_id`: Unique identifier for this commit
- `creates`: List of objects to create in the store
  - `type`: Category (e.g., "finding", "evidence", "summary")
  - `id`: Local ID within this commit (becomes globally unique after apply)
  - `description`: Short description
  - `content`: Any JSON-serializable data
- `links`: Relationships between objects
  - `type`: Link type ("supports", "contradicts", "refines")
  - `src`: Source object ID
  - `dst`: Destination object ID
- `proposes_updates`: Suggested updates to existing objects

### Using Commits

```repl
# Spawn worker that returns a commit (requires worker_commit_prompt=True)
commit = rlm_worker("Analyze data and return findings as a commit")

# Apply commit to store (creates objects, establishes links)
result = apply_commit(commit, batch_prefix="wave0")
print(f"Created {len(result.created_ids)} objects")
print(f"Errors: {result.errors}")  # Check for any issues
```

### Link Types

- `supports` - Evidence supporting a hypothesis
- `contradicts` - Evidence against a hypothesis
- `refines` - Elaborates or updates another finding

### Example Worker Prompt (for commit-enabled workers)

When spawning a worker with commit protocol enabled, instruct it like:
```
Analyze the following data and return a JSON commit with your findings.
Create objects for each key insight with type="finding".
If findings support or contradict known hypotheses, add links.

Data: {data}
```
""").strip()
