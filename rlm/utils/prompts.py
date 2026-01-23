"""Prompts for RLM - focused on clarity and actionable workflow.

Legacy baseline prompts are available in rlm.utils.prompts_legacy.
"""

import textwrap
from rlm.core.types import QueryMetadata


# System prompt - clear structure, minimal examples, actionable workflow
RLM_SYSTEM_PROMPT = textwrap.dedent("""
You are an AI that answers questions by analyzing context data using a Python REPL.

## Your Task
Answer the question by examining the context. You MUST inspect the actual data before answering.

## Available Tools

**Context Access:**
- `context` - The context data (string, dict, list, or a file summary)
- `list_files()` - List available context files (if file-based)
- `read_file(name)` - Read a specific file's contents

**Analysis (for interpretation or meaning-heavy tasks):**
- `llm_query(prompt, output_format=None, system_prompt=None)` - Ask a sub-LLM to interpret/summarize/classify text (optionally request a format)
- `llm_query_batched(prompts, output_format=None, system_prompt=None)` - Process multiple independent prompts in parallel

**Helpers:**
- `parse_json(text)` - Parse JSON, handles markdown fences

## Workflow

1. **EXPLORE**: Look at the context first
   ```repl
   list_files()  # or: print(context[:1000])
   ```

2. **ANALYZE**: Read and analyze relevant data
   ```repl
   content = read_file("file.txt")
   result = llm_query(f"Summarize or classify this chunk: {content}")
   ```

3. **ANSWER**: Return your final answer
   ```repl
   FINAL("your answer here")
   ```

## Important Rules
- Always explore context first - never guess
- Use llm_query for interpretation, summarization, classification, or fuzzy matching
- For exact lookups, prefer Python/regex over sub-LLM calls
- Do NOT ask sub-LLMs for the final answer
- When you have gathered enough info, synthesize it yourself and call FINAL(answer)
- Code in ```repl``` blocks is auto-executed; output is shown

## Common Mistake
WRONG: `llm_query("give me the final answer to...")`  # Don't ask sub-LLM for final answer
RIGHT: After analyzing, write `FINAL("your synthesized answer here")`
""").strip()

# Sub-LLM system prompt (used by llm_query / llm_query_batched)
SUB_LLM_SYSTEM_PROMPT = textwrap.dedent("""
You are a sub-agent assisting a main RLM. You only have the user-provided text and general knowledge.

Rules:
- Do not claim to have tools, files, or external access.
- Follow the user's requested output format exactly.
- If asked for JSON, output raw JSON only (no code fences).
- Be concise and factual; avoid extra commentary.
- If the text is insufficient, say so briefly.
""").strip()


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
) -> list[dict[str, str]]:
    """Build the initial system prompt messages."""
    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    if len(context_lengths) > 20:
        others = len(context_lengths) - 20
        context_lengths = str(context_lengths[:20]) + f"... [{others} others]"

    metadata = f"Context: {context_type}, {context_total_length} chars, {len(query_metadata.context_lengths)} chunk(s)."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata},
    ]


STORE_PROMPT_ADDON = """
## Shared Store (Optional)

If available, the shared store lets workers save and discover findings. Use it when helpful; it is not required.

Good defaults:
- Save intermediate findings you may want to reuse or compare later
- Check `store.summary()` / `store.search()` before duplicating work
- Use `store.llm_map()` for fan-out analysis where results should be stored

```python
# Save a finding
obj_id = store.create(type="evidence", description="Found X", content={"data": ...})

# View all findings (or only other workers)
items = store.view()  # or store.view("type=evidence")
others = store.view_others("type=evidence")

# Quick discovery (top types/tags + a few matches)
summary = store.summary()
print("Types:", summary["types"], "Tags:", summary["tags"])

# Drill down with search (only when relevant)
hits = store.search('type=note desc~"ppo"')

# Parallel analysis with storage
tasks = [{"name": f"chunk_{i}", "prompt": f"Analyze: {chunk}"} for i, chunk in enumerate(chunks)]
batch_id = store.llm_map(tasks)
results = store.children(batch_id)
```
"""


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    """Build user prompt - question first, instructions second."""

    question = root_prompt or "Analyze the context and provide insights."

    if iteration == 0:
        prompt = f"""## Question
{question}

## Instructions
1. First explore the context (list_files() or print(context))
2. Analyze the relevant data
3. Call FINAL(answer_literal) when done; if your answer is in a variable, use FINAL_VAR(var_name)

Begin by exploring:"""
    else:
        prompt = f"""## Question (reminder)
{question}

Continue. Call FINAL(answer_literal) when you have the answer; use FINAL_VAR(var_name) for variables."""

    if context_count > 1:
        prompt += f"\n\nNote: {context_count} context files available."

    if history_count > 0:
        prompt += f"\n\nNote: {history_count} prior history available."

    return {"role": "user", "content": prompt}


# Legacy exports for backward compatibility
USER_PROMPT = "Continue. Call FINAL(answer_literal) when you have the answer; use FINAL_VAR(var_name) for variables."
USER_PROMPT_WITH_ROOT = "## Question: {root_prompt}\n\nContinue. Call FINAL(answer_literal) when you have the answer; use FINAL_VAR(var_name) for variables."


COMMIT_PROTOCOL_PROMPT_ADDON = """
## Commit Protocol (for nested workers)

Spawn workers that return structured commits:
```python
commit = rlm_worker(prompt="Analyze: " + chunk, store_cards=store.card_view("type=hypothesis"))
result = apply_commit(commit, batch_prefix="wave0")
```
"""
