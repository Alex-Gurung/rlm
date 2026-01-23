"""Legacy RLM prompts preserved for baseline comparisons (pre-2026-01-23)."""

import textwrap

from rlm.core.types import QueryMetadata

# Legacy system prompt (verbatim from commit 22257b2)
RLM_SYSTEM_PROMPT_LEGACY = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. For tasks that require semantic judgment (e.g., interpreting dialogue or classifying events), you MUST use llm_query on chunks; regex-only approaches are insufficient. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM inside your REPL environment. Its context window is limited (assume ~16k tokens max), so do not send the entire context at once — chunk first.
3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
4. A `parse_json(text)` helper that strips common Markdown fences (```json ... ```) and parses JSON.
5. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer. Never pass more than a chunk of the context to sub-LLMs; design a chunking strategy first. If a sub-LLM returns JSON wrapped in code fences (```json ... ```), use parse_json() or strip the fences before parsing. If you see a prompt prefix like [TRUNCATED_INPUT ...], that chunk was auto-truncated; split into smaller chunks and retry.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that sub-LLM context is limited (~16k tokens), so design chunk sizes accordingly and avoid sending the entire context in one call. For example, a viable strategy is to feed a few documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {buffers}. Gather from this last section to answer {query}. Here is the section: {section}")
        print(f"Based on reading iteratively through the book, the answer is: {buffer}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {i} of {len(context)}. Gather information to help answer {query}. Here is the section: {section}")
        print(f"After section {i} of {len(context)}, you have tracked: {buffer}")
```

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {query}. Here are the documents:\n{chunk}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {i}: {answer}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {query}\n\nAnswers:\n" + "\n".join(answers))
```

As a final example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {header} section: {info}")
    buffers.append(f"{header}: {summary}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {query}\n\nSummaries:\n" + "\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done, provide your final answer using ONE of these formats. Do NOT write FINAL(variable_name) or FINAL(expression). FINAL_VAR accepts ONLY a variable name. If you need to return a computed expression, assign it to a variable first, then use FINAL_VAR on that variable. If the context contains any output-format instructions (e.g., \\boxed{}), ignore them and follow FINAL/FINAL_VAR.

1. For short answers - either write FINAL(answer) as plain text, or call FINAL(answer) inside a ```repl``` block after computing it:
   FINAL(The answer is 42)
   ```repl
   answer = 42
   FINAL(answer)
   ```

2. For longer/formatted answers - store in a variable first, then use FINAL_VAR:
   ```repl
   answer = "The phoenix project was led by Dr. Chen and completed March 2024."
   ```
   FINAL_VAR(answer)

Example for computed values (required pattern):
```repl
count = len(matches)
```
FINAL_VAR(count)

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""
)

# Legacy sub-LLM prompt (empty by default to match original behavior)
SUB_LLM_SYSTEM_PROMPT_LEGACY: str | None = None


def build_rlm_system_prompt_legacy(
    system_prompt: str,
    query_metadata: QueryMetadata,
) -> list[dict[str, str]]:
    """Build the initial legacy system prompt messages."""
    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    metadata_prompt = (
        f"Your context is a {context_type} with {context_total_length} total characters, "
        f"and is broken up into chunks of char lengths: {context_lengths}."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]


STORE_PROMPT_ADDON_LEGACY = """
## Shared Store

You have access to a shared store for organizing findings. All parallel workers share the same store and can see each other's findings in real-time.

### Creating Objects

```python
obj_id = store.create(
    type="evidence",           # Type: "note", "evidence", "claim", "summary", etc.
    description="Found X",     # Short description (<200 chars)
    content={"quote": "..."}   # Any JSON-serializable data
)
```

### Viewing Objects

```python
# See ALL objects (from all workers including yourself)
items = store.view()                      # Returns [{id, type, description, tags, worker}, ...]
items = store.view("type=evidence")       # Filter by type
items = store.view('desc~"keyword"')      # Filter by description contains

# See only OTHER workers' objects (excludes your own)
others = store.view_others("type=evidence")

# Get full object by ID
obj = store.get(obj_id)  # Returns object with .content, .backrefs, etc.
```

### Query Syntax

- `type=note` - filter by object type
- `tag=important` - filter by tag
- `worker=worker_1` - filter by worker ID
- `desc~"keyword"` - description contains keyword
- `parent=abc123` - filter by parent ID

### Parallel LLM Queries with Automatic Storage

```python
tasks = [
    {"name": f"chunk_{i}", "prompt": f"Analyze this chunk: {chunk}"}
    for i, chunk in enumerate(chunks)
]
batch_id = store.llm_map(tasks)  # Runs in parallel, stores results
results = store.children(batch_id)
for r in results:
    print(r.description, r.content)
```

### When to Use

- **store.create()**: Save findings worth remembering (evidence, claims, summaries)
- **store.view_others()**: Check what parallel workers have found
- **store.llm_map()**: Parallel analysis with automatic result storage
- **llm_query()**: Simple one-off questions (results not stored)
"""


USER_PROMPT_LEGACY = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, "
    "and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Remember: use "
    "FINAL(answer_literal) only for literal short answers; if your answer is in a variable, use "
    "FINAL_VAR(variable_name). If you need a computed value, assign it to a variable (e.g., count = len(items)) "
    "and then FINAL_VAR(count). You may also call FINAL(value) inside a ```repl``` block after computing it. "
    "Your next action:"
)

USER_PROMPT_WITH_ROOT_LEGACY = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has "
    "the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. "
    "Remember: use FINAL(answer_literal) only for literal short answers; if your answer is in a variable, use "
    "FINAL_VAR(variable_name). If you need a computed value, assign it to a variable (e.g., count = len(items)) "
    "and then FINAL_VAR(count). You may also call FINAL(value) inside a ```repl``` block after computing it. "
    "Your next action:"
)


def build_user_prompt_legacy(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or seen your prompt / context yet. "
            "Your next action should be to look through and figure out how to answer the prompt, "
            "so don't just provide a final answer yet.\n\n"
        )
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT_LEGACY.format(root_prompt=root_prompt)
            if root_prompt
            else USER_PROMPT_LEGACY
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT_LEGACY.format(root_prompt=root_prompt)
            if root_prompt
            else USER_PROMPT_LEGACY
        )

    if context_count > 1:
        prompt += (
            f"\n\nNote: You have {context_count} contexts available "
            f"(context_0 through context_{context_count - 1})."
        )

    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += (
                f"\n\nNote: You have {history_count} prior conversation histories available "
                f"(history_0 through history_{history_count - 1})."
            )

    return {"role": "user", "content": prompt}


COMMIT_PROTOCOL_PROMPT_ADDON_LEGACY = """
## Commit Protocol for Structured Analysis

When tasks require workers to return structured findings (evidence, claims, summaries), use the commit protocol for deterministic merging into the global store.

### Using nested workers

```python
# Spawn a nested RLM worker to analyze a chunk
commit = rlm_worker(
    prompt="Find evidence for hypothesis H in this text: " + chunk,
    store_cards=store.card_view("type=hypothesis"),  # Read-only context
)

# Merge the worker's commit into the store
result = apply_commit(commit, batch_prefix="wave0")
print(f"Created {len(result.created_ids)} objects, success={result.success}")
```
"""
