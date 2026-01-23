# Handoff doc: Hierarchical + Hyper-parallel extensions to `alexzhang13/rlm`

This is written so you can drop it directly to an implementation agent. It covers: (1) what the repo/paper already do, (2) what *new behavior* we want to encourage, (3) concrete feature additions (minimal → ambitious), and (4) benchmarks + experiment plan with small models.

---

## 0) Executive summary

**What RLMs already are:** an inference-time paradigm where the *input context* is placed into an external REPL environment as a variable, and the model iteratively writes code to inspect/filter/chunk it and optionally make sub-LM calls over selected snippets. The paper frames this as treating long prompts as part of an external environment and allowing programmatic examination + recursive subcalls. ([ar5iv][1])

**What the repo already provides:** a Python REPL environment whose globals include `context`, `llm_query()`, `llm_query_batched()`, and `FINAL_VAR()`, plus local/docker/modal execution and a JSONL trajectory logger + visualizer. ([Alex L. Zhang][2])

**What we want to add:** *lightweight, inspectable structure* for:

* hierarchical understanding (books/repos/corpora),
* provenance/backreferences,
* and hyper-parallel “map-style” subcalls (using batching) **without** forcing a rigid index.

**Core idea:** keep **sources + store + views** as the primary primitives (your preference), treat any “index” as just a *derived view*, and provide a default “tree-of-work” structure so batched outputs aren’t just an unlabeled list.

---

## 1) What the paper actually did (and what it did not)

### 1.1 Benchmarks and what they stress

The paper evaluates RLMs on several long-context tasks with different failure modes: ([ar5iv][1])

* **BrowseComp-Plus (1K docs)**: multi-hop deep-research QA over a curated corpus; answer requires piecing together several documents. ([ar5iv][1])
* **OOLONG (trec_coarse split)**: semantic transform of (nearly) all entries + aggregation; costs scale linearly with input length. ([ar5iv][1])
* **OOLONG-Pairs**: like OOLONG but requires aggregation over **pairs**; quadratic pressure. ([ar5iv][1])
* **LongBench-v2 CodeQA**: multiple-choice repo understanding over a fixed number of files. ([ar5iv][1])

### 1.2 Small-model-ish / non-frontier behavior matters (even for huge open models)

They explicitly observe *model-dependent* context management policies and that Qwen3-Coder’s RLM behavior can be pathological (too many subcalls), requiring an extra warning line in the system prompt. ([ar5iv][1])
They also show an example where Qwen3-Coder does “semantic transform as a separate subcall per line,” which is exactly the kind of behavior we want to discourage and replace with batching/hierarchy. ([ar5iv][1])

### 1.3 The paper’s own note on parallelism

They explicitly note runtime could be “significantly improved through asynchrony of LM calls” (and prompting to discourage long subcalls). ([ar5iv][1])
So: **hyper-parallel RLM** is not a random idea—it’s a clearly identified bottleneck.

### 1.4 “Can’t the model just create dicts/functions already?”

Yes. In their RLM prompt, the LM is instructed to use a REPL, can write Python, can build “buffers” in variables, and can return `FINAL_VAR(var_name)` for long outputs. ([ar5iv][1])

**But** the critical practical limitation is: the *root LM only sees what gets surfaced* through the REPL transcript (prints, truncated outputs, etc.). The paper even warns REPL outputs are truncated, pushing models toward subcalls or summarized buffers. ([ar5iv][1])
So while “the model can create any dict it wants,” **those dicts are not automatically discoverable/inspectable** across a long run unless you also build a convention for surfacing them.

That’s the real justification for “objects with descriptions/backrefs”:

* Not “because Python can’t store it,”
* but because we want **stable, queryable, inspectable state** that both *the model* and *humans/tools* can reliably navigate.

---

## 2) Current repo state (what exists today)

From the project docs and repo structure:

### 2.1 Repo structure you can count on

The GitHub root shows major folders: `docs/`, `examples/`, `rlm/`, `tests/`, and `visualizer/`. ([GitHub][3])
So you have an obvious place to add: core memory/store code under `rlm/` + new examples + tests + visualizer enhancements.

### 2.2 Public API surface

* The “main class” is `RLM`, used as `rlm = RLM(...); rlm.completion(prompt)`; prompt becomes the REPL `context` variable. ([Alex L. Zhang][4])
* `max_depth` is currently called out as “TODO” with only `max_depth=1` supported. ([Alex L. Zhang][4])
  This is good: you can focus on depth=1 hyper-parallel + hierarchy first.

### 2.3 REPL globals (important)

The docs explicitly list REPL globals: ([Alex L. Zhang][2])

* `context`
* `llm_query(prompt, model=None)`
* `llm_query_batched(prompts, model=None)` → returns a **list of completion strings**
* `FINAL_VAR(var_name)`

### 2.4 Environments + comms architecture

Repo supports local/docker (socket-based) and modal (HTTP broker pattern) execution. ([Alex L. Zhang][2])
This matters because “batched subcalls” needs to be implemented consistently across these backends to measure real speedups.
**Current scope decision:** implement store + batching only for `LocalREPL` first; defer Docker/Modal/Prime until we’re confident in the semantics.

### 2.5 Release note mismatch to flag

The GitHub release notes mention support for `llm_query` and `batch_llm_query` “for multiple asynchronous subcalls.” ([GitHub][5])
Docs use the name `llm_query_batched`. ([Alex L. Zhang][2])

**Action for agent:** reconcile naming (alias one to the other), and treat it as a UX issue: the primitive exists; we want to wrap it into a structured result.
**Revised decision:** standardize on `llm_query_batched` and **do not** add aliases; keep naming explicit and consistent.

---

## 3) What we’re trying to add (behavioral goals)

You asked for “really clear what additional features we’re trying to add / what behavior we’re trying to encourage.”

### 3.1 Desired behaviors

1. **Map → reduce decomposition by default**, especially on OOLONG/OOLONG-Pairs:

   * build a task list in code,
   * batch subcalls,
   * aggregate in Python.

2. **Stable “navigability” of intermediate results**:

   * no more “contents[10] is ???” without metadata.

3. **Hierarchical understanding as a first-class artifact**:

   * books: chapter → section → page → span
   * repos: module → file → symbol → span
   * corpora: doc set → doc → snippet → claim

4. **Provenance/backrefs everywhere**:

   * answers cite object IDs which point to spans in sources
   * reduces “summary drift” and makes BrowseComp-Plus joins much more reliable.

5. **Small-model guardrails**:

   * prevent “one subcall per line” behavior (explicitly observed as a model failure mode). ([ar5iv][1])

### 3.2 Non-goals (at least for MVP)

* Don’t force a heavyweight embedding index upfront.
* Don’t require a specific chunking strategy.
* Don’t hardcode task-specific heuristics (keep it mostly task-agnostic like the paper’s spirit). ([ar5iv][1])

---

## 4) Proposed primitives: **Sources + Store + Views** (no mandatory “index”)

This matches your intuition: avoid a magical “index,” but still enable derived views that *act* like lightweight indexes.

### 4.1 Source objects (immutable)

Represent what the model is “allowed to read”:

* `SourceDoc`: `{source_id, kind, uri/path, content_handle, metadata}`
* `SpanRef`: `{source_id, start, end, unit="chars|lines", preview}`

Keep it simple; for now just support:

* `TextSpanRef(doc_id, char_start, char_end)`
* `FileSpanRef(path, line_start, line_end)`

### 4.2 Store objects (mutable, but versioned)

A minimal record:

```text
Object {
  id: str
  type: str                       # "note"|"claim"|"summary"|"node"|...
  description: str                # short catalog entry
  content: (str | dict | list)    # JSON-serializable preferred
  backrefs: list[SpanRef]
  parents: list[str]
  children: list[str]
  tags: list[str]
  version: int
  updated_at: timestamp
}
```

**Key:**

* `description` is not just a summary; it’s a *catalog card* for navigation.
* parent/child makes it a “tree by default,” but keep it a DAG (multiple parents allowed).

### 4.3 Views (derived, queryable projections)

A “view” is just a filtered projection over the store, e.g.:

* `view("type=claim tag=auth")` → returns `{id, description, backrefs}`
* `view("parent=<node_id>")` → shows children summaries

This gives you “index-like affordances” without committing to embeddings.

**Design constraint:** views should default to returning *metadata only* (IDs + descriptions), not huge content, to avoid context blow-ups.

---

## 5) Hyper-parallel RLM: make batching *the default ergonomic path*

### 5.1 What exists today

* Repo exposes `llm_query_batched(prompts, model=None)` and says it runs **concurrent** sub-LM queries, returning a list of strings. ([Alex L. Zhang][2])
* Release notes emphasize “multiple asynchronous subcalls.” ([GitHub][5])
* Paper highlights the runtime bottleneck and calls out asynchrony as an improvement path. ([ar5iv][1])

### 5.2 The core UX problem you identified

A raw list return is “unorganized.” True.

So don’t change `llm_query_batched` semantics immediately—wrap it.

### 5.3 Proposed wrapper: `store.batch() -> BatchNode`

Add a REPL-global helper, e.g.:

* `batch_node_id = store.llm_map(tasks, parent=None, model=None, task_type="chunk_summarize")`

Where `tasks` is a list of dicts:

```python
tasks = [
  {"name": "chunk_000", "description": "...", "prompt": "...", "backrefs": [...]},
  ...
]
```

**Returns:** a `BatchNode` object (stored in `store`) whose `children` are result nodes:

* each child contains `{task_name, response, usage(optional), backrefs, description}`

Now you automatically get:

* `contents[i].description`
* `contents[i].parent`
* `contents[i].children`

…without forcing the LM to invent its own structuring every time.

### 5.4 Why this is more than “agents can already write to a file”

Sure, an agent can write notes to a file, or a DB.

But the distinguishing research advantage here is:

* the store is **inside the same control loop** as the RLM,
* it’s **versioned and logged** (so you can train/ablate),
* and it’s coupled to `llm_query_batched` so you get a *first-class work graph* instead of an ad-hoc scratchpad.

This matters if your goal is: “take a small model and make it *really good* at these benchmarks.” You’ll want:

* stable intermediate artifacts,
* easy-to-learn conventions,
* and objective logging of decisions.

---

## 6) Prompting changes (especially for small models like Qwen3 4B)

The paper already shows you need model-specific nudges: they added a warning line for Qwen3-Coder to prevent runaway subcalls. ([ar5iv][1])

### 6.1 Add a *budgeted* subcall policy to the default system prompt

For small models, add explicit constraints like:

* **Subcall budget:** `max_total_subcalls = 64` (configurable)
* **Batching rule:** “If you have ≥ 8 similar subqueries, you must use batching.”
* **Chunking guidance:** “Never do one subcall per line/row; group into chunks of ~N lines.”

This directly targets the observed failure mode (“subcall per line”). ([ar5iv][1])

### 6.2 Encourage “map/reduce” structure in the prompt

Add an explicit recipe (short and memorable):

1. **Plan**: decide what you need (what artifacts to create).
2. **Map**: create task list → `store.llm_map(...)` (batched).
3. **Reduce**: aggregate in Python → store derived claims/summaries.
4. **Verify**: use a small number of targeted checks.
5. **Answer**: `FINAL_VAR(...)`.

### 6.3 Prompt “contracts” that improve reliability

To make the store useful and prevent junk:

* Every new object must have:

  * `description` (<= 200 chars),
  * and if it asserts facts, at least one `SpanRef` backref.

This is the simplest “evidence discipline” you can add without turning it into a heavy framework.

---

## 7) Benchmarks: what to run first (cheap → expensive), using their suite

You said you’ll use their benchmarks first, and you want to see performance differences.

### 7.1 Fastest “signal” benchmarks for your new features

**A) OOLONG (trec_coarse split)**
It stresses: semantic transform over lots of entries + aggregation. ([ar5iv][1])
It is *perfect* for:

* batching improvements,
* and “tree of intermediate outputs.”

Dataset availability: `oolongbench/oolong-real` on Hugging Face. ([Hugging Face][6])

**B) OOLONG-Pairs**
It stresses: combinatorics + avoiding quadratic explosion. ([ar5iv][1])
It is perfect for:

* encouraging “factorization” strategies (compute per-user labels once, then do pairs in Python),
* and discouraging runaway subcalls.

The RLM paper provides additional details in its appendix (and a PDF version exists). ([arXiv][7])

### 7.2 Next: code understanding

**C) LongBench-v2 CodeQA**
It stresses: hierarchical repo understanding over a fixed file set. ([ar5iv][1])
Dataset availability: `zai-org/LongBench-v2` on Hugging Face. ([Hugging Face][8])

This is where:

* “module/file/symbol tree objects”
* plus “SpanRef(file:lines)”
  should show obvious wins.

### 7.3 Expensive / slower: deep research

**D) BrowseComp-Plus (1K docs)**
It stresses: multi-hop joins and provenance over many docs. ([ar5iv][1])
Dataset availability and description (fixed curated corpus) appear via:

* GitHub `texttron/BrowseComp-Plus`. ([GitHub][9])
* HF datasets like `Tevatron/browsecomp-plus`. ([Hugging Face][10])

This is where the “claim objects with backrefs” + “link nodes” will matter most.

### 7.4 Recommended experiment ladder (for quick iteration)

Run in this order:

1. OOLONG small subset (e.g., 20–50 tasks)
2. OOLONG-Pairs (all 20 tasks)
3. CodeQA subset (e.g., 50 tasks)
4. BrowseComp-Plus small sample (e.g., 20–50 tasks), then scale

---

## 8) Concrete ablations to prove value (so this isn’t just “nice engineering”)

Your question: “Is this actually interesting research, or just something agents can already do?”

Make it research by setting up sharp hypotheses + ablations.

### 8.1 Hypotheses

H1) **Structured batched outputs** (BatchNode + child nodes) improves accuracy/cost on OOLONG vs vanilla `llm_query_batched` list-of-strings, especially for small models.

H2) **Hierarchical store + views** improves CodeQA by enabling stable repo maps and reducing rereads.

H3) **Provenance contracts** improve BrowseComp-Plus multi-hop accuracy and reduce “confident wrong joins.”

H4) These gains are **larger for small models**, because small models are more likely to:

* over-subcall,
* lose track of state,
* and fail to merge results coherently (consistent with the model-dependent behavior observed in the paper). ([ar5iv][1])

### 8.2 Minimal ablation matrix

For each benchmark:

* **Baseline:** vanilla repo RLM prompt + `llm_query` only
* **+ batching prompt:** instruct to prefer `llm_query_batched`
* **+ BatchNode wrapper:** same prompt but `store.llm_map` exists
* **+ hierarchy store:** require all derived info stored as nodes/claims with descriptions/backrefs

Measure:

* accuracy,
* total subcalls,
* average batch size,
* runtime,
* and cost.

---

## 9) Implementation plan (phased, minimal-risk)

### Phase 0 — “Locate + instrument” (½–1 day)

**Goal:** add zero new behavior; just prove you can log it (per‑iteration, in the existing JSONL).

Tasks:

* Identify where REPL globals are injected (`context`, `llm_query`, `llm_query_batched`, `FINAL_VAR`). (Docs guarantee these exist.) ([Alex L. Zhang][2])
* Extend REPLResult to carry per‑iteration:

  * `batch_calls` (prompt count, model, timing)
  * `store_events` (create/update with IDs + descriptions)

Justification: you want to see whether small models are exploding calls (like Qwen3-Coder does in the paper). ([ar5iv][1])

### Phase 1 — Add `store` + node schema (MVP) (1–2 days)

**Goal:** introduce an inspectable object system, without changing RLM behavior.
**Scope:** store is in‑memory per `completion()` call; events are logged through existing JSONL (no separate store persistence yet).

Deliverables:

* `store.create(type, description, content, backrefs=[], parents=[], tags=[]) -> id`
* `store.get(id)`, `store.list(...)`, `store.children(id)`, `store.parents(id)`
* `store.view(query, limit=...)` returning metadata table

Hard rule:

* store objects are JSON-serializable unless explicitly marked.

### Phase 2 — Add `store.llm_map(...)` wrapper (1–2 days)

**Goal:** solve the “batched outputs are an unlabeled list” problem.

Implementation sketch:

* Accept `tasks: list[dict]` with `name`, `description`, `prompt`, optional `backrefs`
* Internally call `llm_query_batched([t["prompt"] for t in tasks])`
* Create one parent object: `BatchNode`
* Create children objects: `LMResultNode` for each completion
* Return parent ID (or a small Python object containing IDs)

Now the model can:

* browse results by description,
* and relate them via parent/child.

### Phase 3 — Prompt pack for small models (½–1 day)

Create `custom_system_prompt` presets (repo supports override). ([Alex L. Zhang][4])

Two variants:

* `prompt_default` (for strong models)
* `prompt_small_model` (strict budgets + batching rules + “don’t subcall per line” warning)

This is justified directly by the paper’s findings about Qwen3-Coder needing such a warning. ([ar5iv][1])

### Phase 4 — Visualizer: show “work tree” (1–3 days)

Repo already supports trajectory visualization. ([Alex L. Zhang][11])

Add:

* BatchNode as a node type
* Child result nodes as leaves
* (later) Claim nodes linked to SpanRefs

This turns debugging from “read a transcript” into “inspect a tree.”

### Phase 5 — Benchmark harness (time-boxed)

Even if repo lacks a full harness, you can build a thin runner:

* load HF dataset sample,
* call `rlm.completion()` with the expected context structure,
* parse answers,
* compute accuracy / EM / F1.

Dataset pointers:

* OOLONG: HF `oolongbench/oolong-real`. ([Hugging Face][6])
* LongBench-v2: HF `zai-org/LongBench-v2`. ([Hugging Face][8])
* BrowseComp-Plus: GitHub and HF corpora exist. ([GitHub][9])

---

## 10) Model choices + backend notes (Qwen 3 4B, Qwen3-Coder-30B-A3B)

Repo supports multiple backends including local `vLLM`, and a generic OpenAI-compatible endpoint. ([Alex L. Zhang][12])
So you can run Qwen locally via vLLM and point `RLM(backend="vllm", backend_kwargs={"base_url": ..., "model_name": ...})`. ([Alex L. Zhang][12])

### Practical advice for small models

Small models often fail in two ways:

1. **They over-decompose** (too many subcalls) — observed even for Qwen3-Coder 480B unless warned. ([ar5iv][1])
2. **They can’t merge** the outputs coherently.

Your new structure addresses both:

* batching reduces call overhead and discourages “one per line,”
* store descriptions + parent/child structure reduces merge confusion.

---

## 11) Where this helps vs. where RLMs remain weak (research framing)

Even with hierarchy + batching, RLMs still struggle on “global property” tasks where relevance is superadditive (A and B look irrelevant alone, but A+B matters). The paper’s trajectories mostly show simple strategies like uniform chunking and keyword searches, not sophisticated partitioning. ([ar5iv][1])

**How hierarchy helps (but doesn’t magically solve it):**

* Hierarchical “global sketch” nodes (chapter/module summaries) increase the chance A and B are linked.
* Link nodes (“A mentions X”, “B defines X”) make multi-hop joins explicit.

This becomes a genuine research direction if you test:

* RLM vs RLM+hierarchy on tasks that require multi-hop joins and global consistency.

---

## 12) Optional: narrative + book-length evaluation (NarraBench tie-in)

You mentioned NarraBench as potentially relevant for book-length analysis/generation. NarraBench is a “theory-informed taxonomy of narrative-understanding tasks” and a framework/survey of narrative benchmarks. ([arXiv][13])

Why it’s relevant to this project:

* It highlights narrative dimensions (events, style, perspective, revelation) that are **not** well covered by many existing deterministic benchmarks. ([arXiv][13])
* Those dimensions are exactly where “hierarchical representations + global consistency” matter.

I’d treat NarraBench as a **Phase 2 evaluation expansion** (after you’ve proven wins on OOLONG/CodeQA/BrowseComp-Plus).

---

## 13) Summary: what to tell the implementation agent to do first

If you want a crisp “do this now” list:

1. **Add `store`** to REPL globals (create/get/list/view; objects have description + parent/child + backrefs).
2. **Add `store.llm_map()`** wrapper that calls `llm_query_batched` and stores results as a BatchNode tree (fixes “list of strings is unorganized”).
3. **Add small-model prompt preset**: budgets + batching rule + explicit prohibition on “subcall per line” (motivated by paper’s observed Qwen behavior). ([ar5iv][1])
4. **Run OOLONG + OOLONG-Pairs first** (fast signal), then CodeQA, then BrowseComp-Plus (expensive). ([ar5iv][1])
5. **Log & visualize** BatchNodes so you can debug behavior changes.

This gives you the cleanest path to: “small model gets unusually good on RLM benchmarks,” because it reduces the two biggest bottlenecks the paper itself flags: runaway subcalls and sequential runtime. ([ar5iv][1])

[1]: https://ar5iv.org/abs/2512.24601 "[2512.24601] Recursive Language Models"
[2]: https://alexzhang13.github.io/rlm/environments/ "Recursive Language Models"
[3]: https://github.com/alexzhang13/rlm "GitHub - alexzhang13/rlm: General plug-and-play inference library for Recursive Language Models (RLMs), supporting various sandboxes."
[4]: https://alexzhang13.github.io/rlm/api/ "Recursive Language Models"
[5]: https://github.com/alexzhang13/rlm/releases?utm_source=chatgpt.com "Releases · alexzhang13/rlm"
[6]: https://huggingface.co/datasets/oolongbench/oolong-real?utm_source=chatgpt.com "oolongbench/oolong-real · Datasets at ..."
[7]: https://arxiv.org/pdf/2512.24601.pdf?curius=3971&utm_source=chatgpt.com "Recursive Language Models"
[8]: https://huggingface.co/datasets/zai-org/LongBench-v2?utm_source=chatgpt.com "zai-org/LongBench-v2 · Datasets at Hugging Face"
[9]: https://github.com/texttron/BrowseComp-Plus?utm_source=chatgpt.com "texttron/BrowseComp-Plus"
[10]: https://huggingface.co/datasets/Tevatron/browsecomp-plus?utm_source=chatgpt.com "Tevatron/browsecomp-plus · Datasets at Hugging Face"
[11]: https://alexzhang13.github.io/rlm/trajectories/ "Recursive Language Models"
[12]: https://alexzhang13.github.io/rlm/backends/ "Recursive Language Models"
[13]: https://arxiv.org/html/2510.09869v1 "https://arxiv.org/html/2510.09869v1"
