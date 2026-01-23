from __future__ import annotations

import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.commit import Commit, MergeResult, apply_commit, parse_commit
from rlm.core.store import SpanRef, Store
from rlm.core.types import CommitEvent, REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv

if TYPE_CHECKING:
    from rlm.core.rlm import RLM
    from rlm.logger import RLMLogger

# =============================================================================
# Safe Builtins
# =============================================================================

# Safe builtins - blocks dangerous operations like eval/exec/input
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.
    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        max_depth: int = 2,
        sub_llm_max_chars: int = 64000,
        subcall_budget: int = 64,
        backend: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        logger: "RLMLogger | None" = None,
        **kwargs,
    ):
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.lm_handler_address = lm_handler_address
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0
        self.sub_llm_max_chars = sub_llm_max_chars

        # Depth tracking for nested workers
        self.max_depth = max_depth

        # Subcall budget enforcement
        self.subcall_budget = subcall_budget
        self._subcall_count: int = 0

        # Backend config for spawning nested RLM workers
        self._backend = backend
        self._backend_kwargs = backend_kwargs or {}

        # Logger for hierarchical logging (passed to nested workers)
        self._logger = logger

        # Track commit events during code execution
        self._pending_commit_events: list[CommitEvent] = []

        # Setup globals, locals, and modules in environment.
        self.setup()

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment."""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []

        # Create store for hierarchical data management
        self.store = Store()
        self.store.set_llm_batch_fn(self._llm_query_batched)

        # Add helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["FINAL"] = self._final  # Also allow FINAL() in code
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["parse_json"] = self.parse_json
        self.globals["store"] = self.store
        self.globals["SpanRef"] = SpanRef

        # Commit protocol globals
        self.globals["rlm_worker"] = self._rlm_worker
        self.globals["parse_commit"] = parse_commit
        self.globals["apply_commit"] = self._apply_commit
        self.globals["Commit"] = Commit

    def _final_var(self, variable_name: str) -> str:
        """Return the value of a variable as a final answer."""
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        return f"Error: Variable '{variable_name}' not found"

    def _final(self, answer: str) -> str:
        """Return the answer directly (allows FINAL() in code blocks).

        Prints FINAL(answer) to stdout so the parser can detect it.
        """
        print(f"FINAL({answer})")
        return str(answer)

    def parse_json(self, text: str) -> Any:
        """Parse JSON from a string, stripping common Markdown fences."""
        import json

        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Strip outer fences, and optional language tag like ```json
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

    def _rlm_worker(
        self,
        prompt: str,
        store_cards: list[dict] | None = None,
        worker_config: dict | None = None,
    ) -> dict:
        """
        Spawn a nested RLM worker.

        The nested worker's environment runs at depth+1 relative to this environment.
        If max_depth is reached, returns an error commit instead.

        Args:
            prompt: The task prompt for the worker
            store_cards: Optional store cards to include as context
            worker_config: Optional config dict with keys:
                - max_iterations: int (default 2)
                - timeout: float (default 60.0)
                - inherit_context: bool (default True)

        Returns:
            Commit dict (parsed from worker's response)
        """
        worker_config = worker_config or {}

        # Check depth limit
        if self.depth >= self.max_depth:
            return {
                "commit_id": "depth_exceeded",
                "creates": [],
                "error": f"Max depth reached (depth={self.depth}, max_depth={self.max_depth})",
            }

        # Check subcall budget
        self._subcall_count += 1
        if self._subcall_count > self.subcall_budget:
            return {
                "commit_id": "budget_exceeded",
                "creates": [],
                "error": f"Subcall budget exceeded ({self.subcall_budget})",
            }

        # Check if we have backend config
        if not self._backend:
            return {
                "commit_id": "no_backend",
                "creates": [],
                "error": "No backend configured for nested workers",
            }

        try:
            # Import RLM here to avoid circular imports
            from rlm.core.rlm import RLM

            # Build worker prompt and context payload
            worker_prompt = self._build_worker_prompt(prompt)
            worker_context: dict[str, Any] = {
                "task": prompt,
                "depth": self.depth,
                "max_depth": self.max_depth,
            }
            if store_cards:
                worker_context["store_cards"] = store_cards
            if worker_config.get("inherit_context", True) and "context" in self.locals:
                parent_context = self.locals["context"]
                # Enforce JSON-serializable context for the worker
                json.dumps(parent_context)
                worker_context["parent_context"] = parent_context

            # Create child logger for hierarchical logging
            child_logger = None
            if self._logger:
                child_logger = self._logger.create_child_logger(child_depth=self.depth + 1)
                # Log worker spawn event
                self._logger.log_worker_spawn(
                    child_run_id=child_logger.run_id,
                    worker_prompt=prompt[:500] if prompt else "",
                )

            # Create nested RLM
            worker_rlm = RLM(
                backend=self._backend,
                backend_kwargs=self._backend_kwargs,
                environment="local",
                depth=self.depth,  # Worker starts at current depth
                max_depth=self.max_depth,
                max_iterations=worker_config.get("max_iterations", 2),
                verbose=False,
                logger=child_logger,  # Pass child logger for hierarchical logging
            )

            # Execute with timeout (via max_iterations limit)
            result = worker_rlm.completion(worker_context, root_prompt=worker_prompt)

            # Log worker completion
            if self._logger and child_logger:
                self._logger.log_worker_complete(
                    child_run_id=child_logger.run_id,
                    result_summary=result.response[:200] if result.response else None,
                )

            # Parse commit from response
            commit = parse_commit(result.response, fallback_id=f"worker_{self.depth}")

            # Track the LLM call
            self._pending_llm_calls.append(result)

            return commit.to_dict()

        except Exception as e:
            return {
                "commit_id": f"worker_error_{self.depth}",
                "creates": [],
                "error": f"Worker failed: {e}",
            }

    def _build_worker_prompt(self, prompt: str) -> str:
        """Build the worker prompt with task and commit instructions."""
        parts = []

        parts.append("## Context")
        parts.append(
            "Use the REPL `context` variable (a dict) for task data. "
            "It may include `task`, `store_cards`, `parent_context`, `depth`, and `max_depth`."
        )
        parts.append("")

        parts.append("## Task")
        parts.append(prompt)
        parts.append("")
        parts.append("## Method Notes")
        parts.append("- store.create(type, description, content, ...) -> returns an id string")
        parts.append("- store.card_view(query) -> small list of ids + descriptions")
        parts.append("- llm_query / llm_query_batched are available; check context['depth'] and context['max_depth']")
        parts.append("- To finish, output the JSON commit via FINAL(commit_dict) or assign then FINAL_VAR(var_name)")
        parts.append("## Instructions")
        parts.append("Return a JSON commit with your findings. Format:")
        parts.append('```json')
        parts.append('{')
        parts.append('  "commit_id": "your_id",')
        parts.append('  "creates": [{"type": "...", "id": "...", "description": "...", "content": ...}],')
        parts.append('  "links": [],')
        parts.append('  "proposes_updates": []')
        parts.append('}')
        parts.append('```')

        return "\n".join(parts)

    def _apply_commit(self, commit: Commit | dict, batch_prefix: str = "") -> MergeResult:
        """
        Apply a commit to the store and track the event.

        Args:
            commit: Commit object or dict
            batch_prefix: Optional prefix for namespacing

        Returns:
            MergeResult with created IDs and any errors
        """
        # Convert dict to Commit if needed
        if isinstance(commit, dict):
            commit = Commit.from_dict(commit)

        # Apply to store
        result = apply_commit(self.store, commit, batch_prefix)

        # Track commit event
        status = "ok" if result.success else "error"
        created_count = len(set(result.created_ids.values()))
        event = CommitEvent(
            op="apply_commit",
            commit_id=result.commit_id,
            creates_count=created_count,
            links_count=result.links_created,
            proposals_count=result.proposals_stored,
            status=status,
            errors=result.errors,
            ts=time.time(),
        )
        self._pending_commit_events.append(event)

        return result

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Query the LM via socket connection to the handler.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
        """
        # Enforce subcall budget
        self._subcall_count += 1
        if self._subcall_count > self.subcall_budget:
            return f"Error: Subcall budget exceeded ({self.subcall_budget})"

        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        try:
            if len(prompt) > self.sub_llm_max_chars:
                original_len = len(prompt)
                prompt = (
                    f"[TRUNCATED_INPUT original_chars={original_len} max_chars={self.sub_llm_max_chars}]\n"
                    + prompt[: self.sub_llm_max_chars]
                )
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            # Track this LLM call
            self._pending_llm_calls.append(
                response.chat_completion,
            )

            return response.chat_completion.response
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Query the LM with multiple prompts concurrently.

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use (if handler has multiple clients).

        Returns:
            List of responses in the same order as input prompts.
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)

        try:
            truncated_prompts: list[str] = []
            for prompt in prompts:
                if len(prompt) > self.sub_llm_max_chars:
                    original_len = len(prompt)
                    prompt = (
                        f"[TRUNCATED_INPUT original_chars={original_len} max_chars={self.sub_llm_max_chars}]\n"
                        + prompt[: self.sub_llm_max_chars]
                    )
                truncated_prompts.append(prompt)
            responses = send_lm_request_batched(
                self.lm_handler_address, truncated_prompts, model=model, depth=self.depth
            )

            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    # Track this LLM call in list of all calls -- we may want to do this hierarchically
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()

        # Clear pending calls from previous execution
        self._pending_llm_calls = []
        self._pending_commit_events = []

        # Create sinks for this iteration's store/batch logging
        store_events: list[dict] = []
        batch_calls: list[dict] = []
        self.store.set_event_sink(store_events)
        self.store.set_batch_sink(batch_calls)

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            store_events=store_events,
            batch_calls=batch_calls,
            commit_events=self._pending_commit_events.copy(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.globals.clear()
        self.locals.clear()

    def __del__(self):
        self.cleanup()
