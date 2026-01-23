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

from concurrent.futures import ThreadPoolExecutor

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.commit import Commit, MergeResult, apply_commit, parse_commit
from rlm.core.shared_store import SharedStore, WorkerStoreProxy
from rlm.core.store import SpanRef, Store
from rlm.core.types import CommitEvent, REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv
from rlm.utils.prompts import COMMIT_PROTOCOL_PROMPT_ADDON

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
        worker_system_prompt: str | None = None,
        worker_max_iterations: int | None = None,
        worker_environment_kwargs: dict[str, Any] | None = None,
        worker_other_backends: list[str] | None = None,
        worker_other_backend_kwargs: list[dict[str, Any]] | None = None,
        worker_commit_prompt: bool = True,
        logger: "RLMLogger | None" = None,
        store_mode: str = "shared",
        shared_store: SharedStore | None = None,
        worker_id: str | None = None,
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
        self._worker_system_prompt = worker_system_prompt
        self._worker_max_iterations = worker_max_iterations
        self._worker_environment_kwargs = worker_environment_kwargs or {}
        self._worker_other_backends = worker_other_backends
        self._worker_other_backend_kwargs = worker_other_backend_kwargs
        self._worker_commit_prompt = worker_commit_prompt

        # Logger for hierarchical logging (passed to nested workers)
        self._logger = logger

        # Track commit events during code execution
        self._pending_commit_events: list[CommitEvent] = []

        # Store mode: "shared" (default) or "none" (benchmark mode)
        self._store_mode = store_mode

        # Shared store for parallel workers
        self._shared_store = shared_store
        if store_mode == "shared":
            # Generate worker_id if not provided
            self._worker_id = worker_id or f"worker_{self.depth}_{uuid.uuid4().hex[:6]}"
        else:
            self._worker_id = worker_id

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

        # Add helper functions (always available)
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["FINAL"] = self._final  # Also allow FINAL() in code
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["parse_json"] = self.parse_json
        self.globals["list_files"] = self._list_files
        self.globals["read_file"] = self._read_file

        # Create store for hierarchical data management (only when store_mode="shared")
        if self._store_mode == "shared":
            if self._shared_store is not None:
                worker_id = self._worker_id or f"worker_{self.depth}_{uuid.uuid4().hex[:6]}"
                self._worker_id = worker_id
                self.store = WorkerStoreProxy(self._shared_store, worker_id)
            else:
                # Standalone REPL without RLM - create isolated SharedStore
                self._shared_store = SharedStore()
                worker_id = self._worker_id or f"worker_{self.depth}_{uuid.uuid4().hex[:6]}"
                self._worker_id = worker_id
                self.store = WorkerStoreProxy(self._shared_store, worker_id)
            self.store.set_llm_batch_fn(self._llm_query_batched)
            self.globals["store"] = self.store
            self.globals["SpanRef"] = SpanRef

            # Commit protocol globals (only available with store)
            self.globals["rlm_worker"] = self._rlm_worker
            self.globals["rlm_worker_batched"] = self._rlm_worker_batched
            self.globals["parse_commit"] = parse_commit
            self.globals["apply_commit"] = self._apply_commit
            self.globals["Commit"] = Commit
        else:
            # store_mode="none" - no store exposed (benchmark mode)
            self.store = None

    def _final_var(self, variable_name_or_value: str) -> str:
        """Return a final answer, either by variable name or direct value.

        Handles two cases:
        1. FINAL_VAR("result") - looks up variable named "result" in locals
        2. FINAL_VAR(result) - if result='answer', uses 'answer' directly

        Prints FINAL(value) to stdout so the parser can detect it,
        even when called inside code blocks.
        """
        arg = str(variable_name_or_value).strip().strip("\"'")

        # First, check if it's a variable name in locals
        if arg in self.locals:
            value = str(self.locals[arg])
            print(f"FINAL({value})")
            return value

        # Otherwise, treat the argument itself as the final answer
        # This handles FINAL_VAR(result) where result='1,2,5,10'
        print(f"FINAL({arg})")
        return arg

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

    def _list_files(self) -> list[str]:
        """List all context files available in the working directory.

        Returns a list of filenames (not full paths) that can be read with read_file().
        """
        files = []
        for f in os.listdir(self.temp_dir):
            # Include text and json files, exclude hidden files
            if not f.startswith(".") and (f.endswith(".txt") or f.endswith(".json")):
                files.append(f)
        return sorted(files)

    def _read_file(self, filename: str) -> str:
        """Read the contents of a context file.

        Args:
            filename: The name of the file (e.g., 'chunk_1.txt')

        Returns:
            The file contents as a string (JSON files are returned as formatted JSON string)
        """
        # Security: only allow reading from temp_dir
        filepath = os.path.join(self.temp_dir, os.path.basename(filename))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filename}")

        with open(filepath, "r") as f:
            content = f.read()

        return content

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
            worker_prompt = prompt
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
            worker_system_prompt = self._worker_system_prompt
            if self._worker_commit_prompt:
                if worker_system_prompt:
                    worker_system_prompt = worker_system_prompt + "\n" + COMMIT_PROTOCOL_PROMPT_ADDON
                else:
                    worker_system_prompt = COMMIT_PROTOCOL_PROMPT_ADDON

            # Prepare environment kwargs, passing shared_store if available
            env_kwargs = self._worker_environment_kwargs.copy()
            if self._shared_store is not None:
                env_kwargs["shared_store"] = self._shared_store
                # Generate unique worker_id for the child
                child_worker_id = f"worker_{self.depth + 1}_{uuid.uuid4().hex[:6]}"
                env_kwargs["worker_id"] = child_worker_id

            worker_rlm = RLM(
                backend=self._backend,
                backend_kwargs=self._backend_kwargs,
                environment="local",
                environment_kwargs=env_kwargs,
                depth=self.depth,  # Worker starts at current depth
                max_depth=self.max_depth,
                max_iterations=worker_config.get(
                    "max_iterations", self._worker_max_iterations or 2
                ),
                custom_system_prompt=worker_system_prompt,
                other_backends=self._worker_other_backends,
                other_backend_kwargs=self._worker_other_backend_kwargs,
                verbose=False,
                logger=child_logger,  # Pass child logger for hierarchical logging
                task_name=worker_config.get("task_name"),
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

    def _rlm_worker_batched(
        self,
        tasks: list[dict],
        max_parallel: int = 8,
    ) -> list[dict]:
        """
        Spawn multiple RLM workers in parallel, all sharing the same store.

        Args:
            tasks: List of task dicts with keys:
                - prompt: str (required) - The task prompt for the worker
                - store_cards: list[dict] (optional) - Store cards to include as context
                - worker_config: dict (optional) - Config dict with max_iterations, timeout, inherit_context
            max_parallel: Maximum number of workers to run concurrently (default 8)

        Returns:
            List of commit dicts from each worker, in the same order as tasks
        """
        if not tasks:
            return []

        def run_worker(task: dict) -> dict:
            """Run a single worker task."""
            return self._rlm_worker(
                prompt=task["prompt"],
                store_cards=task.get("store_cards"),
                worker_config=task.get("worker_config"),
            )

        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(tasks))) as executor:
            futures = [executor.submit(run_worker, task) for task in tasks]
            return [f.result() for f in futures]

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
        """Load context into the environment.

        If context_payload is a dict with string keys that look like filenames (contain '.'),
        it's treated as a files dict and loaded via load_files().
        Otherwise, it's loaded as context_0 (and 'context' alias).
        """
        # Check if this looks like a files dict (keys are filenames)
        if isinstance(context_payload, dict):
            keys = list(context_payload.keys())
            if keys and all(isinstance(k, str) and "." in k for k in keys):
                # Treat as files dict
                self.load_files(context_payload)
                return

        # Default behavior: load as single context
        self.add_context(context_payload, 0)

    def load_files(self, files: dict[str, str | dict | list]):
        """Load multiple files into the environment.

        Each key becomes a filename, each value becomes the file content.
        Files are accessible via list_files() and read_file().

        Args:
            files: Dict mapping filename -> content (string or JSON-serializable)

        Example:
            load_files({
                "chunk_1.txt": "Fact 1: ... Fact 5: ...",
                "chunk_2.txt": "Fact 6: ... Fact 10: ...",
                "metadata.json": {"total_facts": 20, "source": "benchmark"}
            })
        """
        for filename, content in files.items():
            # Sanitize filename
            safe_name = os.path.basename(filename)
            if not safe_name:
                continue

            filepath = os.path.join(self.temp_dir, safe_name)

            if isinstance(content, str):
                with open(filepath, "w") as f:
                    f.write(content)
            else:
                # JSON-serialize dicts/lists
                with open(filepath, "w") as f:
                    json.dump(content, f, indent=2)

        # Also set 'context' to be a summary of available files for convenience
        file_list = self._list_files()
        self.locals["context"] = f"Files available: {file_list}. Use list_files() and read_file(name) to access."
        self._context_count = len(files)

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
        if self.store is not None:
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
