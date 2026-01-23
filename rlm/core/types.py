import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

ClientBackend = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "litellm",
    "anthropic",
    "azure_openai",
    "gemini",
]
EnvironmentType = Literal["local", "docker", "modal", "prime"]


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


########################################################
########    Types for LM Cost Tracking         #########
########################################################


@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int

    def to_dict(self):
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsageSummary":
        return cls(
            total_calls=data.get("total_calls"),
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    def to_dict(self):
        return {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UsageSummary":
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in data.get("model_usage_summaries", {}).items()
            },
        )


########################################################
########   Types for REPL and RLM Iterations   #########
########################################################


@dataclass
class CommitEvent:
    """Record of a commit merge operation."""

    op: str
    commit_id: str
    creates_count: int
    links_count: int
    proposals_count: int
    status: str  # "ok" | "error"
    errors: list[str]
    ts: float

    def __init__(
        self,
        commit_id: str,
        op: str = "apply_commit",
        creates_count: int = 0,
        links_count: int = 0,
        proposals_count: int = 0,
        status: str = "ok",
        errors: list[str] | None = None,
        ts: float | None = None,
    ):
        self.commit_id = commit_id
        self.op = op
        self.creates_count = creates_count
        self.links_count = links_count
        self.proposals_count = proposals_count
        self.status = status
        self.errors = errors or []
        self.ts = time.time() if ts is None else ts

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "commit_id": self.commit_id,
            "creates_count": self.creates_count,
            "links_count": self.links_count,
            "proposals_count": self.proposals_count,
            "status": self.status,
            "errors": self.errors,
            "ts": self.ts,
        }


@dataclass
class RLMChatCompletion:
    """Record of a single LLM call made from within the environment."""

    root_model: str
    prompt: str | dict[str, Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float

    def to_dict(self):
        return {
            "root_model": self.root_model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RLMChatCompletion":
        return cls(
            root_model=data.get("root_model"),
            prompt=data.get("prompt"),
            response=data.get("response"),
            usage_summary=UsageSummary.from_dict(data.get("usage_summary")),
            execution_time=data.get("execution_time"),
        )


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    llm_calls: list["RLMChatCompletion"]
    store_events: list[dict]
    batch_calls: list[dict]
    commit_events: list["CommitEvent"]

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict,
        execution_time: float = None,
        rlm_calls: list["RLMChatCompletion"] = None,
        store_events: list[dict] = None,
        batch_calls: list[dict] = None,
        commit_events: list["CommitEvent"] = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.rlm_calls = rlm_calls or []
        self.store_events = store_events or []
        self.batch_calls = batch_calls or []
        self.commit_events = commit_events or []

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time}, rlm_calls={len(self.rlm_calls)}, store_events={len(self.store_events)}, batch_calls={len(self.batch_calls)}, commit_events={len(self.commit_events)})"

    def to_dict(self):
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: _serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "rlm_calls": [call.to_dict() for call in self.rlm_calls],
            "store_events": self.store_events,
            "batch_calls": self.batch_calls,
            "commit_events": [event.to_dict() for event in self.commit_events],
        }


@dataclass
class CodeBlock:
    code: str
    result: REPLResult

    def to_dict(self):
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final_answer: str | None = None
    iteration_time: float | None = None

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [code_block.to_dict() for code_block in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }


########################################################
########   Types for RLM Metadata   #########
########################################################


@dataclass
class RLMMetadata:
    """Metadata about the RLM configuration."""

    root_model: str
    task_name: str | None
    store_mode: str | None
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any]
    environment_type: str
    environment_kwargs: dict[str, Any]
    other_backends: list[str] | None = None

    def to_dict(self):
        return {
            "root_model": self.root_model,
            "task_name": self.task_name,
            "store_mode": self.store_mode,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: _serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: _serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }


########################################################
########   Types for RLM Prompting   #########
########################################################


@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                    continue
                try:
                    import json

                    self.context_lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    self.context_lengths.append(len(repr(chunk)))
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if len(prompt) == 0:
                self.context_lengths = [0]
            elif isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(str(chunk.get("content", ""))) for chunk in prompt]
                else:
                    self.context_lengths = []
                    for chunk in prompt:
                        try:
                            import json

                            self.context_lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            self.context_lengths.append(len(repr(chunk)))
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)
