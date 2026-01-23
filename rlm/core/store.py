"""
Store module for hierarchical understanding, provenance/backrefs, and hyper-parallel map-style subcalls.
"""

from __future__ import annotations

import json
import shlex
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from rlm.core.commit import Commit, MergeResult


@dataclass
class SpanRef:
    """Reference to a span in a source document."""

    source_id: str  # "context_0" for context, or file path
    start: int
    end: int
    unit: str = "chars"  # or "lines"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "start": self.start,
            "end": self.end,
            "unit": self.unit,
        }


@dataclass
class StoreObject:
    """A node in the store hierarchy."""

    id: str
    type: str  # "note"|"claim"|"summary"|"batch_node"|"result"
    description: str  # <200 chars for navigation
    content: str | dict | list  # JSON-serializable
    backrefs: list[dict[str, Any]]  # Stored as dicts (SpanRef.to_dict())
    parents: list[str]
    children: list[str]
    tags: list[str]
    created_at: float


class Store:
    """
    In-memory store for hierarchical structured data with provenance tracking.

    Provides:
    - Hierarchical parent/child relationships
    - Backref tracking to source spans
    - Batched LLM queries via llm_map()
    - Query-based navigation
    """

    def __init__(self):
        self._objects: dict[str, StoreObject] = {}
        self._event_sink: list[dict] = []
        self._batch_sink: list[dict] = []
        self._llm_batch_fn: Callable[[list[str], str | None], list[str]] | None = None

    def set_event_sink(self, sink: list[dict]) -> None:
        """Swap event sink per iteration (for logging)."""
        self._event_sink = sink

    def set_batch_sink(self, sink: list[dict]) -> None:
        """Swap batch sink per iteration (for logging)."""
        self._batch_sink = sink

    def set_llm_batch_fn(self, fn: Callable[[list[str], str | None], list[str]]) -> None:
        """Dependency injection for llm_query_batched."""
        self._llm_batch_fn = fn

    def create(
        self,
        type: str,
        description: str,
        content: Any,
        backrefs: list[SpanRef | dict[str, Any]] | None = None,
        parents: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Create a new object in the store.

        Args:
            type: Object type ("note"|"claim"|"summary"|"batch_node"|"result")
            description: Short description (<200 chars) for navigation
            content: JSON-serializable content
            backrefs: List of SpanRef objects or dicts
            parents: List of parent object IDs
            tags: List of tags for filtering

        Returns:
            The ID of the created object

        Raises:
            ValueError: If content is not JSON-serializable or parents don't exist
        """
        self._validate_json_serializable(content)

        # Validate parents exist
        parents = parents or []
        for parent_id in parents:
            if parent_id not in self._objects:
                raise ValueError(f"Parent ID '{parent_id}' not found in store")

        # Generate short UUID
        obj_id = uuid.uuid4().hex[:8]

        # Normalize backrefs to dicts
        backrefs = backrefs or []
        normalized_backrefs = []
        for ref in backrefs:
            normalized_backrefs.append(self._normalize_backref(ref))

        # Create object
        created_at = time.time()
        obj = StoreObject(
            id=obj_id,
            type=type,
            description=description,
            content=content,
            backrefs=normalized_backrefs,
            parents=parents,
            children=[],
            tags=tags or [],
            created_at=created_at,
        )

        # Store object
        self._objects[obj_id] = obj

        # Update parent.children
        for parent_id in parents:
            self._objects[parent_id].children.append(obj_id)

        # Append event to sink
        self._event_sink.append(
            {
                "op": "create",
                "id": obj_id,
                "type": type,
                "description": description,
                "parents": parents,
                "tags": tags or [],
                "backrefs_count": len(normalized_backrefs),
                "ts": created_at,
            }
        )

        return obj_id

    def get(self, id: str) -> StoreObject | None:
        """Get an object by ID."""
        return self._objects.get(id)

    def children(self, id: str) -> list[StoreObject]:
        """Get all child objects of an object."""
        if id not in self._objects:
            raise ValueError(f"Object ID '{id}' not found in store")
        obj = self._objects[id]
        return [self._objects[child_id] for child_id in obj.children if child_id in self._objects]

    def parents(self, id: str) -> list[StoreObject]:
        """Get all parent objects of an object."""
        if id not in self._objects:
            raise ValueError(f"Object ID '{id}' not found in store")
        obj = self._objects[id]
        return [self._objects[parent_id] for parent_id in obj.parents if parent_id in self._objects]

    def view(self, query: str = "", limit: int = 50) -> list[dict]:
        """
        Return metadata for objects matching the query.

        Returns:
            List of dicts: [{id, type, description, tags}]

        Query syntax (fail-fast on unknown keys):
            type=note        filter by type
            tag=important    filter by tag
            parent=abc123    filter by parent
            desc~"keyword"   description contains
        """
        results = list(self._objects.values())

        if query:
            parts = shlex.split(query)
            for part in parts:
                if "=" in part and "~" not in part:
                    key, value = part.split("=", 1)
                    if key == "type":
                        results = [obj for obj in results if obj.type == value]
                    elif key == "tag":
                        results = [obj for obj in results if value in obj.tags]
                    elif key == "parent":
                        results = [obj for obj in results if value in obj.parents]
                    else:
                        raise ValueError(f"Unknown query key: {key}")
                elif "~" in part:
                    key, value = part.split("~", 1)
                    if key == "desc":
                        value = value.strip('"\'')
                        if value:
                            results = [
                                obj
                                for obj in results
                                if value.lower() in obj.description.lower()
                            ]
                    else:
                        raise ValueError(f"Unknown query key: {key}")
                else:
                    raise ValueError(f"Invalid query syntax: {part}")

        # Sort by creation time (newest first) and limit
        results.sort(key=lambda x: x.created_at, reverse=True)
        results = results[:limit]

        # Return metadata only
        return [
            {
                "id": obj.id,
                "type": obj.type,
                "description": obj.description,
                "tags": obj.tags,
            }
            for obj in results
        ]

    def search(self, query: str = "", limit: int = 50) -> list[dict]:
        """
        Alias for view() to make store discovery more obvious.

        Args:
            query: Query string (same syntax as view())
            limit: Maximum number of results to return
        """
        return self.view(query, limit)

    def summary(self, query: str = "", limit: int = 10) -> dict[str, Any]:
        """
        Provide a lightweight search summary payload.

        Returns:
            Dict with:
              - matches: list of {id, type, description, tags}
              - types: list of top types by frequency
              - tags: list of top tags by frequency
        """
        matches = self.view(query, limit)
        type_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        for obj in self._objects.values():
            type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
            for tag in obj.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "matches": matches,
            "types": [t for t, _ in top_types],
            "tags": [t for t, _ in top_tags],
        }

    def llm_map(
        self,
        tasks: list[dict],
        parent: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Run batched LLM queries and store results as child nodes.

        Args:
            tasks: List of task dicts with keys:
                - name: str (required)
                - description: str (optional, defaults to name)
                - prompt: str (required)
                - backrefs: list[SpanRef|dict] (optional)
                - tags: list[str] (optional)
            parent: Optional parent object ID
            model: Optional model name to use

        Returns:
            The ID of the batch_node containing the results
        """
        if self._llm_batch_fn is None:
            raise RuntimeError("llm_batch_fn not set. Call set_llm_batch_fn() first.")

        prompts = [t["prompt"] for t in tasks]

        start = time.time()
        responses = self._llm_batch_fn(prompts, model)
        elapsed = time.time() - start

        # Log batch call
        self._batch_sink.append(
            {
                "prompts_count": len(prompts),
                "model": model,
                "execution_time": elapsed,
                "ts": start,
            }
        )

        # Create batch node
        batch_id = self.create(
            type="batch_node",
            description=f"Batch: {len(tasks)} tasks",
            content={"task_names": [t["name"] for t in tasks]},
            parents=[parent] if parent else [],
        )

        # Create child result nodes
        for task, response in zip(tasks, responses):
            self.create(
                type="result",
                description=task.get("description", task["name"]),
                content=response,
                backrefs=task.get("backrefs", []),
                parents=[batch_id],
                tags=task.get("tags", []),
            )

        return batch_id

    def card_view(self, query: str = "", limit: int = 20) -> list[dict]:
        """
        Return lightweight cards for objects matching the query.

        Cards are suitable for passing to sub-LLM workers as context,
        containing only essential metadata without full content.

        Args:
            query: Query string (same syntax as view())
            limit: Maximum number of cards to return

        Returns:
            List of dicts: [{id, type, description, tags}]
        """
        return self.view(query, limit)

    def apply_commit(self, commit: "Commit", batch_prefix: str = "") -> "MergeResult":
        """
        Merge a worker commit into the store.

        This is a convenience wrapper around commit.apply_commit().

        Args:
            commit: The Commit object to merge
            batch_prefix: Optional prefix for namespacing created objects

        Returns:
            MergeResult with created IDs and any errors
        """
        from rlm.core.commit import apply_commit

        return apply_commit(self, commit, batch_prefix)

    @staticmethod
    def _validate_json_serializable(content: Any) -> None:
        try:
            json.dumps(content)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Content must be JSON-serializable: {e}") from e

    @staticmethod
    def _normalize_backref(ref: SpanRef | dict[str, Any]) -> dict[str, Any]:
        if isinstance(ref, SpanRef):
            return ref.to_dict()
        if isinstance(ref, dict):
            if "source_id" not in ref or "start" not in ref or "end" not in ref:
                raise ValueError("Backref dict must include source_id, start, and end")
            return {
                "source_id": ref["source_id"],
                "start": ref["start"],
                "end": ref["end"],
                "unit": ref.get("unit", "chars"),
            }
        raise ValueError(f"Invalid backref type: {type(ref)}")
