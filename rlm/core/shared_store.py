"""
Shared append-only store for parallel RLM workers.

This module provides SharedStore and WorkerStoreProxy classes that allow
parallel workers to see each other's findings in real-time through a
shared append-only event log.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from rlm.core.store import SpanRef


@dataclass
class StoreEvent:
    """A single event in the append-only log."""

    seq: int  # Global sequence number
    worker_id: str  # Attribution
    op: Literal["create", "invalidate", "link"]
    payload: dict
    timestamp: float


@dataclass
class SharedStoreObject:
    """A materialized object in the shared store."""

    id: str
    type: str
    description: str
    content: str | dict | list
    backrefs: list[dict[str, Any]]
    parents: list[str]
    children: list[str]
    tags: list[str]
    created_at: float
    created_by: str  # Worker attribution
    invalidated: bool = False
    invalidated_by: str | None = None
    invalidated_reason: str | None = None
    invalidated_at: float | None = None


class SharedStore:
    """
    Thread-safe append-only store for parallel RLM workers.

    All writes append to an event log. A materialized view is updated
    on each append. Workers can query the shared state to see each
    other's findings in real-time.

    Key features:
    - IDs are `{worker_id}_{seq:06d}` - no collisions between workers
    - All writes are append-only (creates and invalidations)
    - Thread-safe with RLock for concurrent access
    - Supports querying by type, worker, description, and tags
    """

    def __init__(self):
        self._events: list[StoreEvent] = []
        self._objects: dict[str, SharedStoreObject] = {}
        self._lock = threading.RLock()
        self._seq_counter: int = 0
        self._worker_seq: dict[str, int] = {}  # Per-worker sequence for IDs
        self._llm_batch_fn: Callable[[list[str], str | None], list[str]] | None = None
        self._tls = threading.local()

    def set_event_sink(self, sink: list[dict]) -> None:
        """Swap event sink per thread/iteration (for logging)."""
        self._tls.event_sink = sink

    def set_batch_sink(self, sink: list[dict]) -> None:
        """Swap batch sink per thread/iteration (for logging)."""
        self._tls.batch_sink = sink

    def _get_event_sink(self) -> list[dict] | None:
        return getattr(self._tls, "event_sink", None)

    def _get_batch_sink(self) -> list[dict] | None:
        return getattr(self._tls, "batch_sink", None)

    def set_llm_batch_fn(self, fn: Callable[[list[str], str | None], list[str]]) -> None:
        """Dependency injection for llm_query_batched."""
        self._llm_batch_fn = fn

    def _next_seq(self) -> int:
        """Get next global sequence number (must hold lock)."""
        seq = self._seq_counter
        self._seq_counter += 1
        return seq

    def _worker_next_seq(self, worker_id: str) -> int:
        """Get next per-worker sequence number (must hold lock)."""
        if worker_id not in self._worker_seq:
            self._worker_seq[worker_id] = 0
        seq = self._worker_seq[worker_id]
        self._worker_seq[worker_id] += 1
        return seq

    def _generate_id(self, worker_id: str) -> str:
        """Generate a unique ID for an object (must hold lock)."""
        worker_seq = self._worker_next_seq(worker_id)
        return f"{worker_id}_{worker_seq:06d}"

    def create(
        self,
        worker_id: str,
        type: str,
        description: str,
        content: Any,
        backrefs: list[Any] | None = None,
        parents: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Append a create event and return the worker-prefixed ID.

        Args:
            worker_id: The worker creating this object
            type: Object type (note, claim, summary, evidence, etc.)
            description: Short description (<200 chars) for navigation
            content: JSON-serializable content
            backrefs: List of SpanRef objects or dicts
            parents: List of parent object IDs
            tags: List of tags for filtering

        Returns:
            The ID of the created object (format: worker_id_NNNNNN)

        Raises:
            ValueError: If content is not JSON-serializable or parents don't exist
        """
        import json

        # Validate JSON-serializable
        try:
            json.dumps(content)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Content must be JSON-serializable: {e}") from e

        with self._lock:
            # Validate parents exist
            parents = parents or []
            for parent_id in parents:
                if parent_id not in self._objects:
                    raise ValueError(f"Parent ID '{parent_id}' not found in store")

            # Generate ID and sequence
            obj_id = self._generate_id(worker_id)
            seq = self._next_seq()
            created_at = time.time()

            # Normalize backrefs
            normalized_backrefs = []
            for ref in backrefs or []:
                normalized_backrefs.append(self._normalize_backref(ref))

            # Create event
            event = StoreEvent(
                seq=seq,
                worker_id=worker_id,
                op="create",
                payload={
                    "id": obj_id,
                    "type": type,
                    "description": description,
                    "content": content,
                    "backrefs": normalized_backrefs,
                    "parents": parents,
                    "tags": tags or [],
                },
                timestamp=created_at,
            )
            self._events.append(event)

            # Create materialized object
            obj = SharedStoreObject(
                id=obj_id,
                type=type,
                description=description,
                content=content,
                backrefs=normalized_backrefs,
                parents=parents,
                children=[],
                tags=tags or [],
                created_at=created_at,
                created_by=worker_id,
            )
            self._objects[obj_id] = obj

            # Update parent.children
            for parent_id in parents:
                self._objects[parent_id].children.append(obj_id)

            # Log to thread-local event sink if set
            event_sink = self._get_event_sink()
            if event_sink is not None:
                event_sink.append({
                    "op": "create",
                    "id": obj_id,
                    "type": type,
                    "description": description,
                    "parents": parents,
                    "tags": tags or [],
                    "backrefs_count": len(normalized_backrefs),
                    "worker_id": worker_id,
                    "ts": created_at,
                })

            return obj_id

    def get(self, id: str) -> SharedStoreObject | None:
        """Get an object by ID."""
        with self._lock:
            return self._objects.get(id)

    def invalidate(
        self,
        worker_id: str,
        target_id: str,
        reason: str,
    ) -> None:
        """
        Soft-delete an object via an invalidation event.

        The object remains in the store but is excluded from queries
        by default. Use include_invalidated=True in query() to see them.

        Args:
            worker_id: The worker performing the invalidation
            target_id: The ID of the object to invalidate
            reason: Reason for invalidation
        """
        with self._lock:
            if target_id not in self._objects:
                raise ValueError(f"Object ID '{target_id}' not found in store")

            obj = self._objects[target_id]
            if obj.invalidated:
                return  # Already invalidated

            seq = self._next_seq()
            invalidated_at = time.time()

            # Create invalidation event
            event = StoreEvent(
                seq=seq,
                worker_id=worker_id,
                op="invalidate",
                payload={
                    "target_id": target_id,
                    "reason": reason,
                },
                timestamp=invalidated_at,
            )
            self._events.append(event)

            # Update materialized object
            obj.invalidated = True
            obj.invalidated_by = worker_id
            obj.invalidated_reason = reason
            obj.invalidated_at = invalidated_at

            # Log to thread-local event sink if set
            event_sink = self._get_event_sink()
            if event_sink is not None:
                event_sink.append({
                    "op": "invalidate",
                    "target_id": target_id,
                    "reason": reason,
                    "worker_id": worker_id,
                    "ts": invalidated_at,
                })

    def query(
        self,
        type: str | None = None,
        worker: str | None = None,
        exclude_worker: str | None = None,
        desc_contains: str | None = None,
        tag: str | None = None,
        parent: str | None = None,
        include_invalidated: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """
        Query materialized state.

        Args:
            type: Filter by object type
            worker: Filter by creating worker
            exclude_worker: Exclude objects from this worker
            desc_contains: Filter by description containing string
            tag: Filter by tag
            parent: Filter by parent ID
            include_invalidated: Include invalidated objects
            limit: Maximum results to return

        Returns:
            List of dicts with object metadata
        """
        with self._lock:
            results = list(self._objects.values())

            # Apply filters
            if not include_invalidated:
                results = [obj for obj in results if not obj.invalidated]

            if type is not None:
                results = [obj for obj in results if obj.type == type]

            if worker is not None:
                results = [obj for obj in results if obj.created_by == worker]

            if exclude_worker is not None:
                results = [obj for obj in results if obj.created_by != exclude_worker]

            if desc_contains is not None:
                results = [
                    obj for obj in results
                    if desc_contains.lower() in obj.description.lower()
                ]

            if tag is not None:
                results = [obj for obj in results if tag in obj.tags]

            if parent is not None:
                results = [obj for obj in results if parent in obj.parents]

            # Sort by creation time (newest first) and limit
            results.sort(key=lambda x: x.created_at, reverse=True)
            results = results[:limit]

            # Return metadata
            return [
                {
                    "id": obj.id,
                    "type": obj.type,
                    "description": obj.description,
                    "tags": obj.tags,
                    "worker": obj.created_by,
                    "invalidated": obj.invalidated,
                }
                for obj in results
            ]

    def view(self, query_str: str = "", limit: int = 50) -> list[dict]:
        """
        Return metadata for objects matching the query string.

        This provides compatibility with the Store.view() interface.

        Query syntax:
            type=note        filter by type
            tag=important    filter by tag
            parent=abc123    filter by parent
            desc~"keyword"   description contains
            worker=worker_1  filter by worker

        Returns:
            List of dicts: [{id, type, description, tags, worker}]
        """
        import shlex

        kwargs: dict[str, Any] = {"limit": limit}

        if query_str:
            parts = shlex.split(query_str)
            for part in parts:
                if "=" in part and "~" not in part:
                    key, value = part.split("=", 1)
                    if key == "type":
                        kwargs["type"] = value
                    elif key == "tag":
                        kwargs["tag"] = value
                    elif key == "parent":
                        kwargs["parent"] = value
                    elif key == "worker":
                        kwargs["worker"] = value
                    else:
                        raise ValueError(f"Unknown query key: {key}")
                elif "~" in part:
                    key, value = part.split("~", 1)
                    if key == "desc":
                        value = value.strip('"\'')
                        if value:
                            kwargs["desc_contains"] = value
                    else:
                        raise ValueError(f"Unknown query key: {key}")
                else:
                    raise ValueError(f"Invalid query syntax: {part}")

        return self.query(**kwargs)

    def search(self, query_str: str = "", limit: int = 50) -> list[dict]:
        """
        Alias for view() to make store discovery more obvious.

        Args:
            query_str: Query string (same syntax as view())
            limit: Maximum number of results to return
        """
        return self.view(query_str, limit)

    def summary(self, query_str: str = "", limit: int = 10) -> dict[str, Any]:
        """
        Provide a lightweight search summary payload.

        Returns:
            Dict with:
              - matches: list of {id, type, description, tags, worker}
              - types: list of top types by frequency
              - tags: list of top tags by frequency
        """
        matches = self.view(query_str, limit)
        type_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        with self._lock:
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

    def children(self, id: str) -> list[SharedStoreObject]:
        """Get all child objects of an object."""
        with self._lock:
            if id not in self._objects:
                raise ValueError(f"Object ID '{id}' not found in store")
            obj = self._objects[id]
            return [
                self._objects[child_id]
                for child_id in obj.children
                if child_id in self._objects
            ]

    def parents(self, id: str) -> list[SharedStoreObject]:
        """Get all parent objects of an object."""
        with self._lock:
            if id not in self._objects:
                raise ValueError(f"Object ID '{id}' not found in store")
            obj = self._objects[id]
            return [
                self._objects[parent_id]
                for parent_id in obj.parents
                if parent_id in self._objects
            ]

    def card_view(self, query_str: str = "", limit: int = 20) -> list[dict]:
        """
        Return lightweight cards for objects matching the query.

        Cards are suitable for passing to sub-LLM workers as context,
        containing only essential metadata without full content.

        Args:
            query_str: Query string (same syntax as view())
            limit: Maximum number of cards to return

        Returns:
            List of dicts: [{id, type, description, tags, worker}]
        """
        return self.view(query_str, limit)

    def llm_map(
        self,
        worker_id: str,
        tasks: list[dict],
        parent: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Run batched LLM queries and store results as child nodes.

        Args:
            worker_id: The worker calling this method
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

        # Log batch call to thread-local sink
        batch_sink = self._get_batch_sink()
        if batch_sink is not None:
            batch_sink.append({
                "prompts_count": len(prompts),
                "model": model,
                "execution_time": elapsed,
                "ts": start,
            })

        # Create batch node
        batch_id = self.create(
            worker_id=worker_id,
            type="batch_node",
            description=f"Batch: {len(tasks)} tasks",
            content={"task_names": [t["name"] for t in tasks]},
            parents=[parent] if parent else [],
        )

        # Create child result nodes
        for task, response in zip(tasks, responses):
            self.create(
                worker_id=worker_id,
                type="result",
                description=task.get("description", task["name"]),
                content=response,
                backrefs=task.get("backrefs", []),
                parents=[batch_id],
                tags=task.get("tags", []),
            )

        return batch_id

    def events_since(self, seq: int) -> list[dict]:
        """
        Get all events since a given sequence number.

        This is useful for future remote sync implementations.

        Args:
            seq: The sequence number to start from (exclusive)

        Returns:
            List of event dicts with keys: seq, worker_id, op, payload, timestamp
        """
        with self._lock:
            return [
                {
                    "seq": e.seq,
                    "worker_id": e.worker_id,
                    "op": e.op,
                    "payload": e.payload,
                    "timestamp": e.timestamp,
                }
                for e in self._events
                if e.seq > seq
            ]

    @staticmethod
    def _normalize_backref(ref: Any) -> dict[str, Any]:
        """Normalize a backref to a dict."""
        # Import SpanRef here to avoid circular import
        from rlm.core.store import SpanRef

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


class WorkerStoreProxy:
    """
    Proxy that gives a worker scoped access to a shared store.

    All creates are attributed to the worker_id. The proxy provides
    a view() method that sees ALL workers' objects, and a view_others()
    method that excludes the current worker's objects.
    """

    def __init__(self, shared_store: SharedStore, worker_id: str):
        self._store = shared_store
        self._worker_id = worker_id

    @property
    def worker_id(self) -> str:
        """The worker ID for this proxy."""
        return self._worker_id

    def set_event_sink(self, sink: list[dict]) -> None:
        """Swap event sink per iteration (for logging)."""
        self._store.set_event_sink(sink)

    def set_batch_sink(self, sink: list[dict]) -> None:
        """Swap batch sink per iteration (for logging)."""
        self._store.set_batch_sink(sink)

    def set_llm_batch_fn(self, fn: Callable[[list[str], str | None], list[str]]) -> None:
        """Dependency injection for llm_query_batched."""
        self._store.set_llm_batch_fn(fn)

    def create(
        self,
        type: str,
        description: str,
        content: Any,
        backrefs: list[Any] | None = None,
        parents: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Create a new object attributed to this worker.

        Args:
            type: Object type (note, claim, summary, evidence, etc.)
            description: Short description (<200 chars) for navigation
            content: JSON-serializable content
            backrefs: List of SpanRef objects or dicts
            parents: List of parent object IDs
            tags: List of tags for filtering

        Returns:
            The ID of the created object
        """
        return self._store.create(
            worker_id=self._worker_id,
            type=type,
            description=description,
            content=content,
            backrefs=backrefs,
            parents=parents,
            tags=tags,
        )

    def get(self, id: str) -> SharedStoreObject | None:
        """Get an object by ID."""
        return self._store.get(id)

    def view(self, query: str = "", limit: int = 50) -> list[dict]:
        """
        Return metadata for objects matching the query.

        This sees ALL workers' objects, not just this worker's.

        Query syntax (same as Store.view()):
            type=note        filter by type
            tag=important    filter by tag
            parent=abc123    filter by parent
            desc~"keyword"   description contains
            worker=worker_1  filter by worker

        Returns:
            List of dicts: [{id, type, description, tags, worker}]
        """
        return self._store.view(query, limit)

    def search(self, query: str = "", limit: int = 50) -> list[dict]:
        """
        Alias for view() to make store discovery more obvious.
        """
        return self._store.search(query, limit)

    def summary(self, query: str = "", limit: int = 10) -> dict[str, Any]:
        """
        Provide a lightweight search summary payload.
        """
        return self._store.summary(query, limit)

    def view_others(self, query: str = "", limit: int = 50) -> list[dict]:
        """
        Return metadata for objects from OTHER workers only.

        This excludes the current worker's objects, showing only
        findings from other parallel workers.

        Args:
            query: Query string (same syntax as view())
            limit: Maximum results to return

        Returns:
            List of dicts: [{id, type, description, tags, worker}]
        """
        # Parse query to build kwargs, then add exclude_worker
        import shlex

        kwargs: dict[str, Any] = {"limit": limit, "exclude_worker": self._worker_id}

        if query:
            parts = shlex.split(query)
            for part in parts:
                if "=" in part and "~" not in part:
                    key, value = part.split("=", 1)
                    if key == "type":
                        kwargs["type"] = value
                    elif key == "tag":
                        kwargs["tag"] = value
                    elif key == "parent":
                        kwargs["parent"] = value
                    elif key == "worker":
                        kwargs["worker"] = value
                    else:
                        raise ValueError(f"Unknown query key: {key}")
                elif "~" in part:
                    key, value = part.split("~", 1)
                    if key == "desc":
                        value = value.strip('"\'')
                        if value:
                            kwargs["desc_contains"] = value
                    else:
                        raise ValueError(f"Unknown query key: {key}")
                else:
                    raise ValueError(f"Invalid query syntax: {part}")

        return self._store.query(**kwargs)

    def invalidate(self, target_id: str, reason: str) -> None:
        """Soft-delete an object via invalidation."""
        self._store.invalidate(
            worker_id=self._worker_id,
            target_id=target_id,
            reason=reason,
        )

    def children(self, id: str) -> list[SharedStoreObject]:
        """Get all child objects of an object."""
        return self._store.children(id)

    def parents(self, id: str) -> list[SharedStoreObject]:
        """Get all parent objects of an object."""
        return self._store.parents(id)

    def card_view(self, query: str = "", limit: int = 20) -> list[dict]:
        """Return lightweight cards for objects matching the query."""
        return self._store.card_view(query, limit)

    def llm_map(
        self,
        tasks: list[dict],
        parent: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Run batched LLM queries and store results as child nodes.

        Results are attributed to this worker.
        """
        return self._store.llm_map(
            worker_id=self._worker_id,
            tasks=tasks,
            parent=parent,
            model=model,
        )
