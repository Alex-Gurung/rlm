"""
Unit tests for SharedStore and WorkerStoreProxy.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rlm.core.shared_store import SharedStore, SharedStoreObject, StoreEvent, WorkerStoreProxy
from rlm.core.store import SpanRef


class TestSharedStore:
    """Tests for the SharedStore class."""

    def test_create_returns_worker_prefixed_id(self):
        """Created objects have IDs prefixed with worker_id."""
        store = SharedStore()
        obj_id = store.create(
            worker_id="worker_A",
            type="note",
            description="Test note",
            content="Test content",
        )
        assert obj_id.startswith("worker_A_")
        assert len(obj_id) == len("worker_A_") + 6  # 6 digits

    def test_worker_attribution(self):
        """Objects are attributed to creating worker."""
        store = SharedStore()
        id1 = store.create(
            worker_id="worker_A",
            type="note",
            description="Note from A",
            content="Content A",
        )
        id2 = store.create(
            worker_id="worker_B",
            type="note",
            description="Note from B",
            content="Content B",
        )

        obj1 = store.get(id1)
        obj2 = store.get(id2)

        assert obj1.created_by == "worker_A"
        assert obj2.created_by == "worker_B"

    def test_cross_worker_visibility(self):
        """Worker B can see Worker A's objects."""
        store = SharedStore()
        store.create(
            worker_id="worker_A",
            type="evidence",
            description="Found X in document",
            content={"quote": "X is true"},
        )

        results = store.query(desc_contains="Found X")
        assert len(results) == 1
        assert results[0]["worker"] == "worker_A"
        assert "Found X" in results[0]["description"]

    def test_no_id_collisions_parallel(self):
        """Parallel workers don't collide on IDs."""
        store = SharedStore()
        ids = set()
        lock = threading.Lock()

        def worker(worker_id: str):
            for _ in range(100):
                obj_id = store.create(
                    worker_id=worker_id,
                    type="note",
                    description=f"Note from {worker_id}",
                    content="content",
                )
                with lock:
                    ids.add(obj_id)

        threads = [
            threading.Thread(target=worker, args=(f"w{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 1000 unique IDs (10 workers * 100 objects each)
        assert len(ids) == 1000

    def test_no_id_collisions_threadpool(self):
        """ThreadPoolExecutor workers don't collide on IDs."""
        store = SharedStore()
        ids = []

        def create_objects(worker_id: str) -> list[str]:
            return [
                store.create(
                    worker_id=worker_id,
                    type="note",
                    description=f"Note {i}",
                    content=f"content {i}",
                )
                for i in range(50)
            ]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(create_objects, f"worker_{i}")
                for i in range(8)
            ]
            for f in futures:
                ids.extend(f.result())

        # Should have 400 unique IDs (8 workers * 50 objects each)
        assert len(set(ids)) == 400

    def test_invalidation(self):
        """Invalidated objects excluded from queries by default."""
        store = SharedStore()
        obj_id = store.create(
            worker_id="worker_A",
            type="note",
            description="Will be invalidated",
            content="content",
        )

        # Before invalidation
        assert len(store.query()) == 1

        # Invalidate
        store.invalidate(
            worker_id="worker_A",
            target_id=obj_id,
            reason="Wrong information",
        )

        # After invalidation - excluded by default
        assert len(store.query()) == 0

        # Can still see with include_invalidated=True
        results = store.query(include_invalidated=True)
        assert len(results) == 1
        assert results[0]["invalidated"] is True

    def test_invalidation_metadata(self):
        """Invalidation stores metadata about who and why."""
        store = SharedStore()
        obj_id = store.create(
            worker_id="worker_A",
            type="note",
            description="Test",
            content="content",
        )

        store.invalidate(
            worker_id="worker_B",
            target_id=obj_id,
            reason="Superseded by better analysis",
        )

        obj = store.get(obj_id)
        assert obj.invalidated is True
        assert obj.invalidated_by == "worker_B"
        assert obj.invalidated_reason == "Superseded by better analysis"
        assert obj.invalidated_at is not None

    def test_query_by_type(self):
        """Can filter by type."""
        store = SharedStore()
        store.create(worker_id="w1", type="evidence", description="E1", content="c1")
        store.create(worker_id="w1", type="note", description="N1", content="c2")
        store.create(worker_id="w1", type="evidence", description="E2", content="c3")

        evidence = store.query(type="evidence")
        assert len(evidence) == 2
        assert all(r["type"] == "evidence" for r in evidence)

    def test_query_by_worker(self):
        """Can filter by worker."""
        store = SharedStore()
        store.create(worker_id="w1", type="note", description="N1", content="c1")
        store.create(worker_id="w2", type="note", description="N2", content="c2")
        store.create(worker_id="w1", type="note", description="N3", content="c3")

        w1_notes = store.query(worker="w1")
        assert len(w1_notes) == 2
        assert all(r["worker"] == "w1" for r in w1_notes)

    def test_query_exclude_worker(self):
        """Can exclude a specific worker."""
        store = SharedStore()
        store.create(worker_id="w1", type="note", description="N1", content="c1")
        store.create(worker_id="w2", type="note", description="N2", content="c2")
        store.create(worker_id="w3", type="note", description="N3", content="c3")

        not_w1 = store.query(exclude_worker="w1")
        assert len(not_w1) == 2
        assert all(r["worker"] != "w1" for r in not_w1)

    def test_query_by_tag(self):
        """Can filter by tag."""
        store = SharedStore()
        store.create(worker_id="w1", type="note", description="N1", content="c1", tags=["important"])
        store.create(worker_id="w1", type="note", description="N2", content="c2", tags=["draft"])
        store.create(worker_id="w1", type="note", description="N3", content="c3", tags=["important", "urgent"])

        important = store.query(tag="important")
        assert len(important) == 2
        assert all("important" in r["tags"] for r in important)

    def test_query_by_description(self):
        """Can filter by description contains."""
        store = SharedStore()
        store.create(worker_id="w1", type="note", description="Found evidence of X", content="c1")
        store.create(worker_id="w1", type="note", description="No relevant data", content="c2")
        store.create(worker_id="w1", type="note", description="More evidence found", content="c3")

        evidence_notes = store.query(desc_contains="evidence")
        assert len(evidence_notes) == 2

    def test_query_by_parent(self):
        """Can filter by parent."""
        store = SharedStore()
        parent_id = store.create(worker_id="w1", type="batch", description="Batch", content={})
        store.create(worker_id="w1", type="result", description="R1", content="c1", parents=[parent_id])
        store.create(worker_id="w1", type="result", description="R2", content="c2", parents=[parent_id])
        store.create(worker_id="w1", type="note", description="Orphan", content="c3")

        children = store.query(parent=parent_id)
        assert len(children) == 2
        assert all("R" in r["description"] for r in children)

    def test_view_query_syntax(self):
        """view() supports query string syntax."""
        store = SharedStore()
        store.create(worker_id="w1", type="evidence", description="E1", content="c1", tags=["important"])
        store.create(worker_id="w2", type="note", description="N1", content="c2")

        # Query by type
        results = store.view("type=evidence")
        assert len(results) == 1
        assert results[0]["type"] == "evidence"

        # Query by tag
        results = store.view("tag=important")
        assert len(results) == 1

        # Query by description
        results = store.view('desc~"E1"')
        assert len(results) == 1

        # Query by worker
        results = store.view("worker=w2")
        assert len(results) == 1
        assert results[0]["worker"] == "w2"

    def test_view_invalid_query_key(self):
        """view() raises on unknown query key."""
        store = SharedStore()
        with pytest.raises(ValueError, match="Unknown query key"):
            store.view("unknown=value")

    def test_parent_child_relationships(self):
        """Parent-child relationships are maintained."""
        store = SharedStore()
        parent_id = store.create(worker_id="w1", type="batch", description="Parent", content={})
        child_id = store.create(worker_id="w1", type="result", description="Child", content={}, parents=[parent_id])

        parent = store.get(parent_id)
        child = store.get(child_id)

        assert child_id in parent.children
        assert parent_id in child.parents

        # Test children() and parents() methods
        assert len(store.children(parent_id)) == 1
        assert store.children(parent_id)[0].id == child_id

        assert len(store.parents(child_id)) == 1
        assert store.parents(child_id)[0].id == parent_id

    def test_invalid_parent_raises(self):
        """Creating with non-existent parent raises."""
        store = SharedStore()
        with pytest.raises(ValueError, match="Parent ID 'nonexistent' not found"):
            store.create(
                worker_id="w1",
                type="note",
                description="N",
                content="c",
                parents=["nonexistent"],
            )

    def test_events_since(self):
        """events_since returns events after given sequence."""
        store = SharedStore()
        store.create(worker_id="w1", type="note", description="N1", content="c1")
        store.create(worker_id="w1", type="note", description="N2", content="c2")

        # Get events since seq 0 (should return event at seq 1)
        events = store.events_since(0)
        assert len(events) == 1
        assert events[0]["seq"] == 1

        # Get events since seq -1 (should return both)
        events = store.events_since(-1)
        assert len(events) == 2

    def test_backrefs_normalized(self):
        """Backrefs are normalized to dicts."""
        store = SharedStore()
        span_ref = SpanRef(source_id="doc_1", start=0, end=100)
        obj_id = store.create(
            worker_id="w1",
            type="evidence",
            description="E1",
            content="c1",
            backrefs=[span_ref],
        )

        obj = store.get(obj_id)
        assert len(obj.backrefs) == 1
        assert obj.backrefs[0]["source_id"] == "doc_1"
        assert obj.backrefs[0]["start"] == 0
        assert obj.backrefs[0]["end"] == 100

    def test_non_json_serializable_raises(self):
        """Content must be JSON-serializable."""
        store = SharedStore()
        with pytest.raises(ValueError, match="JSON-serializable"):
            store.create(
                worker_id="w1",
                type="note",
                description="N",
                content=lambda x: x,  # Functions are not JSON-serializable
            )

    def test_thread_safety_stress(self):
        """Stress test for thread safety."""
        store = SharedStore()
        errors = []

        def worker(worker_id: str):
            try:
                for i in range(50):
                    # Create
                    obj_id = store.create(
                        worker_id=worker_id,
                        type="note",
                        description=f"Note {i}",
                        content=f"content {i}",
                    )
                    # Query
                    store.query(worker=worker_id)
                    store.query(type="note")
                    # Get
                    store.get(obj_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"w{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during stress test: {errors}"


class TestWorkerStoreProxy:
    """Tests for the WorkerStoreProxy class."""

    def test_create_attributed_to_worker(self):
        """Creates through proxy are attributed to the proxy's worker."""
        shared = SharedStore()
        proxy = WorkerStoreProxy(shared, "my_worker")

        obj_id = proxy.create(type="note", description="N1", content="c1")

        obj = shared.get(obj_id)
        assert obj.created_by == "my_worker"

    def test_view_sees_all_workers(self):
        """view() sees objects from all workers."""
        shared = SharedStore()
        proxy_a = WorkerStoreProxy(shared, "worker_A")
        proxy_b = WorkerStoreProxy(shared, "worker_B")

        proxy_a.create(type="note", description="From A", content="c1")
        proxy_b.create(type="note", description="From B", content="c2")

        # Both proxies see both objects
        assert len(proxy_a.view()) == 2
        assert len(proxy_b.view()) == 2

    def test_view_others_excludes_own(self):
        """view_others() excludes the proxy's own worker's objects."""
        shared = SharedStore()
        proxy_a = WorkerStoreProxy(shared, "worker_A")
        proxy_b = WorkerStoreProxy(shared, "worker_B")

        proxy_a.create(type="note", description="From A", content="c1")
        proxy_b.create(type="note", description="From B", content="c2")

        # view_others excludes own objects
        others_a = proxy_a.view_others()
        others_b = proxy_b.view_others()

        assert len(others_a) == 1
        assert others_a[0]["worker"] == "worker_B"

        assert len(others_b) == 1
        assert others_b[0]["worker"] == "worker_A"

    def test_view_others_with_query(self):
        """view_others() supports query filters."""
        shared = SharedStore()
        proxy_a = WorkerStoreProxy(shared, "worker_A")
        proxy_b = WorkerStoreProxy(shared, "worker_B")

        proxy_a.create(type="evidence", description="E1", content="c1")
        proxy_b.create(type="evidence", description="E2", content="c2")
        proxy_b.create(type="note", description="N1", content="c3")

        # view_others with type filter
        evidence_from_others = proxy_a.view_others("type=evidence")
        assert len(evidence_from_others) == 1
        assert evidence_from_others[0]["type"] == "evidence"
        assert evidence_from_others[0]["worker"] == "worker_B"

    def test_invalidate(self):
        """Invalidation through proxy is attributed."""
        shared = SharedStore()
        proxy_a = WorkerStoreProxy(shared, "worker_A")
        proxy_b = WorkerStoreProxy(shared, "worker_B")

        obj_id = proxy_a.create(type="note", description="N1", content="c1")
        proxy_b.invalidate(obj_id, reason="Superseded")

        obj = shared.get(obj_id)
        assert obj.invalidated is True
        assert obj.invalidated_by == "worker_B"

    def test_get_children_parents(self):
        """get(), children(), parents() work through proxy."""
        shared = SharedStore()
        proxy = WorkerStoreProxy(shared, "worker_A")

        parent_id = proxy.create(type="batch", description="Parent", content={})
        child_id = proxy.create(type="result", description="Child", content={}, parents=[parent_id])

        # get
        assert proxy.get(parent_id).id == parent_id

        # children
        children = proxy.children(parent_id)
        assert len(children) == 1
        assert children[0].id == child_id

        # parents
        parents = proxy.parents(child_id)
        assert len(parents) == 1
        assert parents[0].id == parent_id

    def test_card_view(self):
        """card_view works through proxy."""
        shared = SharedStore()
        proxy = WorkerStoreProxy(shared, "worker_A")

        proxy.create(type="note", description="N1", content="c1")
        proxy.create(type="evidence", description="E1", content="c2")

        cards = proxy.card_view("type=note")
        assert len(cards) == 1
        assert cards[0]["type"] == "note"

    def test_worker_id_property(self):
        """worker_id property returns the worker ID."""
        shared = SharedStore()
        proxy = WorkerStoreProxy(shared, "my_worker_id")

        assert proxy.worker_id == "my_worker_id"


class TestSharedStoreLlmMap:
    """Tests for llm_map functionality."""

    def test_llm_map_creates_batch_and_results(self):
        """llm_map creates batch node and result children."""
        shared = SharedStore()

        # Mock LLM batch function
        def mock_llm_batch(prompts, model):
            return [f"Response to: {p}" for p in prompts]

        shared.set_llm_batch_fn(mock_llm_batch)

        batch_id = shared.llm_map(
            worker_id="worker_A",
            tasks=[
                {"name": "task1", "prompt": "What is X?"},
                {"name": "task2", "prompt": "What is Y?"},
            ],
        )

        # Check batch node exists
        batch = shared.get(batch_id)
        assert batch.type == "batch_node"
        assert batch.created_by == "worker_A"

        # Check children
        children = shared.children(batch_id)
        assert len(children) == 2
        assert all(c.type == "result" for c in children)
        assert "Response to: What is X?" in [c.content for c in children]

    def test_llm_map_through_proxy(self):
        """llm_map works through WorkerStoreProxy."""
        shared = SharedStore()
        proxy = WorkerStoreProxy(shared, "worker_A")

        def mock_llm_batch(prompts, model):
            return ["answer"] * len(prompts)

        proxy.set_llm_batch_fn(mock_llm_batch)

        batch_id = proxy.llm_map(
            tasks=[{"name": "t1", "prompt": "p1"}],
        )

        batch = shared.get(batch_id)
        assert batch.created_by == "worker_A"

    def test_llm_map_without_batch_fn_raises(self):
        """llm_map raises if batch function not set."""
        shared = SharedStore()

        with pytest.raises(RuntimeError, match="llm_batch_fn not set"):
            shared.llm_map(
                worker_id="worker_A",
                tasks=[{"name": "t1", "prompt": "p1"}],
            )


class TestSharedStoreSearchSummary:
    """Tests for search() and summary()."""

    def test_search_alias(self):
        shared = SharedStore()
        shared.create(worker_id="w1", type="note", description="Alpha note", content="c1")
        shared.create(worker_id="w1", type="note", description="Beta note", content="c2")

        results = shared.search('desc~"Alpha"')
        assert len(results) == 1
        assert results[0]["description"] == "Alpha note"

    def test_summary_returns_matches_and_facets(self):
        shared = SharedStore()
        shared.create(worker_id="w1", type="note", description="Alpha note", content="c1", tags=["t1"])
        shared.create(worker_id="w2", type="evidence", description="Alpha evidence", content="c2", tags=["t2"])

        summary = shared.summary('desc~"Alpha"')
        assert "matches" in summary
        assert "types" in summary
        assert "tags" in summary
        assert len(summary["matches"]) == 2
