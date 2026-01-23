"""Integration tests for SharedStore with LocalREPL.

These tests verify that:
1. LocalREPL correctly uses WorkerStoreProxy when shared_store is provided
2. Multiple LocalREPL instances sharing a store can see each other's objects
3. Isolated stores (baseline) don't share between REPLs
"""

import pytest
from rlm.core.shared_store import SharedStore, WorkerStoreProxy
from rlm.core.store import Store
from rlm.environments.local_repl import LocalREPL


class TestLocalREPLWithSharedStore:
    """Tests for LocalREPL using SharedStore."""

    def test_uses_worker_store_proxy(self):
        """LocalREPL uses WorkerStoreProxy when shared_store provided."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, worker_id="test_worker")

        assert isinstance(repl.store, WorkerStoreProxy)
        assert repl.store.worker_id == "test_worker"
        repl.cleanup()

    def test_uses_isolated_store_by_default(self):
        """LocalREPL uses isolated Store when no shared_store provided."""
        repl = LocalREPL()

        assert isinstance(repl.store, Store)
        assert not isinstance(repl.store, WorkerStoreProxy)
        repl.cleanup()

    def test_generates_worker_id_if_not_provided(self):
        """LocalREPL generates worker_id if shared_store provided but no worker_id."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared)

        assert isinstance(repl.store, WorkerStoreProxy)
        assert repl._worker_id is not None
        assert repl._worker_id.startswith("worker_")
        repl.cleanup()

    def test_creates_objects_attributed_to_worker(self):
        """Objects created through REPL are attributed to worker."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, worker_id="worker_A")

        repl.execute_code('obj_id = store.create(type="note", description="Test", content="...")')

        # Get the created object ID
        obj_id = repl.locals.get("obj_id")
        obj = shared.get(obj_id)

        assert obj is not None
        assert obj.created_by == "worker_A"
        repl.cleanup()

    def test_multiple_repls_share_store(self):
        """Multiple REPLs with same shared_store see each other's objects."""
        shared = SharedStore()
        repl_a = LocalREPL(shared_store=shared, worker_id="worker_A")
        repl_b = LocalREPL(shared_store=shared, worker_id="worker_B")

        # Worker A creates an object
        repl_a.execute_code('store.create(type="evidence", description="Found X", content="...")')

        # Worker B can see it
        repl_b.execute_code('results = store.view("type=evidence")')
        results = repl_b.locals.get("results")

        assert len(results) == 1
        assert results[0]["worker"] == "worker_A"
        assert "Found X" in results[0]["description"]

        repl_a.cleanup()
        repl_b.cleanup()

    def test_view_others_excludes_own(self):
        """store.view_others() in REPL excludes own objects."""
        shared = SharedStore()
        repl_a = LocalREPL(shared_store=shared, worker_id="worker_A")
        repl_b = LocalREPL(shared_store=shared, worker_id="worker_B")

        # Both workers create objects
        repl_a.execute_code('store.create(type="note", description="From A", content="...")')
        repl_b.execute_code('store.create(type="note", description="From B", content="...")')

        # Worker A sees only B's objects with view_others
        repl_a.execute_code('others = store.view_others()')
        others = repl_a.locals.get("others")

        assert len(others) == 1
        assert others[0]["worker"] == "worker_B"

        # Worker B sees only A's objects with view_others
        repl_b.execute_code('others = store.view_others()')
        others_b = repl_b.locals.get("others")

        assert len(others_b) == 1
        assert others_b[0]["worker"] == "worker_A"

        repl_a.cleanup()
        repl_b.cleanup()

    def test_view_sees_all_workers(self):
        """store.view() sees objects from all workers including own."""
        shared = SharedStore()
        repl_a = LocalREPL(shared_store=shared, worker_id="worker_A")
        repl_b = LocalREPL(shared_store=shared, worker_id="worker_B")

        # Both workers create objects
        repl_a.execute_code('store.create(type="note", description="From A", content="...")')
        repl_b.execute_code('store.create(type="note", description="From B", content="...")')

        # Worker A sees both with view()
        repl_a.execute_code('all_items = store.view()')
        all_items = repl_a.locals.get("all_items")

        assert len(all_items) == 2
        workers = {item["worker"] for item in all_items}
        assert workers == {"worker_A", "worker_B"}

        repl_a.cleanup()
        repl_b.cleanup()


class TestIsolatedVsSharedStore:
    """Compare isolated store (baseline) vs shared store behavior."""

    def test_isolated_stores_dont_share(self):
        """Baseline: separate REPLs have isolated stores."""
        repl_a = LocalREPL()  # No shared_store
        repl_b = LocalREPL()  # No shared_store

        # Worker A creates an object
        repl_a.execute_code('store.create(type="note", description="From A", content="...")')

        # Verify A has it
        repl_a.execute_code('my_items = store.view()')
        my_items = repl_a.locals.get("my_items")
        assert len(my_items) == 1

        # Worker B cannot see it (isolated stores)
        repl_b.execute_code('results = store.view()')
        results = repl_b.locals.get("results")

        assert len(results) == 0  # B's store is empty

        repl_a.cleanup()
        repl_b.cleanup()

    def test_shared_stores_do_share(self):
        """Shared store: separate REPLs see each other's objects."""
        shared = SharedStore()
        repl_a = LocalREPL(shared_store=shared, worker_id="worker_A")
        repl_b = LocalREPL(shared_store=shared, worker_id="worker_B")

        # Worker A creates an object
        repl_a.execute_code('store.create(type="note", description="From A", content="...")')

        # Worker B CAN see it (shared store)
        repl_b.execute_code('results = store.view()')
        results = repl_b.locals.get("results")

        assert len(results) == 1  # B sees A's object

        repl_a.cleanup()
        repl_b.cleanup()

    def test_shared_store_real_time_visibility(self):
        """Objects created by one worker are immediately visible to another."""
        shared = SharedStore()
        repl_a = LocalREPL(shared_store=shared, worker_id="worker_A")
        repl_b = LocalREPL(shared_store=shared, worker_id="worker_B")

        # Initially both see nothing
        repl_a.execute_code('count_before = len(store.view())')
        repl_b.execute_code('count_before = len(store.view())')
        assert repl_a.locals.get("count_before") == 0
        assert repl_b.locals.get("count_before") == 0

        # A creates an object
        repl_a.execute_code('store.create(type="note", description="Created by A", content="...")')

        # B immediately sees it (no need to "sync" or "refresh")
        repl_b.execute_code('count_after = len(store.view())')
        assert repl_b.locals.get("count_after") == 1

        repl_a.cleanup()
        repl_b.cleanup()


class TestSharedStoreWithCommitProtocol:
    """Test that shared store works alongside commit protocol."""

    def test_apply_commit_works_with_shared_store(self):
        """apply_commit works when using shared store."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, worker_id="worker_A")

        code = '''
commit = {"commit_id": "test_commit", "creates": [{"type": "evidence", "id": "e1", "description": "Found something", "content": "..."}]}
result = apply_commit(commit)
success = result.success
'''
        repl.execute_code(code)

        assert repl.locals.get("success") is True

        # Verify the object is in the shared store
        # Note: apply_commit creates in the REPL's store (which is a proxy to shared store)
        repl.execute_code('items = store.view("type=evidence")')
        items = repl.locals.get("items")
        assert len(items) >= 1

        repl.cleanup()

    def test_store_events_tracked_with_shared_store(self):
        """Store events are tracked in REPLResult when using shared store."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, worker_id="worker_A")

        result = repl.execute_code('store.create(type="note", description="Test", content="...")')

        # store_events should be tracked
        assert len(result.store_events) >= 1
        assert result.store_events[0]["op"] == "create"

        repl.cleanup()


class TestWorkerIdNaming:
    """Test worker ID generation and naming."""

    def test_worker_id_includes_depth(self):
        """Auto-generated worker_id includes depth."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, depth=3)

        assert "3" in repl._worker_id
        repl.cleanup()

    def test_explicit_worker_id_used(self):
        """Explicitly provided worker_id is used."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, worker_id="my_custom_worker")

        assert repl._worker_id == "my_custom_worker"
        assert repl.store.worker_id == "my_custom_worker"
        repl.cleanup()

    def test_object_ids_prefixed_with_worker_id(self):
        """Created object IDs are prefixed with worker_id."""
        shared = SharedStore()
        repl = LocalREPL(shared_store=shared, worker_id="worker_X")

        repl.execute_code('obj_id = store.create(type="note", description="Test", content="...")')
        obj_id = repl.locals.get("obj_id")

        assert obj_id.startswith("worker_X_")
        repl.cleanup()
