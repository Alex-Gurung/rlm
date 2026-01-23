"""
Unit tests for the commit protocol.
"""

import pytest

from rlm.core.commit import (
    Commit,
    CommitCreate,
    CommitLink,
    CommitUpdate,
    MergeResult,
    apply_commit,
    parse_commit,
)
from rlm.core.store import Store


class TestParseCommit:
    """Tests for parse_commit function."""

    def test_parse_pure_json(self):
        """Parse pure JSON commit."""
        text = '{"commit_id": "test_1", "creates": [{"type": "note", "id": "n1", "description": "A note", "content": "Hello"}]}'
        commit = parse_commit(text)

        assert commit.commit_id == "test_1"
        assert len(commit.creates) == 1
        assert commit.creates[0].type == "note"
        assert commit.creates[0].id == "n1"
        assert commit.creates[0].description == "A note"
        assert commit.creates[0].content == "Hello"
        assert commit.error is None

    def test_parse_fenced_json(self):
        """Parse JSON in markdown fences."""
        text = """Here is the commit:
```json
{"commit_id": "fenced", "creates": []}
```
"""
        commit = parse_commit(text)
        assert commit.commit_id == "fenced"
        assert commit.error is None

    def test_parse_embedded_json(self):
        """Parse JSON embedded in text."""
        text = """
The analysis found nothing relevant.

{"commit_id": "embedded", "creates": [], "links": []}

That's all.
"""
        commit = parse_commit(text)
        assert commit.commit_id == "embedded"
        assert commit.error is None

    def test_parse_with_fallback_id(self):
        """Use fallback ID when commit_id missing."""
        text = '{"creates": []}'
        commit = parse_commit(text, fallback_id="fallback_123")
        assert commit.commit_id == "fallback_123"

    def test_parse_invalid_returns_error(self):
        """Invalid JSON returns commit with error."""
        text = "This is not JSON at all"
        commit = parse_commit(text, fallback_id="error_test")
        assert commit.commit_id == "error_test"
        assert commit.error is not None
        assert "Failed to parse" in commit.error

    def test_parse_full_commit(self):
        """Parse a full commit with creates, links, and updates."""
        text = """{
            "commit_id": "full_test",
            "creates": [
                {"type": "evidence", "id": "e1", "description": "Found X", "content": {"quote": "..."}}
            ],
            "links": [
                {"type": "supports", "src": "e1", "dst": "hypothesis/H"}
            ],
            "proposes_updates": [
                {"target_id": "hypothesis/H", "patch": {"evidence_count": 1}}
            ]
        }"""
        commit = parse_commit(text)

        assert commit.commit_id == "full_test"
        assert len(commit.creates) == 1
        assert commit.creates[0].type == "evidence"
        assert len(commit.links) == 1
        assert commit.links[0].type == "supports"
        assert len(commit.proposes_updates) == 1
        assert commit.proposes_updates[0].target_id == "hypothesis/H"


class TestCommitFromDict:
    """Tests for Commit.from_dict."""

    def test_from_dict_minimal(self):
        """Create commit from minimal dict."""
        commit = Commit.from_dict({"commit_id": "min"})
        assert commit.commit_id == "min"
        assert commit.creates == []
        assert commit.links == []
        assert commit.proposes_updates == []

    def test_from_dict_full(self):
        """Create commit from full dict."""
        d = {
            "commit_id": "full",
            "creates": [{"type": "note", "id": "n1", "description": "Note 1", "content": "..."}],
            "links": [{"type": "supports", "src": "n1", "dst": "target"}],
            "proposes_updates": [{"target_id": "target", "patch": {"key": "value"}}],
        }
        commit = Commit.from_dict(d)

        assert commit.commit_id == "full"
        assert len(commit.creates) == 1
        assert len(commit.links) == 1
        assert len(commit.proposes_updates) == 1

    def test_to_dict_roundtrip(self):
        """Commit.to_dict and from_dict roundtrip."""
        original = Commit(
            commit_id="roundtrip",
            creates=[CommitCreate(type="note", id="n1", description="Note", content="...")],
            links=[CommitLink(type="supports", src="n1", dst="target")],
            proposes_updates=[CommitUpdate(target_id="target", patch={"k": "v"})],
        )

        d = original.to_dict()
        restored = Commit.from_dict(d)

        assert restored.commit_id == original.commit_id
        assert len(restored.creates) == len(original.creates)
        assert restored.creates[0].id == original.creates[0].id


class TestApplyCommit:
    """Tests for apply_commit function."""

    def test_apply_empty_commit(self):
        """Apply commit with no operations."""
        store = Store()
        commit = Commit(commit_id="empty")

        result = apply_commit(store, commit)

        assert result.success
        assert result.commit_id == "empty"
        assert result.created_ids == {}
        assert result.links_created == 0
        assert result.proposals_stored == 0

    def test_apply_creates_objects(self):
        """Apply commit creates objects in store."""
        store = Store()
        commit = Commit(
            commit_id="create_test",
            creates=[
                CommitCreate(type="note", id="n1", description="Note 1", content="Hello"),
                CommitCreate(type="claim", id="c1", description="Claim 1", content="World"),
            ],
        )

        result = apply_commit(store, commit)

        assert result.success
        assert len(result.created_ids) >= 2  # At least the two objects
        # Check objects exist in store
        for local_id in ["n1", "c1"]:
            global_id = result.created_ids.get(local_id)
            assert global_id is not None
            obj = store.get(global_id)
            assert obj is not None

    def test_apply_with_batch_prefix(self):
        """Apply commit with batch prefix for namespacing."""
        store = Store()
        commit = Commit(
            commit_id="worker_7",
            creates=[CommitCreate(type="note", id="n1", description="Note", content="...")],
        )

        result = apply_commit(store, commit, batch_prefix="batch_123")

        assert result.success
        # The global ID should be namespaced
        assert "n1" in result.created_ids

    def test_apply_creates_links(self):
        """Apply commit creates link objects."""
        store = Store()
        commit = Commit(
            commit_id="link_test",
            creates=[CommitCreate(type="note", id="n1", description="Note", content="...")],
            links=[CommitLink(type="supports", src="n1", dst="external_target")],
        )

        result = apply_commit(store, commit)

        assert result.success
        assert result.links_created == 1

    def test_apply_stores_proposals(self):
        """Apply commit stores update proposals."""
        store = Store()
        commit = Commit(
            commit_id="proposal_test",
            proposes_updates=[CommitUpdate(target_id="some_target", patch={"key": "value"})],
        )

        result = apply_commit(store, commit)

        assert result.success
        assert result.proposals_stored == 1
        # Check proposal was stored
        proposals = store.view("type=update_proposal")
        assert len(proposals) == 1

    def test_apply_commit_with_error(self):
        """Apply commit that has an error returns immediately."""
        store = Store()
        commit = Commit(commit_id="error_commit", error="Parse failed")

        result = apply_commit(store, commit)

        assert not result.success
        assert "Parse failed" in result.errors


class TestMergeResult:
    """Tests for MergeResult class."""

    def test_success_property(self):
        """MergeResult.success reflects error state."""
        success_result = MergeResult(commit_id="ok", created_ids={}, links_created=0, proposals_stored=0)
        assert success_result.success

        error_result = MergeResult(
            commit_id="fail",
            created_ids={},
            links_created=0,
            proposals_stored=0,
            errors=["Something went wrong"],
        )
        assert not error_result.success

    def test_to_dict(self):
        """MergeResult.to_dict serializes correctly."""
        result = MergeResult(
            commit_id="test",
            created_ids={"n1": "abc123"},
            links_created=2,
            proposals_stored=1,
            errors=[],
        )
        d = result.to_dict()

        assert d["commit_id"] == "test"
        assert d["created_ids"] == {"n1": "abc123"}
        assert d["links_created"] == 2
        assert d["proposals_stored"] == 1
        assert d["success"] is True


class TestStoreIntegration:
    """Integration tests for commit protocol with Store."""

    def test_store_card_view(self):
        """Store.card_view returns lightweight cards."""
        store = Store()
        store.create(type="hypothesis", description="Hypothesis H", content="...")
        store.create(type="note", description="Some note", content="...")

        cards = store.card_view("type=hypothesis")

        assert len(cards) == 1
        assert cards[0]["type"] == "hypothesis"
        assert "id" in cards[0]
        assert "description" in cards[0]
        assert "tags" in cards[0]

    def test_store_apply_commit_wrapper(self):
        """Store.apply_commit wrapper works."""
        store = Store()
        commit = Commit(
            commit_id="wrapper_test",
            creates=[CommitCreate(type="note", id="n1", description="Note", content="...")],
        )

        result = store.apply_commit(commit)

        assert result.success
        assert "n1" in result.created_ids


class TestLocalREPLIntegration:
    """Integration tests for commit protocol with LocalREPL."""

    def test_commit_events_flow_to_result(self):
        """CommitEvents flow from apply_commit() to REPLResult."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL()
        code = '''
commit = {"commit_id": "test_event", "creates": [{"type": "note", "id": "n1", "description": "Test", "content": "..."}]}
result = apply_commit(commit)
'''
        result = repl.execute_code(code)

        # Verify commit event was tracked
        assert len(result.commit_events) == 1
        assert result.commit_events[0].commit_id == "test_event"
        # creates_count includes both local_id and global_id mappings (2 entries per object)
        assert result.commit_events[0].creates_count >= 1
        assert result.commit_events[0].status == "ok"

        repl.cleanup()

    def test_multiple_commits_tracked(self):
        """Multiple apply_commit calls create multiple events."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL()
        code = '''
commit1 = {"commit_id": "first", "creates": []}
commit2 = {"commit_id": "second", "creates": [{"type": "note", "id": "n1", "description": "Test", "content": "..."}]}
apply_commit(commit1)
apply_commit(commit2)
'''
        result = repl.execute_code(code)

        assert len(result.commit_events) == 2
        assert result.commit_events[0].commit_id == "first"
        assert result.commit_events[1].commit_id == "second"

        repl.cleanup()

    def test_commit_events_cleared_between_executions(self):
        """Commit events are cleared between execute_code calls."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL()

        # First execution
        result1 = repl.execute_code('apply_commit({"commit_id": "first", "creates": []})')
        assert len(result1.commit_events) == 1

        # Second execution - should not include first commit
        result2 = repl.execute_code('apply_commit({"commit_id": "second", "creates": []})')
        assert len(result2.commit_events) == 1
        assert result2.commit_events[0].commit_id == "second"

        repl.cleanup()

    def test_subcall_budget_enforcement(self):
        """llm_query returns error when subcall budget exceeded."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(subcall_budget=2)
        # Manually set subcall count to exceed budget
        repl._subcall_count = 3

        result = repl._llm_query("test prompt")

        assert "budget exceeded" in result.lower()

        repl.cleanup()

    def test_rlm_worker_depth_exceeded(self):
        """rlm_worker returns error commit at max depth."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(depth=2, max_depth=2)
        result = repl._rlm_worker("test prompt")

        assert result.get("error") is not None
        assert "depth" in result["error"].lower()
        assert result["commit_id"] == "depth_exceeded"

        repl.cleanup()

    def test_rlm_worker_no_backend(self):
        """rlm_worker returns error when no backend configured."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(depth=0, max_depth=2, backend=None)
        result = repl._rlm_worker("test prompt")

        assert result.get("error") is not None
        assert "backend" in result["error"].lower()
        assert result["commit_id"] == "no_backend"

        repl.cleanup()

    def test_rlm_worker_budget_exceeded(self):
        """rlm_worker returns error when subcall budget exceeded."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(depth=0, max_depth=2, subcall_budget=2, backend="mock")
        repl._subcall_count = 3  # Already exceeded

        result = repl._rlm_worker("test prompt")

        assert result.get("error") is not None
        assert "budget" in result["error"].lower()
        assert result["commit_id"] == "budget_exceeded"

        repl.cleanup()

    def test_commit_event_serialization(self):
        """CommitEvent.to_dict serializes all fields correctly."""
        from rlm.core.types import CommitEvent

        event = CommitEvent(
            commit_id="test",
            creates_count=3,
            links_count=2,
            proposals_count=1,
            status="ok",
            errors=[],
        )

        d = event.to_dict()

        assert d["commit_id"] == "test"
        assert d["creates_count"] == 3
        assert d["links_count"] == 2
        assert d["proposals_count"] == 1
        assert d["status"] == "ok"
        assert d["errors"] == []

    def test_repl_result_includes_commit_events_in_dict(self):
        """REPLResult.to_dict includes commit_events."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL()
        code = 'apply_commit({"commit_id": "serialization_test", "creates": []})'
        result = repl.execute_code(code)

        d = result.to_dict()

        assert "commit_events" in d
        assert len(d["commit_events"]) == 1
        assert d["commit_events"][0]["commit_id"] == "serialization_test"

        repl.cleanup()
