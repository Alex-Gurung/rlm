"""Unit tests for Store (non-shared) behavior."""

import pytest

from rlm.core.store import Store


class TestStoreSummary:
    def test_summary_filters_query(self):
        store = Store()
        store.create(type="note", description="Alpha note", content="c1", tags=["t1"])
        store.create(type="evidence", description="Alpha evidence", content="c2", tags=["t2"])

        summary = store.summary('type=note desc~"Alpha"')
        assert summary["types"] == ["note"]
        assert "t1" in summary["tags"]
        assert all(item["type"] == "note" for item in summary["matches"])


class TestStoreLlmMapValidation:
    def test_llm_map_empty_tasks_raises(self):
        store = Store()
        store.set_llm_batch_fn(lambda prompts, model: [])
        with pytest.raises(ValueError, match="non-empty"):
            store.llm_map([])

    def test_llm_map_missing_name_or_prompt_raises(self):
        store = Store()
        store.set_llm_batch_fn(lambda prompts, model: ["ok"])
        with pytest.raises(ValueError, match="name"):
            store.llm_map([{"prompt": "p1"}])
        with pytest.raises(ValueError, match="name"):
            store.llm_map([{"name": "t1"}])

    def test_llm_map_response_count_mismatch_raises(self):
        store = Store()
        store.set_llm_batch_fn(lambda prompts, model: ["only_one"])
        with pytest.raises(ValueError, match="response count"):
            store.llm_map(
                [
                    {"name": "t1", "prompt": "p1"},
                    {"name": "t2", "prompt": "p2"},
                ]
            )
