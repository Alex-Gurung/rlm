"""Tests for RLM prompt presets."""

from rlm import RLM
from rlm.utils.prompts import RLM_SYSTEM_PROMPT, STORE_PROMPT_ADDON
from rlm.utils.prompts_legacy import RLM_SYSTEM_PROMPT_LEGACY


def test_prompt_preset_default():
    rlm = RLM(backend="openai", backend_kwargs={"model_name": "dummy"})
    assert RLM_SYSTEM_PROMPT[:40] in rlm.system_prompt


def test_prompt_preset_legacy():
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "dummy"},
        prompt_preset="legacy",
    )
    assert RLM_SYSTEM_PROMPT_LEGACY[:40] in rlm.system_prompt


def test_prompt_preset_invalid():
    try:
        RLM(
            backend="openai",
            backend_kwargs={"model_name": "dummy"},
            prompt_preset="unknown",
        )
    except ValueError as exc:
        assert "prompt_preset" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid prompt_preset")


def test_store_prompt_not_duplicated_when_custom_includes_addon():
    custom = RLM_SYSTEM_PROMPT + STORE_PROMPT_ADDON
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "dummy"},
        custom_system_prompt=custom,
        store_mode="shared",
    )
    assert rlm.system_prompt.count("## Shared Store") == 1
