"""Helpers to construct conversation states for testing."""

from dataclasses import dataclass, field
from typing import Any

from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    STORE_PROMPT_ADDON,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.core.types import QueryMetadata


@dataclass
class ConversationState:
    """Represents a conversation state for testing."""
    messages: list[dict[str, str]] = field(default_factory=list)
    context_type: str = "string"
    context_lengths: list[int] = field(default_factory=lambda: [100])

    def add_message(self, role: str, content: str) -> "ConversationState":
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        return self

    def to_messages(self) -> list[dict[str, str]]:
        """Return the message list for API calls."""
        return self.messages.copy()


class MessageBuilder:
    """Build conversation states for prompt testing."""

    def __init__(
        self,
        include_store: bool = False,
        custom_system_prompt: str | None = None,
    ):
        self.include_store = include_store
        self.custom_system_prompt = custom_system_prompt

    def _get_system_prompt(self) -> str:
        """Get the system prompt to use."""
        if self.custom_system_prompt:
            return self.custom_system_prompt

        prompt = RLM_SYSTEM_PROMPT
        if self.include_store:
            prompt += "\n\n" + STORE_PROMPT_ADDON
        return prompt

    def build_initial_state(
        self,
        question: str,
        context_type: str = "string",
        context_lengths: list[int] | None = None,
        context_count: int = 1,
    ) -> ConversationState:
        """Build initial conversation state with system prompt and user question."""
        if context_lengths is None:
            context_lengths = [1000]

        # QueryMetadata expects a prompt and calculates metadata from it
        # We create a dummy prompt matching the expected size
        dummy_prompt = "x" * sum(context_lengths)
        metadata = QueryMetadata(dummy_prompt)

        system_messages = build_rlm_system_prompt(
            self._get_system_prompt(),
            metadata,
        )

        user_message = build_user_prompt(
            root_prompt=question,
            iteration=0,
            context_count=context_count,
        )

        state = ConversationState(
            context_type=context_type,
            context_lengths=context_lengths,
        )

        for msg in system_messages:
            state.add_message(msg["role"], msg["content"])
        state.add_message(user_message["role"], user_message["content"])

        return state

    def build_continuation_state(
        self,
        question: str,
        prior_assistant_response: str,
        prior_execution_output: str,
        context_type: str = "string",
        context_lengths: list[int] | None = None,
        iteration: int = 1,
    ) -> ConversationState:
        """Build a continuation state after one round of execution."""
        state = self.build_initial_state(
            question=question,
            context_type=context_type,
            context_lengths=context_lengths,
        )

        # Add prior assistant response
        state.add_message("assistant", prior_assistant_response)

        # Add execution output as user message (simulating REPL feedback)
        state.add_message("user", f"```output\n{prior_execution_output}\n```")

        return state

    def build_multi_turn_state(
        self,
        question: str,
        turns: list[tuple[str, str]],  # [(assistant_msg, execution_output), ...]
        context_type: str = "string",
        context_lengths: list[int] | None = None,
    ) -> ConversationState:
        """Build a multi-turn conversation state."""
        state = self.build_initial_state(
            question=question,
            context_type=context_type,
            context_lengths=context_lengths,
        )

        for assistant_msg, execution_output in turns:
            state.add_message("assistant", assistant_msg)
            state.add_message("user", f"```output\n{execution_output}\n```")

        return state

    def build_error_recovery_state(
        self,
        question: str,
        error_message: str,
        prior_code: str,
        context_type: str = "string",
    ) -> ConversationState:
        """Build a state where the model needs to recover from an error."""
        prior_response = f"```repl\n{prior_code}\n```"
        error_output = f"Error: {error_message}"

        return self.build_continuation_state(
            question=question,
            prior_assistant_response=prior_response,
            prior_execution_output=error_output,
            context_type=context_type,
        )


def create_simple_test_conversation(
    question: str,
    include_store: bool = False,
) -> list[dict[str, str]]:
    """Convenience function to create a simple test conversation."""
    builder = MessageBuilder(include_store=include_store)
    state = builder.build_initial_state(question=question)
    return state.to_messages()


def create_file_context_conversation(
    question: str,
    file_count: int = 5,
    avg_file_size: int = 2000,
    include_store: bool = False,
) -> list[dict[str, str]]:
    """Create a conversation with file-based context."""
    builder = MessageBuilder(include_store=include_store)
    context_lengths = [avg_file_size] * file_count
    state = builder.build_initial_state(
        question=question,
        context_type="files",
        context_lengths=context_lengths,
        context_count=file_count,
    )
    return state.to_messages()
