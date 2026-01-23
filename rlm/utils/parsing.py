"""
Parsing utilities for RLM trjaectories.
"""

import re
from typing import TYPE_CHECKING

from rlm.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from rlm.environments.base_env import BaseEnv


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def find_final_answer(text: str, environment: "BaseEnv | None" = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return the final answer string.

    If FINAL_VAR is found and an environment is provided, executes code to retrieve the variable value.
    Returns None if neither pattern is found.

    Note: FINAL() calls inside code blocks are ignored - those are handled by the REPL execution.

    Args:
        text: The response text to parse
        environment: Optional environment to execute code for FINAL_VAR retrieval

    Returns:
        The final answer string, or None if no final answer pattern is found
    """
    # Strip out code blocks to avoid matching FINAL() inside code
    # This prevents capturing variable names like FINAL(result) in code blocks
    text_without_code = re.sub(r"```(?:repl|python)?\s*\n.*?\n```", "", text, flags=re.DOTALL)

    # Find FINAL_VAR and FINAL occurrences, then select the last by position.
    final_var_pattern = r"^\s*FINAL_VAR\(([^)]+)\)"
    final_pattern = r"^\s*FINAL\(([\s\S]*?)\)\s*$"

    final_var_matches = [
        ("final_var", match)
        for match in re.finditer(final_var_pattern, text_without_code, re.MULTILINE)
    ]
    final_matches = [
        ("final", match)
        for match in re.finditer(final_pattern, text_without_code, re.MULTILINE)
    ]

    if not final_var_matches and not final_matches:
        return None

    kind, match = max(final_var_matches + final_matches, key=lambda item: item[1].start())

    if kind == "final_var":
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is not None:
            # FINAL_VAR prints FINAL(value) to stdout - we just need to call it
            result = environment.execute_code(f"FINAL_VAR({variable_name!r})")
            # Parse the last FINAL(...) pattern from stdout
            stdout = result.stdout.strip()
            finals = re.findall(r"FINAL\((.*?)\)", stdout, re.DOTALL)
            if finals:
                return finals[-1].strip()
            if stdout:
                return stdout
            if result.stderr.strip():
                return result.stderr.strip()
            return None
        return None

    return match.group(1).strip()

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger) -> str | None:
    """Check if response contains a final answer."""
    # Use the new find_final_answer function which handles both FINAL and FINAL_VAR
    return find_final_answer(response, environment=repl_env)


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
