"""Parse model output to detect REPL code blocks, FINAL() calls, and API usage patterns."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    """Types of actions detected in model output."""
    REPL_CODE = "repl_code"
    FINAL_CALL = "final_call"
    FINAL_VAR_CALL = "final_var_call"
    LLM_QUERY = "llm_query"
    LLM_QUERY_BATCHED = "llm_query_batched"
    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    STORE_CREATE = "store_create"
    STORE_VIEW = "store_view"
    STORE_LLM_MAP = "store_llm_map"
    RLM_WORKER = "rlm_worker"
    PARSE_JSON = "parse_json"
    PRINT = "print"
    UNKNOWN = "unknown"


@dataclass
class DetectedAction:
    """A single detected action from model output."""
    action_type: ActionType
    raw_text: str
    arguments: list[str] = field(default_factory=list)
    line_number: int = 0


@dataclass
class ParsedOutput:
    """Parsed model output with detected patterns."""
    raw_output: str
    repl_blocks: list[str] = field(default_factory=list)
    actions: list[DetectedAction] = field(default_factory=list)
    final_answer: str | None = None
    final_var_name: str | None = None
    has_exploration: bool = False
    has_analysis: bool = False
    prose_sections: list[str] = field(default_factory=list)

    @property
    def has_final(self) -> bool:
        return self.final_answer is not None or self.final_var_name is not None

    @property
    def api_calls(self) -> list[DetectedAction]:
        """Return only API call actions."""
        api_types = {
            ActionType.LLM_QUERY, ActionType.LLM_QUERY_BATCHED,
            ActionType.LIST_FILES, ActionType.READ_FILE,
            ActionType.STORE_CREATE, ActionType.STORE_VIEW,
            ActionType.STORE_LLM_MAP, ActionType.RLM_WORKER,
            ActionType.PARSE_JSON,
        }
        return [a for a in self.actions if a.action_type in api_types]


class OutputParser:
    """Parse model output to extract structured information."""

    # Pattern for REPL code blocks
    REPL_BLOCK_PATTERN = re.compile(r'```repl\n(.*?)```', re.DOTALL)

    # Pattern for FINAL() calls - multiple patterns to catch different styles
    FINAL_PATTERN = re.compile(r'FINAL\s*\(\s*(["\'])(.*?)\1\s*\)', re.DOTALL)  # FINAL("string")
    FINAL_PATTERN_MULTILINE = re.compile(r'FINAL\s*\(\s*"""(.*?)"""\s*\)', re.DOTALL)  # FINAL("""multiline""")
    FINAL_PATTERN_VAR = re.compile(r'FINAL\s*\(\s*(\w+)\s*\)')  # FINAL(variable)
    FINAL_PATTERN_ANY = re.compile(r'FINAL\s*\([^)]+\)')  # FINAL(anything) - for detection
    FINAL_VAR_PATTERN = re.compile(r'FINAL_VAR\s*\(\s*(\w+)\s*\)')

    # API call patterns
    API_PATTERNS = {
        ActionType.LLM_QUERY: re.compile(r'llm_query\s*\('),
        ActionType.LLM_QUERY_BATCHED: re.compile(r'llm_query_batched\s*\('),
        ActionType.LIST_FILES: re.compile(r'list_files\s*\('),
        ActionType.READ_FILE: re.compile(r'read_file\s*\('),
        ActionType.STORE_CREATE: re.compile(r'store\.create\s*\('),
        ActionType.STORE_VIEW: re.compile(r'store\.view\s*\('),
        ActionType.STORE_LLM_MAP: re.compile(r'store\.llm_map\s*\('),
        ActionType.RLM_WORKER: re.compile(r'rlm_worker\s*\('),
        ActionType.PARSE_JSON: re.compile(r'parse_json\s*\('),
        ActionType.PRINT: re.compile(r'print\s*\('),
    }

    # Exploration patterns (indicates the model is exploring context)
    EXPLORATION_PATTERNS = [
        re.compile(r'list_files\s*\(\s*\)'),
        re.compile(r'print\s*\(\s*context'),
        re.compile(r'context\s*\['),
        re.compile(r'len\s*\(\s*context'),
    ]

    # Analysis patterns (indicates the model is doing analysis)
    ANALYSIS_PATTERNS = [
        re.compile(r'llm_query'),
        re.compile(r'llm_query_batched'),
        re.compile(r'store\.llm_map'),
    ]

    def parse(self, output: str) -> ParsedOutput:
        """Parse model output and extract structured information."""
        result = ParsedOutput(raw_output=output)

        # Extract REPL blocks
        result.repl_blocks = self._extract_repl_blocks(output)

        # Extract actions from REPL blocks
        for i, block in enumerate(result.repl_blocks):
            block_actions = self._extract_actions_from_code(block)
            result.actions.extend(block_actions)

        # Check for FINAL calls
        result.final_answer = self._extract_final_answer(output)
        result.final_var_name = self._extract_final_var(output)

        # Check for workflow steps
        result.has_exploration = self._check_exploration(output)
        result.has_analysis = self._check_analysis(output)

        # Extract prose sections (text outside code blocks)
        result.prose_sections = self._extract_prose(output)

        return result

    def _extract_repl_blocks(self, output: str) -> list[str]:
        """Extract all ```repl code blocks."""
        matches = self.REPL_BLOCK_PATTERN.findall(output)
        return [m.strip() for m in matches]

    def _extract_actions_from_code(self, code: str) -> list[DetectedAction]:
        """Extract detected actions from a code block."""
        actions = []

        for line_num, line in enumerate(code.split('\n'), 1):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            for action_type, pattern in self.API_PATTERNS.items():
                if pattern.search(line):
                    actions.append(DetectedAction(
                        action_type=action_type,
                        raw_text=line,
                        line_number=line_num,
                    ))

        return actions

    def _extract_final_answer(self, output: str) -> str | None:
        """Extract the answer from FINAL() call."""
        # Try multiline first
        match = self.FINAL_PATTERN_MULTILINE.search(output)
        if match:
            return match.group(1).strip()

        # Try single-line string literal patterns
        match = self.FINAL_PATTERN.search(output)
        if match:
            return match.group(2)

        # Try variable pattern FINAL(var_name)
        match = self.FINAL_PATTERN_VAR.search(output)
        if match:
            return f"<var:{match.group(1)}>"  # Indicate it's a variable reference

        # Check for any FINAL() call (catches expressions like FINAL(x + y))
        if self.FINAL_PATTERN_ANY.search(output):
            return "<expression>"  # Some FINAL call exists

        return None

    def _extract_final_var(self, output: str) -> str | None:
        """Extract variable name from FINAL_VAR() call."""
        match = self.FINAL_VAR_PATTERN.search(output)
        if match:
            return match.group(1)
        return None

    def _check_exploration(self, output: str) -> bool:
        """Check if output contains exploration patterns."""
        for pattern in self.EXPLORATION_PATTERNS:
            if pattern.search(output):
                return True
        return False

    def _check_analysis(self, output: str) -> bool:
        """Check if output contains analysis patterns."""
        for pattern in self.ANALYSIS_PATTERNS:
            if pattern.search(output):
                return True
        return False

    def _extract_prose(self, output: str) -> list[str]:
        """Extract text outside of code blocks."""
        # Remove all code blocks
        text = re.sub(r'```.*?```', '', output, flags=re.DOTALL)
        # Split into paragraphs and filter empty
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs


def quick_parse(output: str) -> dict[str, Any]:
    """Quick parse for simple use cases - returns a dict summary."""
    parser = OutputParser()
    result = parser.parse(output)

    return {
        "has_repl_code": len(result.repl_blocks) > 0,
        "repl_block_count": len(result.repl_blocks),
        "has_final": result.has_final,
        "final_answer": result.final_answer,
        "final_var_name": result.final_var_name,
        "has_exploration": result.has_exploration,
        "has_analysis": result.has_analysis,
        "api_call_count": len(result.api_calls),
        "api_calls": [a.action_type.value for a in result.api_calls],
    }
