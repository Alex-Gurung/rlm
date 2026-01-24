"""Validate API correctness in model output."""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found in model output."""
    severity: ValidationSeverity
    message: str
    code_snippet: str = ""
    line_number: int = 0
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of API validation."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    api_calls_found: list[str] = field(default_factory=list)
    syntax_errors: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class APIChecker:
    """Validate API usage in model-generated code."""

    # Valid API functions and their expected signatures
    VALID_APIS = {
        # Context access
        "list_files": {"min_args": 0, "max_args": 0},
        "read_file": {"min_args": 1, "max_args": 1},
        # Analysis
        "llm_query": {"min_args": 1, "max_args": 3, "kwargs": ["output_format", "system_prompt"]},
        "llm_query_batched": {"min_args": 1, "max_args": 3, "kwargs": ["output_format", "system_prompt"]},
        # Helpers
        "parse_json": {"min_args": 1, "max_args": 1},
        "FINAL": {"min_args": 1, "max_args": 1},
        "FINAL_VAR": {"min_args": 1, "max_args": 1},
        # Store APIs
        "store.create": {"min_args": 0, "max_args": 0, "kwargs": ["type", "description", "content", "tags", "parent"]},
        "store.view": {"min_args": 0, "max_args": 1},
        "store.view_others": {"min_args": 0, "max_args": 0},
        "store.get": {"min_args": 1, "max_args": 1},
        "store.search": {"min_args": 1, "max_args": 1},
        "store.children": {"min_args": 1, "max_args": 1},
        "store.summary": {"min_args": 0, "max_args": 0},
        "store.llm_map": {"min_args": 1, "max_args": 1},
        # Worker
        "rlm_worker": {"min_args": 1, "max_args": 1},
    }

    # Common mistakes and their corrections
    # Use negative lookbehind to avoid matching valid APIs
    COMMON_MISTAKES = {
        r"context\.read": ("context.read() is not valid", "Use read_file(name) instead"),
        r"(?<!list_)files\(\)": ("files() is not valid", "Use list_files() instead"),
        r"get_files\(\)": ("get_files() is not valid", "Use list_files() instead"),
        r"(?<!read_)open\(": ("open() is not available", "Use read_file(name) instead"),
        r"(?<!llm_)(?<!_)llm\(": ("llm() is not valid", "Use llm_query(prompt) instead"),
        r"query_llm\(": ("query_llm() is not valid", "Use llm_query(prompt) instead"),
        r"ANSWER\(": ("ANSWER() is not valid", "Use FINAL(answer) instead"),
        r"RESULT\(": ("RESULT() is not valid", "Use FINAL(answer) instead"),
        r"store\.save\(": ("store.save() is not valid", "Use store.create(...) instead"),
        r"store\.add\(": ("store.add() is not valid", "Use store.create(...) instead"),
    }

    # Patterns that suggest asking sub-LLM for final answer (a common mistake)
    SUB_LLM_FINAL_ANSWER_PATTERNS = [
        r'llm_query\s*\([^)]*final\s+answer',
        r'llm_query\s*\([^)]*answer\s+the\s+question',
        r'llm_query\s*\([^)]*give\s+me\s+the\s+answer',
        r'llm_query\s*\([^)]*what\s+is\s+the\s+answer',
    ]

    def __init__(self, include_store_apis: bool = False):
        self.include_store_apis = include_store_apis

    def validate_code(self, code: str) -> ValidationResult:
        """Validate a code block for API correctness."""
        result = ValidationResult(is_valid=True)

        # Check syntax
        syntax_issues = self._check_syntax(code)
        if syntax_issues:
            result.syntax_errors = syntax_issues
            result.is_valid = False

        # Check for common mistakes
        mistake_issues = self._check_common_mistakes(code)
        result.issues.extend(mistake_issues)

        # Check for sub-LLM final answer pattern
        sub_llm_issues = self._check_sub_llm_final_answer(code)
        result.issues.extend(sub_llm_issues)

        # Extract and validate API calls
        api_calls, api_issues = self._validate_api_calls(code)
        result.api_calls_found = api_calls
        result.issues.extend(api_issues)

        # Mark as invalid if there are errors
        if result.errors:
            result.is_valid = False

        return result

    def _check_syntax(self, code: str) -> list[str]:
        """Check for Python syntax errors."""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Line {e.lineno}: {e.msg}")
        return errors

    def _check_common_mistakes(self, code: str) -> list[ValidationIssue]:
        """Check for common API usage mistakes."""
        issues = []
        for pattern, (message, suggestion) in self.COMMON_MISTAKES.items():
            if re.search(pattern, code, re.IGNORECASE):
                match = re.search(pattern, code, re.IGNORECASE)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=message,
                    code_snippet=match.group(0) if match else "",
                    suggestion=suggestion,
                ))
        return issues

    def _check_sub_llm_final_answer(self, code: str) -> list[ValidationIssue]:
        """Check for pattern of asking sub-LLM for final answer."""
        issues = []
        code_lower = code.lower()
        for pattern in self.SUB_LLM_FINAL_ANSWER_PATTERNS:
            if re.search(pattern, code_lower):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Possible anti-pattern: asking sub-LLM for final answer",
                    suggestion="Synthesize the answer yourself and call FINAL(answer)",
                ))
                break
        return issues

    def _validate_api_calls(self, code: str) -> tuple[list[str], list[ValidationIssue]]:
        """Extract and validate API calls."""
        api_calls = []
        issues = []

        # Find all function calls
        call_pattern = re.compile(r'(\w+(?:\.\w+)?)\s*\(')

        for match in call_pattern.finditer(code):
            func_name = match.group(1)

            # Check if it's a known API
            if func_name in self.VALID_APIS:
                api_calls.append(func_name)
            elif func_name.startswith("store.") and func_name in self.VALID_APIS:
                if self.include_store_apis:
                    api_calls.append(func_name)
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Store API '{func_name}' used but store not enabled",
                        code_snippet=func_name,
                    ))

        return api_calls, issues

    def validate_repl_blocks(self, repl_blocks: list[str]) -> ValidationResult:
        """Validate multiple REPL blocks."""
        combined_result = ValidationResult(is_valid=True)

        for block in repl_blocks:
            result = self.validate_code(block)
            combined_result.issues.extend(result.issues)
            combined_result.api_calls_found.extend(result.api_calls_found)
            combined_result.syntax_errors.extend(result.syntax_errors)

            if not result.is_valid:
                combined_result.is_valid = False

        return combined_result


def validate_output(output: str, include_store: bool = False) -> dict[str, Any]:
    """Convenience function to validate model output."""
    from .output_parser import OutputParser

    parser = OutputParser()
    parsed = parser.parse(output)

    checker = APIChecker(include_store_apis=include_store)
    result = checker.validate_repl_blocks(parsed.repl_blocks)

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "syntax_errors": result.syntax_errors,
        "api_calls": result.api_calls_found,
        "issues": [
            {
                "severity": i.severity.value,
                "message": i.message,
                "suggestion": i.suggestion,
            }
            for i in result.issues
        ],
    }
