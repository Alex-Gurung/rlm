"""Main test harness for prompt testbed."""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

from rlm.utils.prompts import RLM_SYSTEM_PROMPT, STORE_PROMPT_ADDON

from .message_builder import MessageBuilder
from .output_parser import OutputParser, ParsedOutput
from .api_checker import APIChecker, ValidationResult
from .test_cases import TestCase, TestCategory, ALL_TEST_CASES, get_tests_by_category


# Default model for testing (use openai/ prefix for vLLM compatibility)
DEFAULT_MODEL = "openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
DEFAULT_API_BASE = "http://localhost:8000/v1"

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_id: str
    test_name: str
    category: str
    passed: bool
    model_output: str
    parsed_output: ParsedOutput | None = None
    validation_result: ValidationResult | None = None
    failure_reasons: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "has_repl_code": self.parsed_output.has_final if self.parsed_output else False,
            "has_exploration": self.parsed_output.has_exploration if self.parsed_output else False,
            "has_analysis": self.parsed_output.has_analysis if self.parsed_output else False,
            "api_calls": [a.action_type.value for a in self.parsed_output.api_calls] if self.parsed_output else [],
        }


@dataclass
class TestRunSummary:
    """Summary of a test run."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    results: list[TestResult]
    start_time: datetime
    end_time: datetime | None = None
    model_used: str = ""

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": f"{self.pass_rate:.1%}",
            "model_used": self.model_used,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": [r.to_dict() for r in self.results],
        }


class PromptTestRunner:
    """Run prompt tests against a model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_base: str | None = DEFAULT_API_BASE,
        api_key: str | None = "dummy",  # vLLM accepts any key
        verbose: bool = True,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.verbose = verbose

        self.parser = OutputParser()
        self.checker = APIChecker()

    def _log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(message)

    def _call_model(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        """Call the model and return response + token usage."""
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        response = litellm.completion(**kwargs)

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return content, usage

    def _evaluate_output(
        self,
        test: TestCase,
        parsed: ParsedOutput,
        validation: ValidationResult,
    ) -> tuple[bool, list[str]]:
        """Evaluate if the output meets expectations."""
        failures = []
        expected = test.expected

        # Check exploration
        if expected.should_explore and not parsed.has_exploration:
            failures.append("Expected exploration but none detected")

        # Check REPL usage
        if expected.should_use_repl and len(parsed.repl_blocks) == 0:
            failures.append("Expected REPL code but none found")

        # Check FINAL call
        if expected.should_call_final and not parsed.has_final:
            failures.append("Expected FINAL() call but none found")

        # Check sub-LLM usage
        if expected.should_use_sub_llm and not parsed.has_analysis:
            failures.append("Expected sub-LLM usage but none detected")

        # Check expected API calls
        found_apis = [a.action_type.value for a in parsed.api_calls]
        for expected_api in expected.expected_api_calls:
            if expected_api not in found_apis:
                failures.append(f"Expected API call '{expected_api}' not found")

        # Check forbidden patterns
        for pattern in expected.forbidden_patterns:
            if re.search(pattern, parsed.raw_output, re.IGNORECASE):
                failures.append(f"Forbidden pattern found: {pattern}")

        # Check validation errors
        if validation.errors:
            for error in validation.errors:
                failures.append(f"Validation error: {error.message}")

        passed = len(failures) == 0
        return passed, failures

    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case."""
        self._log(f"\n[{test.id}] Running: {test.name}")

        # Use prebuilt messages if available (for fake multi-turn tests)
        if test.prebuilt_messages is not None:
            messages = test.prebuilt_messages
        else:
            # Build messages from scratch
            builder = MessageBuilder(include_store=test.include_store)
            context = test.get_context()

            state = builder.build_initial_state(
                question=test.question,
                context_type=context.context_type,
                context_lengths=[context.length],
            )
            messages = state.to_messages()

        # Call model
        start_time = time.time()
        try:
            output, token_usage = self._call_model(messages)
        except Exception as e:
            return TestResult(
                test_id=test.id,
                test_name=test.name,
                category=test.category.value,
                passed=False,
                model_output="",
                failure_reasons=[f"Model call failed: {str(e)}"],
            )
        execution_time = time.time() - start_time

        # Parse output
        parsed = self.parser.parse(output)

        # Validate API usage
        self.checker.include_store_apis = test.include_store
        validation = self.checker.validate_repl_blocks(parsed.repl_blocks)

        # Evaluate
        passed, failures = self._evaluate_output(test, parsed, validation)

        result = TestResult(
            test_id=test.id,
            test_name=test.name,
            category=test.category.value,
            passed=passed,
            model_output=output,
            parsed_output=parsed,
            validation_result=validation,
            failure_reasons=failures,
            execution_time=execution_time,
            token_usage=token_usage,
        )

        # Log result
        status = "PASS" if passed else "FAIL"
        self._log(f"  [{status}] {execution_time:.2f}s")
        if failures:
            for f in failures:
                self._log(f"    - {f}")

        return result

    def run_category(self, category: TestCategory) -> TestRunSummary:
        """Run all tests in a category."""
        tests = get_tests_by_category(category)
        return self.run_tests(tests)

    def run_tests(self, tests: list[TestCase]) -> TestRunSummary:
        """Run a list of test cases."""
        summary = TestRunSummary(
            total_tests=len(tests),
            passed=0,
            failed=0,
            skipped=0,
            results=[],
            start_time=datetime.now(),
            model_used=self.model,
        )

        self._log(f"\n{'='*60}")
        self._log(f"Running {len(tests)} tests with model: {self.model}")
        self._log(f"{'='*60}")

        for test in tests:
            result = self.run_test(test)
            summary.results.append(result)

            if result.passed:
                summary.passed += 1
            else:
                summary.failed += 1

        summary.end_time = datetime.now()

        # Log summary
        self._log(f"\n{'='*60}")
        self._log(f"SUMMARY: {summary.passed}/{summary.total_tests} passed ({summary.pass_rate:.1%})")
        self._log(f"{'='*60}")

        return summary

    def run_all(self) -> TestRunSummary:
        """Run all test cases."""
        return self.run_tests(ALL_TEST_CASES)

    def save_results(self, summary: TestRunSummary, filename: str | None = None):
        """Save test results to a JSON file."""
        RESULTS_DIR.mkdir(exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        filepath = RESULTS_DIR / filename

        with open(filepath, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)

        self._log(f"\nResults saved to: {filepath}")
        return filepath


def run_quick_test(
    model: str = DEFAULT_MODEL,
    api_base: str | None = DEFAULT_API_BASE,
) -> TestRunSummary:
    """Run a quick subset of tests for validation."""
    runner = PromptTestRunner(model=model, api_base=api_base)

    # Run just a few tests from each category
    quick_tests = [
        ALL_TEST_CASES[0],  # A1 - basic workflow
        ALL_TEST_CASES[4],  # B1 - API usage
        ALL_TEST_CASES[8],  # C1 - context handling
    ]

    return runner.run_tests(quick_tests)


def run_category_test(
    category: str,
    model: str = DEFAULT_MODEL,
    api_base: str | None = DEFAULT_API_BASE,
) -> TestRunSummary:
    """Run all tests in a specific category."""
    try:
        cat = TestCategory[f"{category}_" if len(category) == 1 else category.upper()]
    except KeyError:
        # Try mapping single letter
        cat_map = {c.value: c for c in TestCategory}
        cat = cat_map.get(category.upper())
        if not cat:
            raise ValueError(f"Unknown category: {category}")

    runner = PromptTestRunner(model=model, api_base=api_base)
    return runner.run_category(cat)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run prompt testbed")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to test")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--category", help="Run specific category (A-H)")
    parser.add_argument("--quick", action="store_true", help="Run quick test subset")
    parser.add_argument("--save", action="store_true", help="Save results to file")

    args = parser.parse_args()

    if args.quick:
        summary = run_quick_test(model=args.model, api_base=args.api_base)
    elif args.category:
        summary = run_category_test(args.category, model=args.model, api_base=args.api_base)
    else:
        runner = PromptTestRunner(model=args.model, api_base=args.api_base)
        summary = runner.run_all()

    if args.save:
        runner = PromptTestRunner(model=args.model, api_base=args.api_base, verbose=False)
        runner.save_results(summary)
