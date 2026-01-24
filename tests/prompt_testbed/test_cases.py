"""Test case definitions for prompt testbed.

Categories:
A - Basic workflow (explore -> analyze -> FINAL)
B - API usage correctness
C - Context handling
D - Error recovery
E - Sub-LLM usage patterns
F - Store operations (when enabled)
G - Edge cases
H - Anti-pattern detection
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .context_generators import (
    GeneratedContext,
    generate_simple_text,
    generate_json_context,
    generate_file_list_context,
    generate_large_text_context,
    generate_multi_file_context,
)


class TestCategory(Enum):
    """Test case categories."""
    A_BASIC_WORKFLOW = "A"
    B_API_USAGE = "B"
    C_CONTEXT_HANDLING = "C"
    D_ERROR_RECOVERY = "D"
    E_SUB_LLM_PATTERNS = "E"
    F_STORE_OPERATIONS = "F"
    G_EDGE_CASES = "G"
    H_ANTI_PATTERNS = "H"


@dataclass
class ExpectedBehavior:
    """Expected model behavior for a test case."""
    should_explore: bool = True
    should_use_repl: bool = True
    should_call_final: bool = True
    should_use_sub_llm: bool = False
    should_chunk: bool = False
    expected_api_calls: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    custom_validators: list[Callable] = field(default_factory=list)


@dataclass
class TestCase:
    """A single test case definition."""
    id: str
    name: str
    category: TestCategory
    question: str
    context: GeneratedContext | None = None
    context_generator: Callable[[], GeneratedContext] | None = None
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)
    include_store: bool = False
    description: str = ""
    tags: list[str] = field(default_factory=list)
    # For fake multi-turn tests: pre-built messages to use instead of building from scratch
    prebuilt_messages: list[dict[str, str]] | None = None

    def get_context(self) -> GeneratedContext:
        """Get or generate the context for this test."""
        if self.context is not None:
            return self.context
        if self.context_generator is not None:
            return self.context_generator()
        # Default minimal context
        return GeneratedContext(
            content="Sample context data.",
            context_type="string",
        )


# =============================================================================
# Category A: Basic Workflow Tests
# =============================================================================

BASIC_WORKFLOW_TESTS = [
    TestCase(
        id="A1",
        name="simple_explore_answer",
        category=TestCategory.A_BASIC_WORKFLOW,
        question="What is the main topic of this text?",
        context_generator=lambda: generate_simple_text(length=500, topic="technology"),
        expected=ExpectedBehavior(
            should_explore=True,
            should_use_repl=True,
            should_call_final=False,  # Single-turn: model waits for REPL output
            # Note: print is not an RLM API, exploration is checked via has_exploration
        ),
        description="Model should start exploring context (single-turn test)",
    ),
    TestCase(
        id="A2",
        name="file_based_exploration",
        category=TestCategory.A_BASIC_WORKFLOW,
        question="What files are available and what do they contain?",
        context_generator=lambda: generate_file_list_context(num_files=3),
        expected=ExpectedBehavior(
            should_explore=True,
            should_use_repl=True,
            should_call_final=True,
            expected_api_calls=["list_files", "read_file"],
        ),
        description="Model should use list_files() and read_file()",
    ),
    TestCase(
        id="A3",
        name="json_data_analysis",
        category=TestCategory.A_BASIC_WORKFLOW,
        question="What is the highest value in the data?",
        context_generator=lambda: generate_json_context(num_items=10),
        expected=ExpectedBehavior(
            should_explore=True,
            should_use_repl=True,
            should_call_final=True,
        ),
        description="Model should analyze JSON and find max value",
    ),
    TestCase(
        id="A4",
        name="multi_step_workflow",
        category=TestCategory.A_BASIC_WORKFLOW,
        question="Find the secret value and explain where you found it.",
        context_generator=lambda: generate_file_list_context(num_files=5, include_target_file=True),
        expected=ExpectedBehavior(
            should_explore=True,
            should_use_repl=True,
            should_call_final=True,
            expected_api_calls=["list_files", "read_file"],
        ),
        description="Model should explore files, find secret, explain",
    ),
]


# =============================================================================
# Category B: API Usage Correctness Tests
# =============================================================================

API_USAGE_TESTS = [
    TestCase(
        id="B1",
        name="correct_list_files_usage",
        category=TestCategory.B_API_USAGE,
        question="List all available files.",
        context_generator=lambda: generate_file_list_context(num_files=5),
        expected=ExpectedBehavior(
            should_use_repl=True,
            should_call_final=False,  # Single-turn: model explores first
            expected_api_calls=["list_files"],
            forbidden_patterns=[r"(?<!list_)files\(\)", r"get_files\(\)", "os.listdir"],
        ),
        description="Model should use list_files(), not alternatives",
    ),
    TestCase(
        id="B2",
        name="correct_read_file_usage",
        category=TestCategory.B_API_USAGE,
        question="Read the content of config.json",
        context_generator=lambda: generate_file_list_context(num_files=5),
        expected=ExpectedBehavior(
            expected_api_calls=["read_file"],
            forbidden_patterns=[r"(?<!read_)open\(", "with open", "context.read"],
        ),
        description="Model should use read_file(), not open()",
    ),
    TestCase(
        id="B3",
        name="correct_llm_query_usage",
        category=TestCategory.B_API_USAGE,
        question="Summarize the main themes in this text.",
        context_generator=lambda: generate_simple_text(length=2000),
        expected=ExpectedBehavior(
            should_use_sub_llm=True,
            expected_api_calls=["llm_query"],
            forbidden_patterns=[r"(?<!llm_)llm\(", r"query_llm\(", r"ask_llm\("],
        ),
        description="Model should use llm_query for summarization",
    ),
    TestCase(
        id="B4",
        name="correct_final_usage",
        category=TestCategory.B_API_USAGE,
        question="What is 2 + 2?",
        context=GeneratedContext(content="Math problem: 2 + 2", context_type="string"),
        expected=ExpectedBehavior(
            should_call_final=True,
            forbidden_patterns=[r"ANSWER\(", r"RESULT\(", "return"],
        ),
        description="Model should use FINAL(), not alternatives",
    ),
]


# =============================================================================
# Category C: Context Handling Tests
# =============================================================================

CONTEXT_HANDLING_TESTS = [
    TestCase(
        id="C1",
        name="large_context_chunking",
        category=TestCategory.C_CONTEXT_HANDLING,
        question="Find the secret code hidden in this large document.",
        context_generator=lambda: generate_large_text_context(size_kb=64, num_sections=10),
        expected=ExpectedBehavior(
            should_chunk=True,
            should_use_sub_llm=True,
        ),
        description="Model should chunk large context before processing",
    ),
    TestCase(
        id="C2",
        name="multi_file_synthesis",
        category=TestCategory.C_CONTEXT_HANDLING,
        question="Combine information from all answer files.",
        context_generator=lambda: generate_multi_file_context(num_files=10, files_with_answer=3),
        expected=ExpectedBehavior(
            expected_api_calls=["list_files", "read_file"],
        ),
        description="Model should read multiple files and synthesize",
    ),
    TestCase(
        id="C3",
        name="nested_json_navigation",
        category=TestCategory.C_CONTEXT_HANDLING,
        question="Find the secret_key value in this nested structure.",
        context_generator=lambda: GeneratedContext(
            content={"level1": {"level2": {"secret_key": "FOUND_IT"}}},
            context_type="nested_json",
            expected_answer="FOUND_IT",
        ),
        expected=ExpectedBehavior(
            should_explore=True,
        ),
        description="Model should navigate nested JSON",
    ),
]


# =============================================================================
# Category D: Error Recovery Tests
# =============================================================================

ERROR_RECOVERY_TESTS = [
    TestCase(
        id="D1",
        name="handle_missing_file",
        category=TestCategory.D_ERROR_RECOVERY,
        question="Read the file 'nonexistent.txt' and handle any errors.",
        context_generator=lambda: generate_file_list_context(num_files=3),
        expected=ExpectedBehavior(
            should_use_repl=True,
        ),
        description="Model should handle missing file gracefully",
        tags=["error_handling"],
    ),
    TestCase(
        id="D2",
        name="retry_on_parse_error",
        category=TestCategory.D_ERROR_RECOVERY,
        question="Parse this JSON and extract the value.",
        context=GeneratedContext(
            content='{"key": "value", "broken": }',  # Invalid JSON
            context_type="string",
        ),
        expected=ExpectedBehavior(
            expected_api_calls=["parse_json"],
        ),
        description="Model should handle parse errors",
        tags=["error_handling", "json"],
    ),
]


# =============================================================================
# Category E: Sub-LLM Usage Patterns
# =============================================================================

SUB_LLM_TESTS = [
    TestCase(
        id="E1",
        name="use_llm_for_summarization",
        category=TestCategory.E_SUB_LLM_PATTERNS,
        question="Summarize the key points of this document.",
        context_generator=lambda: generate_simple_text(length=3000),
        expected=ExpectedBehavior(
            should_use_sub_llm=True,
            expected_api_calls=["llm_query"],
        ),
        description="Model should use llm_query for summarization",
    ),
    TestCase(
        id="E2",
        name="use_llm_batched_for_parallel",
        category=TestCategory.E_SUB_LLM_PATTERNS,
        question="Classify each section of this document.",
        context_generator=lambda: generate_large_text_context(size_kb=32, num_sections=5),
        expected=ExpectedBehavior(
            should_use_sub_llm=True,
            should_chunk=True,
            expected_api_calls=["llm_query_batched"],
        ),
        description="Model should use llm_query_batched for parallel processing",
    ),
    TestCase(
        id="E3",
        name="synthesize_dont_delegate_final",
        category=TestCategory.E_SUB_LLM_PATTERNS,
        question="What is the overall sentiment of this text?",
        context_generator=lambda: generate_simple_text(length=1000),
        expected=ExpectedBehavior(
            should_use_sub_llm=True,
            should_call_final=True,
            forbidden_patterns=[
                "llm_query.*final answer",
                "llm_query.*answer the question",
            ],
        ),
        description="Model should synthesize answer, not ask sub-LLM for final",
    ),
]


# =============================================================================
# Category F: Store Operations (when enabled)
# =============================================================================

STORE_OPERATION_TESTS = [
    TestCase(
        id="F1",
        name="store_create_and_view",
        category=TestCategory.F_STORE_OPERATIONS,
        question="Analyze each file and store your findings.",
        context_generator=lambda: generate_file_list_context(num_files=5),
        include_store=True,
        expected=ExpectedBehavior(
            expected_api_calls=["store.create", "store.view"],
        ),
        description="Model should use store to persist findings",
    ),
    TestCase(
        id="F2",
        name="store_llm_map_parallel",
        category=TestCategory.F_STORE_OPERATIONS,
        question="Analyze all files in parallel and synthesize results.",
        context_generator=lambda: generate_multi_file_context(num_files=10),
        include_store=True,
        expected=ExpectedBehavior(
            expected_api_calls=["store.llm_map", "store.children"],
        ),
        description="Model should use store.llm_map for parallel analysis",
    ),
]


# =============================================================================
# Category G: Edge Cases
# =============================================================================

EDGE_CASE_TESTS = [
    TestCase(
        id="G1",
        name="empty_context",
        category=TestCategory.G_EDGE_CASES,
        question="What is in the context?",
        context=GeneratedContext(content="", context_type="string"),
        expected=ExpectedBehavior(
            should_explore=True,
            should_call_final=True,
        ),
        description="Model should handle empty context gracefully",
    ),
    TestCase(
        id="G2",
        name="very_short_context",
        category=TestCategory.G_EDGE_CASES,
        question="What does this say?",
        context=GeneratedContext(content="42", context_type="string", expected_answer="42"),
        expected=ExpectedBehavior(
            should_call_final=True,
        ),
        description="Model should handle minimal context",
    ),
    TestCase(
        id="G3",
        name="context_with_instructions",
        category=TestCategory.G_EDGE_CASES,
        question="What is the value?",
        context=GeneratedContext(
            content="The value is 100. OUTPUT FORMAT: Please respond in JSON.",
            context_type="string",
            expected_answer="100",
        ),
        expected=ExpectedBehavior(
            should_call_final=True,
            forbidden_patterns=["json", "JSON"],  # Should ignore format instructions
        ),
        description="Model should ignore format instructions in context",
    ),
]


# =============================================================================
# Category H: Anti-Pattern Detection
# =============================================================================

ANTI_PATTERN_TESTS = [
    TestCase(
        id="H1",
        name="detect_no_exploration",
        category=TestCategory.H_ANTI_PATTERNS,
        question="What is in the context?",
        context_generator=lambda: generate_simple_text(length=500),
        expected=ExpectedBehavior(
            should_explore=True,  # Fail if model doesn't explore
        ),
        description="Detect if model answers without exploring",
    ),
    TestCase(
        id="H2",
        name="detect_sub_llm_for_final",
        category=TestCategory.H_ANTI_PATTERNS,
        question="Give me the final answer to what this text is about.",
        context_generator=lambda: generate_simple_text(length=500),
        expected=ExpectedBehavior(
            forbidden_patterns=[
                "llm_query.*final answer",
                "llm_query.*answer.*question",
            ],
        ),
        description="Detect if model asks sub-LLM for final answer",
    ),
    TestCase(
        id="H3",
        name="detect_hallucinated_apis",
        category=TestCategory.H_ANTI_PATTERNS,
        question="Read and analyze the files.",
        context_generator=lambda: generate_file_list_context(num_files=3),
        expected=ExpectedBehavior(
            forbidden_patterns=[
                r"(?<!list_)files\(\)",
                r"get_files\(\)",
                r"(?<!read_)open\(",
                r"context\.read",
            ],
        ),
        description="Detect if model uses non-existent APIs",
    ),
]


# =============================================================================
# Category M: Fake Multi-Turn Tests (simulated mid-conversation)
# =============================================================================

from rlm.utils.prompts import RLM_SYSTEM_PROMPT

# Helper to build fake multi-turn messages
def _build_mid_conversation_messages(
    question: str,
    prior_turns: list[tuple[str, str]],  # [(assistant_msg, repl_output), ...]
) -> list[dict[str, str]]:
    """Build messages simulating a mid-conversation state."""
    messages = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "assistant", "content": "Context: str, 1000 chars, 1 chunk(s)."},
        {"role": "user", "content": f"## Question\n{question}\n\n## Instructions\n1. First explore the context (list_files() or print(context))\n2. Analyze the relevant data\n3. Call FINAL(answer_literal) when done\n\nBegin by exploring:"},
    ]
    for assistant_msg, repl_output in prior_turns:
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": f"```output\n{repl_output}\n```"})
    return messages


FAKE_MULTITURN_TESTS = [
    # M1: After exploration, model should analyze or answer
    TestCase(
        id="M1",
        name="after_exploration_should_analyze",
        category=TestCategory.G_EDGE_CASES,  # Using existing category
        question="What is the secret value in the data?",
        prebuilt_messages=_build_mid_conversation_messages(
            question="What is the secret value in the data?",
            prior_turns=[
                (
                    "Let me explore the context first.\n```repl\nprint(context[:500])\n```",
                    "The data contains: name=test, value=42, secret=HIDDEN_KEY_999, status=active"
                ),
            ]
        ),
        expected=ExpectedBehavior(
            should_explore=False,  # Already explored in prior turn
            should_use_repl=True,
            should_call_final=True,  # Should recognize answer is visible and call FINAL
        ),
        description="After seeing context with answer, model should call FINAL",
        tags=["multiturn", "fake"],
    ),

    # M2: After error, model should retry differently
    TestCase(
        id="M2",
        name="after_error_should_retry",
        category=TestCategory.D_ERROR_RECOVERY,
        question="Read the config file and extract the port number.",
        prebuilt_messages=_build_mid_conversation_messages(
            question="Read the config file and extract the port number.",
            prior_turns=[
                (
                    "```repl\nread_file('config.txt')\n```",
                    "Error: FileNotFoundError: 'config.txt' not found. Available files: ['config.json', 'settings.yaml']"
                ),
            ]
        ),
        expected=ExpectedBehavior(
            should_explore=False,
            should_use_repl=True,
            should_call_final=False,  # Retry first, FINAL comes later
            expected_api_calls=["read_file"],  # Should try a different file
        ),
        description="After file not found error, model should try available files",
        tags=["multiturn", "fake", "error_recovery"],
    ),

    # M3: After getting analysis results, should synthesize and FINAL
    TestCase(
        id="M3",
        name="after_analysis_should_final",
        category=TestCategory.A_BASIC_WORKFLOW,
        question="Summarize the main themes across all documents.",
        prebuilt_messages=_build_mid_conversation_messages(
            question="Summarize the main themes across all documents.",
            prior_turns=[
                (
                    "```repl\nlist_files()\n```",
                    "['doc1.txt', 'doc2.txt', 'doc3.txt']"
                ),
                (
                    "```repl\nsummaries = llm_query_batched([f'Summarize: {read_file(f)}' for f in ['doc1.txt', 'doc2.txt', 'doc3.txt']])\nprint(summaries)\n```",
                    "['Doc1 is about machine learning and neural networks.', 'Doc2 discusses data preprocessing techniques.', 'Doc3 covers model evaluation metrics.']"
                ),
            ]
        ),
        expected=ExpectedBehavior(
            should_explore=False,  # Already explored
            should_call_final=True,  # Has all info, should synthesize
            forbidden_patterns=["llm_query.*final answer"],  # Should synthesize itself
        ),
        description="After getting summaries, model should synthesize and call FINAL",
        tags=["multiturn", "fake"],
    ),

    # M4: With partial results, should continue gathering
    TestCase(
        id="M4",
        name="partial_results_continue",
        category=TestCategory.A_BASIC_WORKFLOW,
        question="Find the total count of errors across all log files.",
        prebuilt_messages=_build_mid_conversation_messages(
            question="Find the total count of errors across all log files.",
            prior_turns=[
                (
                    "```repl\nfiles = list_files()\nprint(files)\n```",
                    "['app.log', 'error.log', 'debug.log']"
                ),
                (
                    "```repl\napp_content = read_file('app.log')\nerror_count_app = app_content.count('ERROR')\nprint(f'app.log errors: {error_count_app}')\n```",
                    "app.log errors: 15"
                ),
            ]
        ),
        expected=ExpectedBehavior(
            should_explore=False,  # Already explored
            should_use_repl=True,
            expected_api_calls=["read_file"],  # Should read remaining files
            should_call_final=False,  # Not done yet - more files to check
        ),
        description="With partial results, model should continue reading remaining files",
        tags=["multiturn", "fake"],
    ),

    # M5: After JSON parse error, should retry
    TestCase(
        id="M5",
        name="after_parse_error_retry",
        category=TestCategory.D_ERROR_RECOVERY,
        question="Extract the user IDs from the JSON data.",
        prebuilt_messages=_build_mid_conversation_messages(
            question="Extract the user IDs from the JSON data.",
            prior_turns=[
                (
                    "```repl\ndata = parse_json(context)\nprint(data)\n```",
                    "Error: JSONDecodeError: Expecting property name enclosed in double quotes at line 3"
                ),
            ]
        ),
        expected=ExpectedBehavior(
            should_explore=False,
            should_use_repl=True,
            should_call_final=False,  # Retry first, FINAL comes later
            # Should try alternative approach (regex, manual parsing, etc.)
        ),
        description="After parse error, model should try alternative extraction",
        tags=["multiturn", "fake", "error_recovery"],
    ),

    # M6: Verify model doesn't ask sub-LLM for final answer
    TestCase(
        id="M6",
        name="dont_delegate_final_answer",
        category=TestCategory.H_ANTI_PATTERNS,
        question="What is the sentiment of this review?",
        prebuilt_messages=_build_mid_conversation_messages(
            question="What is the sentiment of this review?",
            prior_turns=[
                (
                    "```repl\nprint(context)\n```",
                    "Review: The product exceeded my expectations! Great quality and fast shipping. Highly recommend to anyone looking for reliability."
                ),
            ]
        ),
        expected=ExpectedBehavior(
            should_explore=False,  # Already explored
            should_call_final=True,
            forbidden_patterns=[
                r"llm_query.*final answer",
                r"llm_query.*sentiment.*review",
                r"llm_query.*what is the sentiment",
            ],
        ),
        description="Model should determine sentiment itself, not delegate to sub-LLM",
        tags=["multiturn", "fake", "anti_pattern"],
    ),
]


# =============================================================================
# Collected Test Cases
# =============================================================================

ALL_TEST_CASES: list[TestCase] = (
    BASIC_WORKFLOW_TESTS +
    API_USAGE_TESTS +
    CONTEXT_HANDLING_TESTS +
    ERROR_RECOVERY_TESTS +
    SUB_LLM_TESTS +
    STORE_OPERATION_TESTS +
    EDGE_CASE_TESTS +
    ANTI_PATTERN_TESTS +
    FAKE_MULTITURN_TESTS
)


def get_test_by_id(test_id: str) -> TestCase | None:
    """Get a test case by its ID."""
    for test in ALL_TEST_CASES:
        if test.id == test_id:
            return test
    return None


def get_tests_by_category(category: TestCategory) -> list[TestCase]:
    """Get all test cases in a category."""
    return [t for t in ALL_TEST_CASES if t.category == category]


def get_tests_by_tag(tag: str) -> list[TestCase]:
    """Get all test cases with a specific tag."""
    return [t for t in ALL_TEST_CASES if tag in t.tags]


# Quick stats
TEST_COUNTS = {
    cat.name: len(get_tests_by_category(cat))
    for cat in TestCategory
}
