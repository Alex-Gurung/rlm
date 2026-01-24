"""Generate synthetic contexts for prompt testing."""

import json
import random
import string
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratedContext:
    """A generated context with metadata."""
    content: str | dict | list
    context_type: str
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        if isinstance(self.content, str):
            return len(self.content)
        return len(json.dumps(self.content))


def generate_simple_text(
    length: int = 500,
    topic: str = "technology",
) -> GeneratedContext:
    """Generate simple text context with a findable fact."""
    facts = {
        "technology": ("Python was created by Guido van Rossum in 1991.", "1991"),
        "science": ("Water boils at 100 degrees Celsius at sea level.", "100"),
        "history": ("The Great Wall of China was built over many centuries.", "centuries"),
        "geography": ("Mount Everest is 8,849 meters tall.", "8,849"),
    }

    fact, answer = facts.get(topic, facts["technology"])

    # Generate padding text
    padding_words = [
        "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
        "data", "analysis", "system", "process", "method", "approach",
    ]

    padding = " ".join(random.choices(padding_words, k=length // 6))

    # Insert fact at random position
    insert_pos = random.randint(len(padding) // 3, 2 * len(padding) // 3)
    content = padding[:insert_pos] + " " + fact + " " + padding[insert_pos:]

    return GeneratedContext(
        content=content[:length],
        context_type="string",
        expected_answer=answer,
        metadata={"topic": topic, "fact": fact},
    )


def generate_json_context(
    num_items: int = 10,
    include_target: bool = True,
) -> GeneratedContext:
    """Generate JSON context with structured data."""
    items = []

    for i in range(num_items):
        item = {
            "id": f"item_{i}",
            "name": f"Item {i}",
            "value": random.randint(1, 100),
            "category": random.choice(["A", "B", "C"]),
            "active": random.choice([True, False]),
        }
        items.append(item)

    # Add a target item to find
    target_value = None
    if include_target:
        target_value = 999
        items[random.randint(0, len(items) - 1)]["value"] = target_value
        items[random.randint(0, len(items) - 1)]["special"] = "target_marker"

    return GeneratedContext(
        content=items,
        context_type="json",
        expected_answer=str(target_value) if target_value else None,
        metadata={"num_items": num_items},
    )


def generate_file_list_context(
    num_files: int = 5,
    include_target_file: bool = True,
) -> GeneratedContext:
    """Generate a context simulating multiple files."""
    files = {}

    file_templates = [
        ("config.json", '{"setting": "value", "debug": false}'),
        ("README.md", "# Project\n\nThis is a project readme."),
        ("data.csv", "id,name,value\n1,foo,100\n2,bar,200"),
        ("script.py", "def main():\n    print('Hello')\n"),
        ("notes.txt", "Important notes:\n- Item 1\n- Item 2"),
        ("settings.yaml", "database:\n  host: localhost\n  port: 5432"),
        ("log.txt", "[INFO] Started\n[DEBUG] Processing\n[INFO] Done"),
    ]

    # Select files
    selected = random.sample(file_templates, min(num_files, len(file_templates)))

    for name, content in selected:
        files[name] = content

    # Add target file with answer
    target_answer = None
    if include_target_file:
        target_answer = "SECRET_VALUE_42"
        files["secret.txt"] = f"The answer is: {target_answer}"

    # Create file summary as string
    file_summary = f"Files available ({len(files)}):\n"
    for name in files:
        file_summary += f"  - {name}\n"

    return GeneratedContext(
        content=file_summary,
        context_type="files",
        expected_answer=target_answer,
        metadata={"files": files, "num_files": len(files)},
    )


def generate_large_text_context(
    size_kb: int = 64,
    num_sections: int = 10,
) -> GeneratedContext:
    """Generate a large text context that requires chunking."""
    target_size = size_kb * 1024
    section_size = target_size // num_sections

    sections = []
    target_section = random.randint(0, num_sections - 1)
    target_answer = "HIDDEN_FACT_12345"

    for i in range(num_sections):
        section_header = f"\n\n=== Section {i + 1} ===\n\n"

        # Generate filler content
        words = ["data", "analysis", "process", "system", "method",
                 "result", "finding", "observation", "note", "detail"]
        filler = " ".join(random.choices(words, k=section_size // 7))

        if i == target_section:
            # Insert the target fact
            insert_pos = len(filler) // 2
            filler = filler[:insert_pos] + f" The secret code is {target_answer}. " + filler[insert_pos:]

        sections.append(section_header + filler)

    content = "".join(sections)[:target_size]

    return GeneratedContext(
        content=content,
        context_type="large_text",
        expected_answer=target_answer,
        metadata={
            "size_kb": size_kb,
            "num_sections": num_sections,
            "target_section": target_section,
        },
    )


def generate_multi_file_context(
    num_files: int = 10,
    files_with_answer: int = 1,
) -> GeneratedContext:
    """Generate multi-file context where answer spans multiple files."""
    files = {}

    # Generate decoy files
    decoy_templates = [
        "This file contains general information.",
        "Configuration data: debug=false, verbose=true",
        "Log entries from the system...",
        "Documentation for the module...",
        "Test data for validation...",
    ]

    for i in range(num_files - files_with_answer):
        name = f"file_{i}.txt"
        content = random.choice(decoy_templates) + f"\nFile number: {i}\n"
        content += "".join(random.choices(string.ascii_lowercase + " ", k=200))
        files[name] = content

    # Add answer files
    answer_parts = ["Part A: 42", "Part B: Blue", "Part C: North"]
    full_answer = "42, Blue, North"

    for i in range(files_with_answer):
        name = f"answer_file_{i}.txt"
        files[name] = f"Critical information:\n{answer_parts[i % len(answer_parts)]}\n"

    # File summary
    file_list = "\n".join(f"  - {name}" for name in sorted(files.keys()))
    summary = f"Context contains {len(files)} files:\n{file_list}"

    return GeneratedContext(
        content=summary,
        context_type="multi_file",
        expected_answer=full_answer if files_with_answer > 1 else answer_parts[0].split(": ")[1],
        metadata={"files": files, "num_files": num_files},
    )


def generate_nested_json_context(
    depth: int = 3,
    width: int = 3,
) -> GeneratedContext:
    """Generate deeply nested JSON context."""

    def generate_level(current_depth: int) -> dict:
        if current_depth >= depth:
            return {"value": random.randint(1, 100), "leaf": True}

        return {
            f"node_{i}": generate_level(current_depth + 1)
            for i in range(width)
        }

    data = generate_level(0)

    # Add target value at a random path
    target_value = "NESTED_ANSWER_789"

    def set_target(obj: dict, target_depth: int) -> bool:
        if target_depth == 0:
            obj["secret_key"] = target_value
            return True

        for key in list(obj.keys()):
            if isinstance(obj[key], dict):
                if set_target(obj[key], target_depth - 1):
                    return True
        return False

    set_target(data, random.randint(1, depth - 1))

    return GeneratedContext(
        content=data,
        context_type="nested_json",
        expected_answer=target_value,
        metadata={"depth": depth, "width": width},
    )


class ContextGenerator:
    """Factory for generating different types of test contexts."""

    GENERATORS = {
        "simple_text": generate_simple_text,
        "json": generate_json_context,
        "file_list": generate_file_list_context,
        "large_text": generate_large_text_context,
        "multi_file": generate_multi_file_context,
        "nested_json": generate_nested_json_context,
    }

    @classmethod
    def generate(cls, context_type: str, **kwargs) -> GeneratedContext:
        """Generate a context of the specified type."""
        if context_type not in cls.GENERATORS:
            raise ValueError(f"Unknown context type: {context_type}")

        return cls.GENERATORS[context_type](**kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        """List available context types."""
        return list(cls.GENERATORS.keys())
