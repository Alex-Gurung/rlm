"""
Commit protocol for structured sub-LLM outputs.

Workers return JSON commits that are deterministically merged into the global store.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlm.core.store import Store


@dataclass
class CommitCreate:
    """A create operation in a commit."""

    type: str
    id: str  # local or namespaced
    description: str
    content: Any
    backrefs: list[dict] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "CommitCreate":
        return cls(
            type=d.get("type", "note"),
            id=d.get("id", d.get("local_id", "")),
            description=d.get("description", ""),
            content=d.get("content", ""),
            backrefs=d.get("backrefs", []),
            parents=d.get("parents", []),
            tags=d.get("tags", []),
        )


@dataclass
class CommitLink:
    """A link between two objects in a commit."""

    type: str  # supports|contradicts|refines
    src: str
    dst: str

    @classmethod
    def from_dict(cls, d: dict) -> "CommitLink":
        return cls(
            type=d.get("type", d.get("link_type", "supports")),
            src=d.get("src", ""),
            dst=d.get("dst", ""),
        )


@dataclass
class CommitUpdate:
    """A proposed update to an existing object."""

    target_id: str
    patch: dict
    description_update: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "CommitUpdate":
        return cls(
            target_id=d.get("target_id", ""),
            patch=d.get("patch", {}),
            description_update=d.get("description_update"),
        )


@dataclass
class Commit:
    """A structured commit from a sub-LLM worker."""

    commit_id: str
    creates: list[CommitCreate] = field(default_factory=list)
    links: list[CommitLink] = field(default_factory=list)
    proposes_updates: list[CommitUpdate] = field(default_factory=list)
    error: str | None = None  # Set if commit parsing failed

    @classmethod
    def from_dict(cls, d: dict) -> "Commit":
        return cls(
            commit_id=d.get("commit_id", "unknown"),
            creates=[CommitCreate.from_dict(c) for c in d.get("creates", [])],
            links=[CommitLink.from_dict(lk) for lk in d.get("links", [])],
            proposes_updates=[CommitUpdate.from_dict(u) for u in d.get("proposes_updates", [])],
            error=d.get("error"),
        )

    def to_dict(self) -> dict:
        return {
            "commit_id": self.commit_id,
            "creates": [
                {
                    "type": c.type,
                    "id": c.id,
                    "description": c.description,
                    "content": c.content,
                    "backrefs": c.backrefs,
                    "parents": c.parents,
                    "tags": c.tags,
                }
                for c in self.creates
            ],
            "links": [{"type": lk.type, "src": lk.src, "dst": lk.dst} for lk in self.links],
            "proposes_updates": [
                {
                    "target_id": u.target_id,
                    "patch": u.patch,
                    "description_update": u.description_update,
                }
                for u in self.proposes_updates
            ],
            "error": self.error,
        }


@dataclass
class MergeResult:
    """Result of merging a commit into the store."""

    commit_id: str
    created_ids: dict[str, str]  # local_id -> global_id
    links_created: int
    proposals_stored: int
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "commit_id": self.commit_id,
            "created_ids": self.created_ids,
            "links_created": self.links_created,
            "proposals_stored": self.proposals_stored,
            "errors": self.errors,
            "success": self.success,
        }


def parse_commit(text: str, fallback_id: str = "unknown") -> Commit:
    """
    Parse commit from text. Handles:
    - Pure JSON
    - Fenced JSON (```json ... ```)
    - JSON embedded in text (looks for first { ... } block)

    Args:
        text: The text to parse
        fallback_id: Default commit_id if not found

    Returns:
        Parsed Commit object, or Commit with error set if parsing fails
    """
    text = text.strip()

    # Try 1: Pure JSON
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            if "commit_id" not in d:
                d["commit_id"] = fallback_id
            return Commit.from_dict(d)
    except json.JSONDecodeError:
        pass

    # Try 2: Fenced JSON (```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        try:
            d = json.loads(fence_match.group(1).strip())
            if isinstance(d, dict):
                if "commit_id" not in d:
                    d["commit_id"] = fallback_id
                return Commit.from_dict(d)
        except json.JSONDecodeError:
            pass

    # Try 3: Python dict literal (e.g., from FINAL_VAR on dict)
    if text.startswith("{") and "'" in text:
        try:
            d = ast.literal_eval(text)
            if isinstance(d, dict):
                if "commit_id" not in d:
                    d["commit_id"] = fallback_id
                return Commit.from_dict(d)
        except (ValueError, SyntaxError):
            pass

    # Try 4: Find first JSON object in text
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[brace_start:], start=brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_str = text[brace_start : i + 1]
                    try:
                        d = json.loads(json_str)
                        if isinstance(d, dict):
                            if "commit_id" not in d:
                                d["commit_id"] = fallback_id
                            return Commit.from_dict(d)
                    except json.JSONDecodeError:
                        pass
                    break

    # Failed to parse
    return Commit(
        commit_id=fallback_id,
        error=f"Failed to parse commit JSON from text (length={len(text)})",
    )


def apply_commit(
    store: "Store",
    commit: Commit,
    batch_prefix: str = "",
) -> MergeResult:
    """
    Merge a worker commit into the store with namespacing.

    Args:
        store: The Store instance to merge into
        commit: The Commit to apply
        batch_prefix: Optional prefix for namespacing (e.g., batch_id)

    Returns:
        MergeResult with created IDs and any errors
    """
    errors: list[str] = []
    created_ids: dict[str, str] = {}

    # Check for commit-level error
    if commit.error:
        return MergeResult(
            commit_id=commit.commit_id,
            created_ids={},
            links_created=0,
            proposals_stored=0,
            errors=[commit.error],
        )

    # Build namespace prefix
    prefix = f"{batch_prefix}/{commit.commit_id}" if batch_prefix else commit.commit_id

    # 1. Process creates
    for create in commit.creates:
        local_id = create.id
        # Namespace the ID if not already namespaced
        if "/" not in local_id:
            global_id = f"{prefix}/{local_id}"
        else:
            global_id = local_id

        try:
            # Resolve parent references (may be local IDs)
            resolved_parents = []
            for parent_ref in create.parents:
                if parent_ref.startswith("BATCH:"):
                    # Special batch reference - skip for now, will be handled at batch level
                    continue
                elif parent_ref in created_ids:
                    resolved_parents.append(created_ids[parent_ref])
                elif store.get(parent_ref) is not None:
                    resolved_parents.append(parent_ref)
                # Otherwise skip unknown parent

            # Create the object in store
            obj_id = store.create(
                type=create.type,
                description=create.description,
                content=create.content,
                backrefs=create.backrefs,
                parents=resolved_parents,
                tags=create.tags,
            )

            # Map local ID to actual store ID
            created_ids[local_id] = obj_id
            created_ids[global_id] = obj_id

        except Exception as e:
            errors.append(f"Create {local_id}: {e}")

    # 2. Process links
    links_created = 0
    for link in commit.links:
        # Resolve src and dst (may be local IDs)
        src = created_ids.get(link.src, link.src)
        dst = created_ids.get(link.dst, link.dst)

        # Create a link object in the store
        try:
            store.create(
                type="link",
                description=f"{link.type}: {src} -> {dst}",
                content={"link_type": link.type, "src": src, "dst": dst},
                tags=[f"link_type:{link.type}"],
            )
            links_created += 1
        except Exception as e:
            errors.append(f"Link {link.src}->{link.dst}: {e}")

    # 3. Process proposed updates (store as proposals, don't mutate in v1)
    proposals_stored = 0
    for update in commit.proposes_updates:
        try:
            store.create(
                type="update_proposal",
                description=f"Proposal for {update.target_id}",
                content={
                    "target_id": update.target_id,
                    "patch": update.patch,
                    "description_update": update.description_update,
                },
                tags=["proposal"],
            )
            proposals_stored += 1
        except Exception as e:
            errors.append(f"Proposal for {update.target_id}: {e}")

    return MergeResult(
        commit_id=commit.commit_id,
        created_ids=created_ids,
        links_created=links_created,
        proposals_stored=proposals_stored,
        errors=errors,
    )
