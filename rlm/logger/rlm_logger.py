"""
Logger for RLM iterations.

Writes RLMIteration data to JSON-lines files for analysis and debugging.
Supports hierarchical logging for nested worker runs.
"""

import json
import os
import uuid
from datetime import datetime

from rlm.core.types import RLMIteration, RLMMetadata


class RLMLogger:
    """Logger that writes RLMIteration data to a JSON-lines file.

    Supports hierarchical runs where nested workers log to the same file
    with parent-child relationships tracked via run_id/parent_run_id.
    """

    def __init__(
        self,
        log_dir: str,
        file_name: str = "rlm",
        run_id: str | None = None,
        parent_run_id: str | None = None,
        depth: int = 0,
        log_file_path: str | None = None,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory to write logs to
            file_name: Base name for log file (ignored if log_file_path provided)
            run_id: Unique ID for this run (auto-generated if not provided)
            parent_run_id: ID of parent run (for nested workers)
            depth: Depth level in hierarchy (0 = root)
            log_file_path: Explicit path to log file (for sharing between parent/child)
        """
        self.log_dir = log_dir
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.parent_run_id = parent_run_id
        self.depth = depth

        os.makedirs(log_dir, exist_ok=True)

        if log_file_path:
            # Use shared log file from parent
            self.log_file_path = log_file_path
        else:
            # Create new log file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{self.run_id}.jsonl")

        self._iteration_count = 0
        self._metadata_logged = False

    def create_child_logger(self, child_depth: int) -> "RLMLogger":
        """Create a logger for a child worker that writes to the same file.

        Args:
            child_depth: Depth of the child worker

        Returns:
            New RLMLogger instance for the child
        """
        return RLMLogger(
            log_dir=self.log_dir,
            run_id=str(uuid.uuid4())[:8],  # New run_id for child
            parent_run_id=self.run_id,  # This run is the parent
            depth=child_depth,
            log_file_path=self.log_file_path,  # Share the same file
        )

    def log_metadata(self, metadata: RLMMetadata):
        """Log RLM metadata as the first entry in the file."""
        if self._metadata_logged:
            return

        entry = {
            "type": "metadata",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "depth": self.depth,
            **metadata.to_dict(),
        }

        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        self._metadata_logged = True

    def log(self, iteration: RLMIteration):
        """Log an RLMIteration to the file."""
        self._iteration_count += 1

        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "depth": self.depth,
            **iteration.to_dict(),
        }

        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def log_worker_spawn(self, child_run_id: str, worker_prompt: str):
        """Log when a worker is spawned."""
        entry = {
            "type": "worker_spawn",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "child_run_id": child_run_id,
            "depth": self.depth,
            "worker_prompt_preview": worker_prompt[:500] if worker_prompt else None,
        }

        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def log_worker_complete(self, child_run_id: str, result_summary: str | None = None):
        """Log when a worker completes."""
        entry = {
            "type": "worker_complete",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "child_run_id": child_run_id,
            "depth": self.depth,
            "result_summary": result_summary,
        }

        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
