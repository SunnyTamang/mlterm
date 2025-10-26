"""
Core Tracker class for logging ML experiment metrics.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from filelock import FileLock


class Tracker:
    """
    A lightweight experiment tracker that logs metrics to JSONL files.
    
    This tracker is designed to be simple, fast, and compatible with
    terminal-based monitoring tools.
    """
    
    def __init__(
        self,
        project: str,
        run_id: Optional[str] = None,
        log_dir: Union[str, Path] = "./logs",
        auto_timestamp: bool = True
    ):
        """
        Initialize the tracker.
        
        Args:
            project: Name of the project/experiment
            run_id: Unique identifier for this run (auto-generated if None)
            log_dir: Directory to store log files
            auto_timestamp: Whether to automatically add timestamps to logs
        """
        self.project = project
        self.run_id = run_id or f"run_{int(time.time())}"
        self.log_dir = Path(log_dir)
        self.auto_timestamp = auto_timestamp
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up log file path
        self.log_file = self.log_dir / f"{project}_{self.run_id}.jsonl"
        
        # Initialize run metadata
        self._log_metadata()
    
    def _log_metadata(self):
        """Log initial metadata about the run."""
        metadata = {
            "type": "metadata",
            "project": self.project,
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "timestamp": time.time()
        }
        self._write_log_entry(metadata)
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the JSONL file with file locking."""
        with FileLock(f"{self.log_file}.lock"):
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    
    def log(self, **metrics: Any) -> None:
        """
        Log metrics for the current step.
        
        Args:
            **metrics: Key-value pairs of metrics to log
        """
        entry = {
            "type": "metric",
            "run_id": self.run_id,
            "project": self.project,
            **metrics
        }
        
        if self.auto_timestamp:
            entry["timestamp"] = time.time()
            entry["datetime"] = datetime.now().isoformat()
        
        self._write_log_entry(entry)
    
    def log_hyperparameters(self, **hyperparams: Any) -> None:
        """
        Log hyperparameters for this run.
        
        Args:
            **hyperparams: Key-value pairs of hyperparameters
        """
        entry = {
            "type": "hyperparameters",
            "run_id": self.run_id,
            "project": self.project,
            "timestamp": time.time(),
            "hyperparameters": hyperparams
        }
        self._write_log_entry(entry)
    
    def log_artifact(self, name: str, path: str, description: Optional[str] = None) -> None:
        """
        Log an artifact (file) associated with this run.
        
        Args:
            name: Name of the artifact
            path: Path to the artifact file
            description: Optional description of the artifact
        """
        entry = {
            "type": "artifact",
            "run_id": self.run_id,
            "project": self.project,
            "timestamp": time.time(),
            "artifact": {
                "name": name,
                "path": str(Path(path).absolute()),
                "description": description
            }
        }
        self._write_log_entry(entry)
    
    def finish(self, status: str = "completed") -> None:
        """
        Mark the run as finished.
        
        Args:
            status: Status of the run (completed, failed, etc.)
        """
        entry = {
            "type": "finish",
            "run_id": self.run_id,
            "project": self.project,
            "timestamp": time.time(),
            "status": status,
            "end_time": datetime.now().isoformat()
        }
        self._write_log_entry(entry)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.finish("failed")
        else:
            self.finish("completed")
