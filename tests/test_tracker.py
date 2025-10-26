"""
Basic tests for mlterm package.
"""

import pytest
import tempfile
import os
from pathlib import Path
from mlterm import Tracker


class TestTracker:
    """Test cases for the Tracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = Tracker(
                project="test_project",
                run_id="test_run",
                log_dir=temp_dir
            )
            assert tracker.project == "test_project"
            assert tracker.run_id == "test_run"
            assert tracker.log_dir == Path(temp_dir)
    
    def test_log_hyperparameters(self):
        """Test logging hyperparameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = Tracker(
                project="test_project",
                run_id="test_run",
                log_dir=temp_dir
            )
            
            tracker.log_hyperparameters(
                learning_rate=0.001,
                batch_size=32,
                epochs=100
            )
            
            # Check that log file was created
            log_file = tracker.log_file
            assert log_file.exists()
            
            # Check log file content
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2  # metadata + hyperparameters
    
    def test_log_metrics(self):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = Tracker(
                project="test_project",
                run_id="test_run",
                log_dir=temp_dir
            )
            
            tracker.log(
                epoch=1,
                loss=0.5,
                accuracy=0.9
            )
            
            # Check that log file was created
            log_file = tracker.log_file
            assert log_file.exists()
            
            # Check log file content
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2  # metadata + metrics
    
    def test_log_artifact(self):
        """Test logging artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = Tracker(
                project="test_project",
                run_id="test_run",
                log_dir=temp_dir
            )
            
            # Create a dummy artifact file
            artifact_path = Path(temp_dir) / "test_model.pkl"
            artifact_path.write_text("dummy model content")
            
            tracker.log_artifact(
                name="test_model",
                path=str(artifact_path),
                description="Test model artifact"
            )
            
            # Check that log file was created
            log_file = tracker.log_file
            assert log_file.exists()
    
    def test_finish_run(self):
        """Test finishing a run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = Tracker(
                project="test_project",
                run_id="test_run",
                log_dir=temp_dir
            )
            
            tracker.finish("completed")
            
            # Check that log file was created
            log_file = tracker.log_file
            assert log_file.exists()
            
            # Check log file content
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2  # metadata + finish
    
    def test_context_manager(self):
        """Test using tracker as context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with Tracker(
                project="test_project",
                run_id="test_run",
                log_dir=temp_dir
            ) as tracker:
                tracker.log_hyperparameters(learning_rate=0.001)
                tracker.log(epoch=1, loss=0.5)
            
            # Check that log file was created
            log_file = tracker.log_file
            assert log_file.exists()
            
            # Check log file content
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 3  # metadata + hyperparameters + metrics + finish


if __name__ == "__main__":
    pytest.main([__file__])
