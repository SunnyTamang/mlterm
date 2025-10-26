"""
Test configuration and utilities.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_tracker(temp_log_dir):
    """Create a sample tracker for testing."""
    from mlterm import Tracker
    return Tracker(
        project="test_project",
        run_id="test_run",
        log_dir=temp_log_dir
    )
