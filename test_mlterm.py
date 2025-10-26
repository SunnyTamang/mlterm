#!/usr/bin/env python3
"""
Simple test script to verify mlterm functionality.
"""

import time
import tempfile
from pathlib import Path
from mlterm import Tracker, SystemMonitor


def test_tracker():
    """Test basic tracker functionality."""
    print("Testing Tracker...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = Tracker(
            project="test_project",
            run_id="test_run",
            log_dir=temp_dir
        )
        
        # Test logging
        tracker.log_hyperparameters(
            learning_rate=0.001,
            batch_size=32,
            epochs=10
        )
        
        for i in range(5):
            tracker.log(
                epoch=i,
                loss=1.0 - i * 0.1,
                accuracy=0.5 + i * 0.1
            )
            time.sleep(0.1)
        
        tracker.finish("completed")
        
        # Verify log file exists and has content
        log_file = Path(temp_dir) / "test_project_test_run.jsonl"
        assert log_file.exists(), "Log file should exist"
        
        with open(log_file) as f:
            lines = f.readlines()
        
        assert len(lines) >= 7, f"Expected at least 7 lines, got {len(lines)}"
        print("Tracker test passed!")


def test_system_monitor():
    """Test system monitoring functionality."""
    print("Testing SystemMonitor...")
    
    monitor = SystemMonitor()
    
    # Test basic system info
    system_info = monitor.get_system_info()
    assert "cpu" in system_info
    assert "memory" in system_info
    assert "gpu" in system_info
    
    # Test individual components
    cpu_info = monitor.get_cpu_info()
    assert "cpu_percent" in cpu_info
    
    memory_info = monitor.get_memory_info()
    assert "memory_total" in memory_info
    assert "memory_used" in memory_info
    
    print("SystemMonitor test passed!")


def main():
    """Run all tests."""
    print("Running mlterm tests...\n")
    
    try:
        test_tracker()
        test_system_monitor()
        print("\nAll tests passed!")
        return True
    except Exception as e:
        print(f"\nTest failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
