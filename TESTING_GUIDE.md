# 🧪 mlterm Testing Guide

This guide shows you how to run and test all aspects of mlterm.

## 🚀 **Quick Start Testing**

### 1. **Install and Verify**
```bash
# Install mlterm
pip install -e .

# Run basic tests
python test_mlterm.py
```

### 2. **Test CLI Commands**
```bash
# Check CLI is working
mlterm --help

# List available runs (after running examples)
mlterm list-runs

# Get detailed info about a run
mlterm info logs/example_classification_demo_run_001.jsonl
```

## 🎯 **Complete Testing Workflow**

### **Step 1: Run Training Simulation**

**Terminal 1 - Start Training:**
```bash
# Basic example
python examples/simple_training.py

# OR Advanced example (multiple experiments)
python examples/advanced_training.py
```

**What to expect:**
- Training metrics will be logged to `logs/` directory
- You'll see progress output in the terminal
- JSONL files will be created for each run

### **Step 2: Monitor with Dashboard**

**Terminal 2 - Launch Dashboard:**
```bash
# Monitor specific project
mlterm dashboard --project example_classification

# OR monitor all projects
mlterm dashboard

# OR monitor specific run
mlterm dashboard --project hyperparameter_sweep --run-id sweep_01
```

**Dashboard Features to Test:**
- **Overview Tab**: Real-time metrics and system stats
- **System Tab**: CPU, GPU, memory monitoring
- **Logs Tab**: Live log entries
- **Navigation**: Use Tab to switch between tabs
- **Controls**: Press 'q' to quit, 'r' to refresh

### **Step 3: Test CLI Commands**

```bash
# List all available runs
mlterm list-runs

# Get detailed information about a run
mlterm info logs/example_classification_demo_run_001.jsonl

# Initialize a new experiment
mlterm init my_new_project --run-id test_001
```

## 🔍 **Testing Different Scenarios**

### **Test 1: Basic Integration**
```python
# Create test_script.py
from mlterm import Tracker

tracker = Tracker(project="test_project", run_id="test_001")
tracker.log_hyperparameters(learning_rate=0.001, batch_size=32)

for i in range(10):
    tracker.log(epoch=i, loss=1.0 - i*0.1, accuracy=0.5 + i*0.05)
    time.sleep(0.1)

tracker.finish("completed")
```

### **Test 2: Context Manager**
```python
# Create test_context.py
from mlterm import Tracker

with Tracker(project="context_test") as tracker:
    tracker.log_hyperparameters(epochs=5)
    
    for i in range(5):
        tracker.log(epoch=i, loss=1.0 - i*0.2)
        time.sleep(0.1)
    # Automatically calls tracker.finish() on exit
```

### **Test 3: Multiple Experiments**
```python
# Create test_multi.py
from mlterm import Tracker

configs = [
    {"learning_rate": 0.001, "optimizer": "Adam"},
    {"learning_rate": 0.01, "optimizer": "SGD"},
]

for i, config in enumerate(configs):
    with Tracker(project="multi_test", run_id=f"run_{i}") as tracker:
        tracker.log_hyperparameters(**config)
        
        for epoch in range(5):
            tracker.log(epoch=epoch, loss=1.0 - epoch*0.1)
            time.sleep(0.05)
```

## 📊 **Dashboard Testing Checklist**

### **Overview Tab**
- [ ] Metrics table shows real-time data
- [ ] System stats update every 2 seconds
- [ ] Values are formatted correctly
- [ ] Timestamps are accurate

### **System Tab**
- [ ] CPU usage displays correctly
- [ ] Memory usage shows in GB
- [ ] GPU info appears (if available)
- [ ] Network stats are shown

### **Logs Tab**
- [ ] Log entries appear in real-time
- [ ] Different entry types are color-coded
- [ ] Timestamps are displayed
- [ ] Scroll works properly

### **Navigation**
- [ ] Tab switching works with Tab key
- [ ] 'q' quits the dashboard
- [ ] 'r' refreshes manually
- [ ] No crashes or errors

## 🐛 **Troubleshooting Tests**

### **Test Error Handling**
```python
# Test with invalid data
tracker = Tracker(project="error_test")
tracker.log(epoch="invalid", loss="not_a_number")  # Should handle gracefully
```

### **Test File Locking**
```python
# Test concurrent access
import threading

def log_data(tracker, start, end):
    for i in range(start, end):
        tracker.log(epoch=i, loss=1.0 - i*0.01)

tracker1 = Tracker(project="concurrent_test", run_id="run_1")
tracker2 = Tracker(project="concurrent_test", run_id="run_2")

# Should not corrupt log files
thread1 = threading.Thread(target=log_data, args=(tracker1, 0, 5))
thread2 = threading.Thread(target=log_data, args=(tracker2, 5, 10))
```

## 🎯 **Performance Testing**

### **Test Large Log Files**
```python
# Create a large log file
tracker = Tracker(project="large_test", run_id="big_run")

for epoch in range(1000):
    tracker.log(
        epoch=epoch,
        loss=1.0 - epoch*0.001,
        accuracy=0.5 + epoch*0.0005,
        gpu_util=random.uniform(50, 100),
        memory=random.uniform(2, 8)
    )
```

### **Test Dashboard Performance**
- Launch dashboard with large log file
- Verify smooth scrolling
- Check memory usage doesn't grow excessively
- Test refresh rate with large datasets

## ✅ **Success Criteria**

**All tests should pass:**
1. ✅ Basic tracker functionality
2. ✅ Dashboard launches and displays data
3. ✅ CLI commands work correctly
4. ✅ System monitoring shows real stats
5. ✅ Multiple experiments can be tracked
6. ✅ File locking prevents corruption
7. ✅ Context manager works properly
8. ✅ Error handling is graceful

## 🚀 **Quick Test Commands**

```bash
# Run all basic tests
python test_mlterm.py

# Test training simulation
python examples/simple_training.py &
mlterm dashboard --project example_classification

# Test multiple experiments
python examples/advanced_training.py &
mlterm list-runs
mlterm info logs/hyperparameter_sweep_sweep_01.jsonl

# Test CLI help
mlterm --help
mlterm dashboard --help
mlterm list-runs --help
```

## 🎉 **Expected Results**

After running all tests, you should have:
- ✅ Working tracker with JSONL logging
- ✅ Real-time dashboard with live updates
- ✅ System monitoring (CPU, GPU, memory)
- ✅ CLI interface with rich output
- ✅ Multiple experiment support
- ✅ Proper error handling
- ✅ File locking and concurrency safety

**mlterm is ready for production use!** 🧠✨
