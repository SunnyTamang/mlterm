# mlterm Tracker - Complete Guide

## Overview

`mlterm` is a terminal-based machine learning experiment tracker that provides real-time monitoring of your ML training processes. It's designed to be lightweight, offline-first, and perfect for SSH-friendly environments.

## How the Tracker Works

### Core Architecture

The mlterm tracker operates on a simple but powerful principle:

1. **Logging**: Your training script logs metrics, hyperparameters, and artifacts to JSONL files
2. **Monitoring**: The dashboard reads these files in real-time and displays updates
3. **Persistence**: All data is stored locally in structured JSONL format

### Key Components

```
mlterm/
├── tracker.py          # Core Tracker class
├── dashboard.py        # Textual-based TUI dashboard
├── system_monitor.py   # System resource monitoring
└── cli.py             # Command-line interface
```

## Tracker Class Functions

### 1. Initialization

```python
from mlterm import Tracker

tracker = Tracker(
    project="my_project",      # Project name
    run_id="run_001",         # Unique run identifier
    log_dir="./logs"          # Directory for log files
)
```

**What it does:**
- Creates a unique log file: `logs/my_project_run_001.jsonl`
- Initializes the tracking session
- Sets up file locking to prevent conflicts

### 2. Logging Hyperparameters

```python
tracker.log_hyperparameters(
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    model_type="RandomForest",
    optimizer="Adam"
)
```

**What it does:**
- Logs all hyperparameters as a single JSON entry
- Timestamps the entry automatically
- Stores configuration for reproducibility

### 3. Logging Metrics

```python
tracker.log(
    epoch=10,
    train_loss=0.45,
    test_loss=0.52,
    accuracy=0.89,
    learning_rate=0.0008
)
```

**What it does:**
- Logs metrics as they're computed during training
- Each call creates a new JSON entry
- Supports any number of metric names and values
- Automatically timestamps each entry

### 4. Logging Artifacts

```python
tracker.log_artifact(
    name="best_model",
    path="model.pkl",
    description="Best performing model"
)
```

**What it does:**
- Records important files (models, data, plots)
- Stores file paths and descriptions
- Enables artifact tracking and versioning

### 5. Finishing a Run

```python
tracker.finish("completed")  # or "failed", "cancelled"
```

**What it does:**
- Marks the run as finished
- Records the final status
- Closes the log file properly

## Mandatory Code Components

### 1. Basic Setup (Required)

```python
from mlterm import Tracker

# Initialize tracker
tracker = Tracker(
    project="your_project_name",
    run_id="unique_run_id",
    log_dir="./logs"
)

# Log hyperparameters
tracker.log_hyperparameters(
    # Your hyperparameters here
    learning_rate=0.001,
    batch_size=32
)

# Your training code here...

# Finish the run
tracker.finish("completed")
```

### 2. Metric Logging (Required for Monitoring)

```python
# Inside your training loop
for epoch in range(num_epochs):
    # Training code...
    
    # Log metrics
    tracker.log(
        epoch=epoch,
        train_loss=train_loss,
        test_loss=test_loss,
        accuracy=accuracy
    )
```

### 3. Complete Example

```python
import time
from mlterm import Tracker

def train_model():
    # Initialize tracker
    tracker = Tracker(
        project="my_ml_project",
        run_id="experiment_001",
        log_dir="./logs"
    )
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        model_type="RandomForest",
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Training loop
    for epoch in range(10):
        # Simulate training
        train_loss = 1.0 - epoch * 0.1
        test_loss = train_loss + 0.05
        
        # Log metrics
        tracker.log(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            accuracy=0.5 + epoch * 0.05
        )
        
        time.sleep(0.5)  # Simulate work
    
    # Log final results
    tracker.log(
        final_train_loss=0.1,
        final_test_loss=0.15,
        final_accuracy=0.95
    )
    
    # Finish
    tracker.finish("completed")

if __name__ == "__main__":
    train_model()
```

## Dashboard Monitoring

### Starting the Dashboard

```bash
# Monitor a specific project
mlterm dashboard --project my_ml_project

# Monitor with custom refresh rate
mlterm dashboard --project my_ml_project --refresh 1.0
```

### What the Dashboard Shows

1. **System Statistics**
   - CPU usage and core count
   - Memory usage (system and process)
   - GPU status (if available)
   - Disk and network usage

2. **Training Metrics**
   - Latest logged metrics
   - Real-time updates
   - Timestamps for each metric

3. **Log Viewer**
   - Raw log entries
   - JSON structure
   - Historical data

## Data Storage Format

### JSONL Structure

Each log entry is a JSON object on a single line:

```json
{"timestamp": "2024-01-15T10:30:45.123456", "type": "hyperparameters", "learning_rate": 0.001, "batch_size": 32}
{"timestamp": "2024-01-15T10:30:45.234567", "type": "metrics", "epoch": 0, "train_loss": 1.0, "test_loss": 1.05}
{"timestamp": "2024-01-15T10:30:45.345678", "type": "metrics", "epoch": 1, "train_loss": 0.9, "test_loss": 0.95}
```

### File Organization

```
logs/
├── my_project_run_001.jsonl
├── my_project_run_001.jsonl.lock
├── my_project_run_002.jsonl
└── my_project_run_002.jsonl.lock
```

## Integration Patterns

### 1. Sklearn Integration

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlterm import Tracker

def train_sklearn_model():
    tracker = Tracker(project="sklearn_experiment", run_id="rf_001")
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        model_type="RandomForestRegressor",
        n_estimators=100,
        max_depth=10
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Log metrics
    tracker.log(
        train_mse=train_mse,
        test_mse=test_mse,
        train_r2=train_r2,
        test_r2=test_r2,
        feature_importance=model.feature_importances_.tolist()
    )
    
    tracker.finish("completed")
```

### 2. Deep Learning Integration

```python
import torch
from mlterm import Tracker

def train_pytorch_model():
    tracker = Tracker(project="pytorch_experiment", run_id="cnn_001")
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        model_type="CNN",
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        optimizer="Adam"
    )
    
    # Training loop
    for epoch in range(100):
        # Training code...
        train_loss = compute_train_loss()
        test_loss = compute_test_loss()
        accuracy = compute_accuracy()
        
        # Log metrics
        tracker.log(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            accuracy=accuracy,
            learning_rate=scheduler.get_last_lr()[0]
        )
    
    tracker.finish("completed")
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from mlterm import Tracker

def hyperparameter_tuning():
    tracker = Tracker(project="hyperparameter_sweep", run_id="sweep_001")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15]
    }
    
    # Log search space
    tracker.log_hyperparameters(
        search_type="GridSearchCV",
        param_grid=param_grid,
        cv_folds=5
    )
    
    # Perform search
    grid_search = GridSearchCV(
        RandomForestRegressor(),
        param_grid,
        cv=5,
        scoring='r2'
    )
    grid_search.fit(X_train, y_train)
    
    # Log results
    tracker.log(
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        cv_results=grid_search.cv_results_
    )
    
    tracker.finish("completed")
```

## Best Practices

### 1. Naming Conventions

```python
# Good project names
project="customer_churn_prediction"
project="image_classification_resnet"
project="time_series_forecasting"

# Good run IDs
run_id="baseline_model"
run_id="experiment_001"
run_id="hyperparameter_sweep_01"
```

### 2. Metric Logging Frequency

```python
# Log every epoch for short training
for epoch in range(10):
    # ... training code ...
    tracker.log(epoch=epoch, loss=loss)

# Log every 10 epochs for long training
for epoch in range(1000):
    # ... training code ...
    if epoch % 10 == 0:
        tracker.log(epoch=epoch, loss=loss)
```

### 3. Error Handling

```python
try:
    # Training code...
    tracker.finish("completed")
except Exception as e:
    tracker.log(error=str(e))
    tracker.finish("failed")
```

### 4. Resource Monitoring

```python
import psutil

# Log system resources
tracker.log(
    cpu_percent=psutil.cpu_percent(),
    memory_percent=psutil.virtual_memory().percent,
    process_memory=psutil.Process().memory_info().rss / 1024 / 1024  # MB
)
```

## Command Line Interface

### Available Commands

```bash
# Start dashboard
mlterm dashboard --project my_project

# List all runs
mlterm list-runs --project my_project

# Compare runs
mlterm compare --project my_project --runs run_001 run_002

# Export data
mlterm export --project my_project --format csv
```

### Dashboard Controls

- `q` - Quit dashboard
- `r` - Refresh manually
- `Tab` - Switch between tabs
- `↑↓` - Navigate metrics

## Troubleshooting

### Common Issues

1. **Log file not found**
   ```bash
   Error: No log file found for project 'my_project'
   ```
   **Solution**: Ensure the project name matches exactly

2. **JSON serialization error**
   ```python
   TypeError: Object of type ndarray is not JSON serializable
   ```
   **Solution**: Convert numpy arrays to lists
   ```python
   tracker.log(feature_importance=model.feature_importances_.tolist())
   ```

3. **Dashboard not updating**
   **Solution**: Check refresh rate and ensure log file is being written

### Debug Mode

```python
tracker = Tracker(
    project="debug_project",
    run_id="debug_001",
    log_dir="./logs",
    debug=True  # Enable debug logging
)
```

## Performance Considerations

### 1. Logging Frequency
- **High frequency**: Every iteration (for short training)
- **Medium frequency**: Every epoch (for medium training)
- **Low frequency**: Every 10-100 epochs (for long training)

### 2. Data Size
- Keep individual log entries small (< 1MB)
- Use compression for large artifacts
- Consider sampling for very frequent logging

### 3. File I/O
- Logging is synchronous (blocks until written)
- Use background logging for high-frequency scenarios
- Consider buffering for very frequent updates

## Integration with Other Tools

### 1. Jupyter Notebooks

```python
# In Jupyter cell
from mlterm import Tracker

tracker = Tracker(project="notebook_experiment", run_id="cell_001")
# ... training code ...
tracker.finish("completed")
```

### 2. MLflow Integration

```python
import mlflow
from mlterm import Tracker

# Use both tools together
with mlflow.start_run():
    tracker = Tracker(project="mlflow_experiment", run_id="run_001")
    # ... training code ...
    tracker.finish("completed")
```

### 3. Weights & Biases

```python
import wandb
from mlterm import Tracker

wandb.init(project="my_project")
tracker = Tracker(project="wandb_experiment", run_id="run_001")
# ... training code ...
tracker.finish("completed")
```

## Conclusion

The mlterm tracker provides a simple yet powerful way to monitor your machine learning experiments. By following the mandatory code components and best practices outlined in this guide, you can easily integrate real-time monitoring into any ML workflow.

The key is to:
1. Initialize the tracker with a unique project and run ID
2. Log hyperparameters at the start
3. Log metrics during training
4. Finish the run when complete
5. Use the dashboard to monitor progress

This approach works with any ML framework (sklearn, PyTorch, TensorFlow, etc.) and provides valuable insights into your training process.
