# mlterm Examples

This directory contains example scripts demonstrating how to use mlterm for ML experiment tracking.

## Examples

### 1. Simple Training (`simple_training.py`)

A basic example showing how to integrate mlterm into a training loop:

```bash
python examples/simple_training.py
```

This script:
- Initializes a tracker for a classification project
- Logs hyperparameters
- Simulates a training loop with realistic metrics
- Logs artifacts (model checkpoints, plots)
- Demonstrates real-time monitoring

### 2. Advanced Training (`advanced_training.py`)

A more sophisticated example showing:
- Context manager usage
- Multiple experiments with different configurations
- Hyperparameter sweeps
- Different learning rate schedules

```bash
python examples/advanced_training.py
```

## Running the Examples

1. **Start the training script**:
   ```bash
   python examples/simple_training.py
   ```

2. **Monitor in another terminal**:
   ```bash
   mlterm dashboard --project example_classification
   ```

3. **List available runs**:
   ```bash
   mlterm list-runs
   ```

4. **Get detailed info about a run**:
   ```bash
   mlterm info logs/example_classification_demo_run_001.jsonl
   ```

## Key Features Demonstrated

- **Real-time monitoring**: Watch metrics update live in the dashboard
- **System monitoring**: CPU, GPU, and memory usage tracking
- **Multiple experiments**: Compare different runs side-by-side
- **Artifact logging**: Track model checkpoints and visualizations
- **Context management**: Automatic run completion handling

## Dashboard Features

The mlterm dashboard provides:
- **Overview tab**: Real-time metrics and system stats
- **System tab**: Detailed system resource monitoring
- **Logs tab**: Live log entries and run events

## Keyboard Shortcuts

- `q`: Quit the dashboard
- `r`: Manual refresh
- `Tab`: Switch between tabs
