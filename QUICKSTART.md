# 🚀 mlterm Quick Start Guide

Get up and running with mlterm in minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mlterm.git
cd mlterm

# Install in development mode
pip install -e .

# Optional: Install with GPU support
pip install -e .[gpu]
```

## Quick Demo

1. **Start a training simulation**:
   ```bash
   python examples/simple_training.py
   ```

2. **In another terminal, launch the dashboard**:
   ```bash
   mlterm dashboard --project example_classification
   ```

3. **Watch your metrics update in real-time!**

## Basic Usage

### In Your Training Script

```python
from mlterm import Tracker

# Initialize tracker
tracker = Tracker(project="my_experiment", run_id="run_001")

# Log hyperparameters
tracker.log_hyperparameters(
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# In your training loop
for epoch in range(100):
    # Your training code here
    loss = train_step()
    accuracy = validate()
    
    # Log metrics
    tracker.log(
        epoch=epoch,
        loss=loss,
        accuracy=accuracy
    )

# Finish the run
tracker.finish("completed")
```

### Monitor with Dashboard

```bash
# Monitor specific project
mlterm dashboard --project my_experiment

# Monitor all projects
mlterm dashboard

# List available runs
mlterm list-runs

# Get detailed info about a run
mlterm info logs/my_experiment_run_001.jsonl
```

## Key Features

- **🧩 Unified View**: Metrics + system stats in one terminal
- **⚡ Real-time**: Auto-refreshes every 2 seconds
- **🔒 Offline**: No internet required
- **📊 System Monitoring**: CPU, GPU, memory usage
- **📈 Persistent Logs**: JSONL format for easy analysis

## Dashboard Navigation

- **Overview Tab**: Real-time metrics and system stats
- **System Tab**: Detailed system resource monitoring  
- **Logs Tab**: Live log entries and run events
- **Press 'q'**: Quit dashboard
- **Press 'r'**: Manual refresh

## Examples

Check out the `examples/` directory:
- `simple_training.py`: Basic usage example
- `advanced_training.py`: Multiple experiments and hyperparameter sweeps

## Configuration

Create a `.mlterm.yml` file to customize settings:

```yaml
log_dir: "./logs"
refresh_rate: 2.0
dashboard:
  max_log_entries: 20
  theme: "auto"
```

## Troubleshooting

**Dashboard not showing data?**
- Make sure the log file exists: `ls logs/`
- Check the project name matches: `mlterm list-runs`

**GPU monitoring not working?**
- Install GPU support: `pip install GPUtil`
- Check if NVIDIA drivers are installed

**Need help?**
- Check the full documentation in `README.md`
- Run tests: `python test_mlterm.py`

## Next Steps

1. **Integrate with your training scripts**
2. **Set up monitoring for remote servers**
3. **Compare multiple experiments**
4. **Export data for analysis**

Happy experimenting! 🧠✨
