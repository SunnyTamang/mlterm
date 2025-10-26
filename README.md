# mlterm 🧠

[![PyPI version](https://badge.fury.io/py/mlterm.svg)](https://badge.fury.io/py/mlterm)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A Terminal-Based ML Experiment Tracker** - Monitor your machine learning experiments in real-time with a beautiful TUI dashboard.

## ✨ Features

- 🖥️ **Terminal-Based Dashboard**: Beautiful TUI interface built with Textual
- 📊 **Real-Time Monitoring**: Live updates of training metrics and system stats
- 🔒 **Offline-First**: Works without internet connectivity
- 🖥️ **SSH-Friendly**: Perfect for remote server training
- 📝 **JSONL Logging**: Structured, human-readable log format
- 🎯 **Simple Integration**: Just 4 lines of code to get started
- 🔧 **Framework Agnostic**: Works with any ML framework (sklearn, PyTorch, TensorFlow, etc.)
- 📈 **Rich Metrics**: Track loss, accuracy, hyperparameters, artifacts, and more
- 💾 **Artifact Management**: Log models, data files, and other artifacts
- 🖥️ **System Monitoring**: CPU, memory, GPU, disk, and network usage

## 🚀 Quick Start

### Installation

```bash
pip install mlterm
```

### Basic Usage

```python
from mlterm import Tracker

# Initialize tracker
tracker = Tracker(project="my_experiment", run_id="run_001")

# Log hyperparameters
tracker.log_hyperparameters(learning_rate=0.001, batch_size=32)

# Log metrics during training
for epoch in range(100):
    # Your training code...
    tracker.log(epoch=epoch, loss=loss, accuracy=accuracy)

# Finish the run
tracker.finish("completed")
```

### Monitor Your Training

```bash
# Start the dashboard
mlterm dashboard --project my_experiment
```

## 📖 Documentation

### Core Components

- **Tracker**: Log metrics, hyperparameters, and artifacts
- **Dashboard**: Real-time TUI monitoring interface
- **CLI**: Command-line tools for managing experiments

### Integration Examples

#### Sklearn Models
```python
from sklearn.ensemble import RandomForestRegressor
from mlterm import Tracker

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

# Log metrics
tracker.log(
    train_mse=mean_squared_error(y_train, model.predict(X_train)),
    test_mse=mean_squared_error(y_test, model.predict(X_test)),
    feature_importance=model.feature_importances_.tolist()
)

tracker.finish("completed")
```

#### Deep Learning (PyTorch)
```python
import torch
from mlterm import Tracker

tracker = Tracker(project="pytorch_experiment", run_id="cnn_001")

# Log hyperparameters
tracker.log_hyperparameters(
    model_type="CNN",
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Training loop
for epoch in range(100):
    # Training code...
    train_loss = compute_train_loss()
    test_loss = compute_test_loss()
    
    tracker.log(
        epoch=epoch,
        train_loss=train_loss,
        test_loss=test_loss,
        learning_rate=scheduler.get_last_lr()[0]
    )

tracker.finish("completed")
```

### Command Line Interface

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

## 🎯 Key Features Explained

### Real-Time Dashboard

The dashboard provides three main views:

1. **Overview**: System statistics and latest metrics
2. **System**: CPU, memory, GPU, disk, and network monitoring
3. **Logs**: Raw log entries with timestamps

### Offline-First Design

- All data stored locally in JSONL format
- No internet connection required
- Perfect for SSH environments
- Data persists between sessions

### Flexible Logging

```python
# Log any metrics you want
tracker.log(
    epoch=epoch,
    train_loss=loss,
    test_loss=test_loss,
    accuracy=accuracy,
    learning_rate=lr,
    custom_metric=value
)

# Log artifacts
tracker.log_artifact(
    name="best_model",
    path="model.pkl",
    description="Best performing model"
)
```

## 📁 Project Structure

```
mlterm/
├── mlterm/
│   ├── __init__.py          # Package initialization
│   ├── tracker.py           # Core Tracker class
│   ├── dashboard.py         # TUI dashboard
│   ├── system_monitor.py    # System monitoring
│   └── cli.py              # Command-line interface
├── examples/               # Usage examples
├── tests/                  # Test suite
├── pyproject.toml         # Package configuration
└── README.md              # This file
```

## 🔧 Configuration

### Optional Dependencies

```bash
# Install with GPU monitoring support
pip install mlterm[gpu]
```

### Environment Variables

```bash
# Set default log directory
export MLTERM_LOG_DIR="/path/to/logs"

# Set default refresh rate
export MLTERM_REFRESH_RATE="1.0"
```

## 📊 Example Output

### Console Output
```
🚀 Starting training with mlterm tracking...
📊 Project: my_experiment
🆔 Run ID: run_001
📁 Log file: logs/my_experiment_run_001.jsonl
💡 In another terminal, run: mlterm dashboard --project my_experiment
============================================================
Epoch   0: Loss=2.0896, Acc=0.4901, LR=0.001000
Epoch  10: Loss=1.5833, Acc=0.6232, LR=0.000950
Epoch  20: Loss=1.1833, Acc=0.7527, LR=0.000902
...
✅ Training completed!
```

### Dashboard View
```
 ⭘                      mlterm Dashboard — Monitoring:                          
 Overview  System  Logs                                                         
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
System Statistics                                                               
CPU: 55.1% (12 cores, 3504 MHz)                                                 
System Memory: 6.5GB / 16.0GB (74.8%)                                           
GPU: Not available                                                              
 Metric               Value                  Timestamp                          
 iteration            125                    15:10:16                           
 phase                hyperparameter_tuning  15:10:16                           
 elapsed_time         125.6078               15:10:16                           
 progress_percent     41.8693                15:10:16                           
 simulated_train_mse  0.0000                 15:10:16                           
 simulated_test_mse   1164.9410              15:10:16                           
 simulated_train_r2   0.9693                 15:10:16                           
 simulated_test_r2    0.6674                 15:10:16                           
 cpu_usage            24.2517                15:10:16                           
 memory_usage         0.4720                 15:10:16                           
 learning_rate        0.0008                 15:10:16                           
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mlterm/mlterm.git
cd mlterm

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Textual](https://github.com/Textualize/textual) for the beautiful TUI framework
- [Typer](https://github.com/tiangolo/typer) for the CLI interface
- [Rich](https://github.com/Textualize/rich) for terminal formatting
- [psutil](https://github.com/giampaolo/psutil) for system monitoring

## 📞 Support

- 📖 [Documentation](https://github.com/mlterm/mlterm#readme)
- 🐛 [Issue Tracker](https://github.com/mlterm/mlterm/issues)
- 💬 [Discussions](https://github.com/mlterm/mlterm/discussions)

## 🗺️ Roadmap

- [ ] Web dashboard interface
- [ ] Database backend support
- [ ] Plotting and visualization
- [ ] Multi-run comparison tools
- [ ] Export to popular formats (CSV, Excel, etc.)
- [ ] Integration with popular ML platforms
- [ ] Cloud storage support
- [ ] Real-time notifications

---

**Made with ❤️ for the ML community**