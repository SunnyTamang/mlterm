# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of mlterm
- Terminal-based ML experiment tracker
- Real-time TUI dashboard
- JSONL logging format
- System monitoring (CPU, memory, GPU, disk, network)
- CLI interface with dashboard, list-runs, compare, and export commands
- Support for hyperparameter logging
- Artifact management
- Offline-first design
- SSH-friendly interface
- Framework agnostic (works with sklearn, PyTorch, TensorFlow, etc.)
- Examples for various ML frameworks
- Comprehensive documentation

### Features
- **Tracker Class**: Core logging functionality
- **Dashboard**: Beautiful TUI interface built with Textual
- **System Monitor**: Real-time system resource monitoring
- **CLI Tools**: Command-line interface for managing experiments
- **JSONL Logging**: Structured, human-readable log format
- **Artifact Management**: Log models, data files, and other artifacts
- **Real-time Updates**: Live monitoring with configurable refresh rates
- **Multi-phase Training**: Support for different training phases
- **Cross-validation**: Track CV scores and statistics
- **Feature Importance**: Log feature importance for tree-based models

### Examples
- Simple training simulation
- Sklearn models training (Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR)
- Hyperparameter tuning
- Random Forest single model training
- Long-running Decision Tree training (5-minute simulation)
- Linear regression training
- Iterative training examples

### Documentation
- Comprehensive README with installation and usage instructions
- Quick reference guide
- Detailed tracker guide
- Integration patterns for popular ML frameworks
- Best practices and troubleshooting guide

## [0.1.0] - 2024-10-26

### Added
- Initial release
- Core tracker functionality
- TUI dashboard
- CLI interface
- System monitoring
- Example scripts
- Documentation
