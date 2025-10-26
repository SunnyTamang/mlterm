# Contributing to mlterm

Thank you for your interest in contributing to mlterm! We welcome contributions from the community and appreciate your help in making this project better.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please:

1. Check if the issue already exists in the [Issues](https://github.com/mlterm/mlterm/issues) section
2. Create a new issue with:
   - A clear, descriptive title
   - Detailed description of the problem or feature request
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Contributing Code

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/mlterm.git
   cd mlterm
   ```
3. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/your-bugfix-name
   ```
4. **Install in development mode**:
   ```bash
   pip install -e .
   pip install -e ".[dev]"
   ```
5. **Make your changes** and test them
6. **Run tests** to ensure nothing is broken:
   ```bash
   pytest tests/
   ```
7. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```
8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Create a Pull Request** on GitHub

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- git

### Installation

```bash
# Clone the repository
git clone https://github.com/mlterm/mlterm.git
cd mlterm

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Development Dependencies

The project includes optional development dependencies:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=mlterm tests/

# Run specific test file
pytest tests/test_tracker.py
```

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black mlterm/ tests/

# Sort imports
isort mlterm/ tests/

# Lint code
flake8 mlterm/ tests/

# Type check
mypy mlterm/
```

## Code Guidelines

### Python Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Commit Messages

Use clear, descriptive commit messages:

```
Add: feature description
Fix: bug description
Update: change description
Remove: removal description
Docs: documentation update
Test: test addition or update
```

### Pull Request Guidelines

- Keep PRs focused and small
- Include tests for new features
- Update documentation if needed
- Ensure all tests pass
- Add yourself to the contributors list if it's your first contribution

## Project Structure

```
mlterm/
├── mlterm/                 # Main package
│   ├── __init__.py        # Package initialization
│   ├── tracker.py         # Core Tracker class
│   ├── dashboard.py       # TUI dashboard
│   ├── system_monitor.py  # System monitoring
│   └── cli.py            # Command-line interface
├── examples/              # Usage examples
├── tests/                # Test suite
├── docs/                 # Documentation
├── pyproject.toml        # Package configuration
├── README.md             # Main documentation
├── LICENSE               # MIT License
├── CHANGELOG.md          # Version history
└── CONTRIBUTING.md       # This file
```

## Areas for Contribution

### High Priority

- **Tests**: Add more comprehensive test coverage
- **Documentation**: Improve examples and tutorials
- **Performance**: Optimize dashboard refresh rates
- **Features**: Add plotting and visualization capabilities

### Medium Priority

- **CLI**: Add more command-line tools
- **Export**: Support for more export formats
- **Integration**: Better integration with popular ML frameworks
- **UI**: Enhance dashboard appearance and functionality

### Low Priority

- **Web Dashboard**: Browser-based interface
- **Database**: Backend database support
- **Cloud**: Cloud storage integration
- **Notifications**: Real-time alerts

## Getting Help

If you need help or have questions:

- Check the [Issues](https://github.com/mlterm/mlterm/issues) section
- Start a [Discussion](https://github.com/mlterm/mlterm/discussions)
- Review the [README](README.md) and documentation

## Recognition

Contributors will be recognized in:

- The [README](README.md) contributors section
- Release notes for significant contributions
- The project's [CHANGELOG](CHANGELOG.md)

Thank you for contributing to mlterm! 🎉
