#!/usr/bin/env python3
"""
Setup script for mlterm package.
This is a fallback for older pip versions that don't support pyproject.toml.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlterm",
    version="0.1.0",
    author="Sunny Tamang",
    author_email="sunnysinghtamang@gmail.com",
    description="A Terminal-Based ML Experiment Tracker",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SunnyTamang/mlterm",
    project_urls={
        "Homepage": "https://github.com/SunnyTamang/mlterm",
        "Repository": "https://github.com/SunnyTamang/mlterm",
        "Documentation": "https://github.com/SunnyTamang/mlterm#readme",
        "Issues": "https://github.com/SunnyTamang/mlterm/issues",
        "Changelog": "https://github.com/SunnyTamang/mlterm/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: Terminals",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
        "Environment :: Console",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires=">=3.8",
    install_requires=[
        "textual>=0.40.0",
        "psutil>=5.9.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "filelock>=3.12.0",
    ],
    extras_require={
        "gpu": ["GPUtil>=1.4.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlterm=mlterm.cli:app",
        ],
    },
    keywords=[
        "machine-learning", "ml", "terminal", "tui", "experiment-tracking",
        "monitoring", "tracking", "logging", "dashboard", "cli", "offline",
        "ssh", "remote", "jsonl", "metrics", "hyperparameters", "artifacts"
    ],
    include_package_data=True,
    zip_safe=False,
)
