#!/usr/bin/env python3
"""
Development setup script for mlterm.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Setup development environment."""
    print("Setting up mlterm development environment...\n")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: Not in a virtual environment. Consider using venv or conda.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return False
    
    # Install dependencies
    commands = [
        ("pip install -e .", "Installing mlterm in development mode"),
        ("pip install -e .[gpu]", "Installing mlterm with GPU support"),
    ]
    
    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False
            break
    
    if success:
        print("\nDevelopment setup completed!")
        print("\nNext steps:")
        print("1. Run tests: python test_mlterm.py")
        print("2. Try the example: python examples/simple_training.py")
        print("3. Start dashboard: mlterm dashboard")
        print("\nFor GPU support, install: pip install GPUtil")
    else:
        print("\nSetup failed. Check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
