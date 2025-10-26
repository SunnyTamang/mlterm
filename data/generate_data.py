"""
Generate dummy data for linear regression example.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import os

def generate_linear_regression_data(n_samples=1000, n_features=5, noise=0.1, random_state=42):
    """
    Generate dummy data for linear regression.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Amount of noise to add
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Training and test data
    """
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, data_dir="data"):
    """Save the generated data to CSV files."""
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save training data
    train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    train_df['target'] = y_train
    train_df.to_csv(f"{data_dir}/train.csv", index=False)
    
    # Save test data
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    test_df['target'] = y_test
    test_df.to_csv(f"{data_dir}/test.csv", index=False)
    
    print(f"Data saved to {data_dir}/")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

if __name__ == "__main__":
    # Generate data
    print("Generating dummy data for linear regression...")
    X_train, X_test, y_train, y_test = generate_linear_regression_data(
        n_samples=1000,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    
    # Save data
    save_data(X_train, X_test, y_train, y_test)
    
    print("\n Files created:")
    print("  - data/train.csv")
    print("  - data/test.csv")
