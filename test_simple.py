#!/usr/bin/env python3
"""
Simple test script to verify mlterm works with real data.
"""

import pandas as pd
import numpy as np
from mlterm import Tracker
import time

def main():
    """Simple test with real data."""
    
    # Initialize tracker
    tracker = Tracker(
        project="simple_test",
        run_id="test_001",
        log_dir="./logs"
    )
    
    print("Starting Simple Test with mlterm...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project simple_test")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        print(f"Data loaded: {train_df.shape[0]} train, {test_df.shape[0]} test samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Log data info
    tracker.log_hyperparameters(
        n_train_samples=train_df.shape[0],
        n_test_samples=test_df.shape[0],
        n_features=train_df.shape[1] - 1,  # -1 for target column
        data_source="synthetic_regression"
    )
    
    # Simulate training with real metrics
    print("\nSimulating training...")
    
    for epoch in range(10):
        # Simulate realistic metrics
        train_loss = 100.0 * np.exp(-epoch / 3) + np.random.normal(0, 5)
        test_loss = train_loss * (1.1 + np.random.normal(0, 0.05))
        r2_score = 0.5 + 0.4 * (1 - np.exp(-epoch / 2)) + np.random.normal(0, 0.02)
        
        # Log metrics
        tracker.log(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            r2_score=r2_score,
            learning_rate=0.001 * (0.9 ** epoch)
        )
        
        print(f"Epoch {epoch:2d}: Loss={train_loss:.2f}, R²={r2_score:.3f}")
        time.sleep(0.5)
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nSimple test completed!")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")

if __name__ == "__main__":
    main()
