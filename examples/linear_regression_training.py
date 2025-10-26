"""
Realistic linear regression training script with mlterm tracking.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import os
from mlterm import Tracker

def load_data():
    """Load the dummy data."""
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target'].values
    X_test = test_df.drop('target', axis=1).values
    y_test = test_df['target'].values
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, model_name, **model_params):
    """Train a model and return metrics."""
    
    # Initialize model
    if model_name == "linear":
        model = LinearRegression(**model_params)
    elif model_name == "ridge":
        model = Ridge(**model_params)
    elif model_name == "lasso":
        model = Lasso(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0]
    }

def main():
    """Main training function."""
    
    # Initialize tracker
    tracker = Tracker(
        project="linear_regression_experiment",
        run_id="lr_run_001",
        log_dir="./logs"
    )
    
    print("Starting Linear Regression Training with mlterm...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project linear_regression_experiment")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Log data info
    tracker.log_hyperparameters(
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        n_features=X_train.shape[1],
        data_source="synthetic_regression"
    )
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Define models to test
    models_to_test = [
        {"name": "linear", "params": {}},
        {"name": "ridge", "params": {"alpha": 1.0}},
        {"name": "ridge", "params": {"alpha": 0.1}},
        {"name": "lasso", "params": {"alpha": 0.1}},
        {"name": "lasso", "params": {"alpha": 1.0}},
    ]
    
    best_model = None
    best_score = float('inf')
    
    print("\nTraining multiple models...")
    
    for i, model_config in enumerate(models_to_test):
        model_name = model_config["name"]
        model_params = model_config["params"]
        
        print(f"\nTraining {model_name} with params: {model_params}")
        
        # Train model
        results = train_model(X_train, y_train, X_test, y_test, model_name, **model_params)
        
        # Log metrics
        tracker.log(
            model_name=model_name,
            model_params=model_params,
            train_mse=results['train_mse'],
            test_mse=results['test_mse'],
            train_r2=results['train_r2'],
            test_r2=results['test_r2'],
            train_mae=results['train_mae'],
            test_mae=results['test_mae'],
            training_time=results['training_time'],
            model_index=i
        )
        
        # Print results
        print(f"  Train MSE: {results['train_mse']:.4f}")
        print(f"  Test MSE:  {results['test_mse']:.4f}")
        print(f"  Train R-squared:   {results['train_r2']:.4f}")
        print(f"  Test R-squared:    {results['test_r2']:.4f}")
        print(f"  Time:       {results['training_time']:.4f}s")
        
        # Track best model
        if results['test_mse'] < best_score:
            best_score = results['test_mse']
            best_model = results['model']
            best_model_name = model_name
            best_params = model_params
        
        # Small delay to see updates in dashboard
        time.sleep(0.5)
    
    # Log best model
    tracker.log(
        best_model_name=best_model_name,
        best_model_params=best_params,
        best_test_mse=best_score,
        experiment_complete=True
    )
    
    # Log artifacts
    tracker.log_artifact(
        name="best_model",
        path="best_model.pkl",
        description=f"Best model: {best_model_name} with MSE: {best_score:.4f}"
    )
    
    tracker.log_artifact(
        name="training_data",
        path="train.csv",
        description="Training dataset"
    )
    
    tracker.log_artifact(
        name="test_data", 
        path="test.csv",
        description="Test dataset"
    )
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nTraining completed!")
    print(f"Best model: {best_model_name} with test MSE: {best_score:.4f}")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")

if __name__ == "__main__":
    main()
