"""
Simple Random Forest training with mlterm tracking.
Perfect for single model training scenarios.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import time
import joblib
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

def train_random_forest(X_train, y_train, X_test, y_test, tracker):
    """Train a Random Forest model and return comprehensive metrics."""
    
    print("Training Random Forest...")
    
    # Model hyperparameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        model_type="RandomForestRegressor",
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        n_features=X_train.shape[1]
    )
    
    # Create and train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Record training start time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    
    # Feature importance
    feature_importance = model.feature_importances_.tolist()
    
    # Log comprehensive metrics
    tracker.log(
        train_mse=train_mse,
        test_mse=test_mse,
        train_r2=train_r2,
        test_r2=test_r2,
        train_mae=train_mae,
        test_mae=test_mae,
        training_time=training_time,
        cv_r2_mean=cv_mean,
        cv_r2_std=cv_std,
        n_features=len(feature_importance),
        feature_importance=feature_importance,
        model_trained=True
    )
    
    # Print results
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Train R-squared:   {train_r2:.4f}")
    print(f"  Test R-squared:    {test_r2:.4f}")
    print(f"  CV R-squared:      {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Time:       {training_time:.4f}s")
    
    # Print feature importance
    print(f"\nFeature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"    Feature {i+1}: {importance:.4f}")
    
    return {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'training_time': training_time,
        'feature_importance': feature_importance
    }

def main():
    """Main training function."""
    
    # Initialize tracker
    tracker = Tracker(
        project="random_forest_single",
        run_id="rf_001",
        log_dir="./logs"
    )
    
    print("Starting Random Forest Training with mlterm...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project random_forest_single")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Train Random Forest
    result = train_random_forest(X_train, y_train, X_test, y_test, tracker)
    
    # Save model
    model_filename = "random_forest_model.pkl"
    joblib.dump(result['model'], model_filename)
    
    # Log artifacts
    tracker.log_artifact(
        name="random_forest_model",
        path=model_filename,
        description=f"Random Forest model with test MSE: {result['test_mse']:.4f}"
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
    
    # Log final summary
    tracker.log(
        final_test_mse=result['test_mse'],
        final_test_r2=result['test_r2'],
        final_cv_r2=result['cv_r2_mean'],
        training_completed=True
    )
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nTraining completed!")
    print(f"Random Forest - Test MSE: {result['test_mse']:.4f}, Test R-squared: {result['test_r2']:.4f}")
    print(f"Cross-validation R-squared: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    print(f"Training time: {result['training_time']:.4f}s")
    print(f"Model saved: {model_filename}")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")

if __name__ == "__main__":
    main()
