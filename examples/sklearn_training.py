"""
Real sklearn model training with mlterm tracking.
This shows how to use mlterm with actual sklearn models for real ML experiments.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
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

def train_sklearn_model(model, model_name, X_train, y_train, X_test, y_test, tracker):
    """Train a sklearn model and return comprehensive metrics."""
    
    print(f"\nTraining {model_name}...")
    
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
    cv_mean = float(cv_scores.mean())  # Convert to Python float
    cv_std = float(cv_scores.std())    # Convert to Python float
    
    # Model-specific metrics
    model_metrics = {}
    if hasattr(model, 'coef_'):
        model_metrics['n_features'] = len(model.coef_)
        model_metrics['feature_importance'] = model.coef_.tolist()
        model_metrics['intercept'] = float(model.intercept_)  # Convert to Python float
    elif hasattr(model, 'feature_importances_'):
        model_metrics['n_features'] = len(model.feature_importances_)
        model_metrics['feature_importance'] = model.feature_importances_.tolist()
    
    # Log comprehensive metrics
    tracker.log(
        model_name=model_name,
        train_mse=train_mse,
        test_mse=test_mse,
        train_r2=train_r2,
        test_r2=test_r2,
        train_mae=train_mae,
        test_mae=test_mae,
        training_time=training_time,
        cv_r2_mean=cv_mean,
        cv_r2_std=cv_std,
        **model_metrics
    )
    
    # Print results
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Train R-squared:   {train_r2:.4f}")
    print(f"  Test R-squared:    {test_r2:.4f}")
    print(f"  CV R-squared:      {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Time:       {training_time:.4f}s")
    
    return {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'training_time': training_time
    }

def main():
    """Main training function with sklearn models."""
    
    # Initialize tracker
    tracker = Tracker(
        project="sklearn_models_experiment",
        run_id="sklearn_001",
        log_dir="./logs"
    )
    
    print("Starting Sklearn Models Training with mlterm...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project sklearn_models_experiment")
    print("=" * 80)
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log data info
    tracker.log_hyperparameters(
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        n_features=X_train.shape[1],
        data_source="synthetic_regression",
        preprocessing="standardization",
        test_size=0.2
    )
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Define models to test
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge (α=1.0)", Ridge(alpha=1.0)),
        ("Ridge (α=0.1)", Ridge(alpha=0.1)),
        ("Lasso (α=0.1)", Lasso(alpha=0.1)),
        ("Lasso (α=1.0)", Lasso(alpha=1.0)),
        ("ElasticNet (α=0.1)", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("Random Forest (100 trees)", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Random Forest (500 trees)", RandomForestRegressor(n_estimators=500, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("SVR (RBF)", SVR(kernel='rbf', C=1.0)),
        ("SVR (Linear)", SVR(kernel='linear', C=1.0)),
    ]
    
    results = []
    best_model = None
    best_score = float('inf')
    
    print(f"\nTraining {len(models)} different models...")
    
    for i, (model_name, model) in enumerate(models):
        # Use scaled data for models that benefit from it
        if model_name.startswith(('Ridge', 'Lasso', 'ElasticNet', 'SVR')):
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        else:
            X_train_use, X_test_use = X_train, X_test
        
        # Train model
        result = train_sklearn_model(
            model, model_name, X_train_use, y_train, X_test_use, y_test, tracker
        )
        
        results.append((model_name, result))
        
        # Track best model
        if result['test_mse'] < best_score:
            best_score = result['test_mse']
            best_model = result['model']
            best_model_name = model_name
        
        # Small delay to see updates in dashboard
        time.sleep(0.5)
    
    # Log best model summary
    tracker.log(
        best_model_name=best_model_name,
        best_test_mse=best_score,
        total_models_tested=len(models),
        experiment_complete=True
    )
    
    # Save best model
    model_filename = f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_filename)
    
    # Log artifacts
    tracker.log_artifact(
        name="best_model",
        path=model_filename,
        description=f"Best model: {best_model_name} with MSE: {best_score:.4f}"
    )
    
    tracker.log_artifact(
        name="scaler",
        path="scaler.pkl",
        description="StandardScaler used for preprocessing"
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
    
    # Save scaler
    joblib.dump(scaler, "scaler.pkl")
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nTraining completed!")
    print(f"Best model: {best_model_name} with test MSE: {best_score:.4f}")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")
    
    # Print summary table
    print(f"\nResults Summary:")
    print(f"{'Model':<25} {'Test MSE':<10} {'Test R-squared':<10} {'CV R-squared':<15} {'Time (s)':<10}")
    print("-" * 80)
    for model_name, result in results:
        print(f"{model_name:<25} {result['test_mse']:<10.4f} {result['test_r2']:<10.4f} "
              f"{result['cv_r2_mean']:<10.4f}±{result['cv_r2_std']:<4.4f} {result['training_time']:<10.4f}")

if __name__ == "__main__":
    main()
