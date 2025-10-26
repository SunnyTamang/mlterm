"""
Simple Decision Tree training with mlterm tracking.
Runs for approximately 5 minutes to demonstrate long training monitoring.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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

def train_decision_tree_iterative(X_train, y_train, X_test, y_test, tracker):
    """Train Decision Tree with iterative updates for long monitoring."""
    
    print("Training Decision Tree (5-minute simulation)...")
    
    # Model hyperparameters
    max_depth = 20
    min_samples_split = 2
    min_samples_leaf = 1
    random_state = 42
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        model_type="DecisionTreeRegressor",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        n_features=X_train.shape[1],
        training_duration="5_minutes_simulation"
    )
    
    # Create model
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Record training start time
    start_time = time.time()
    
    # Train the model (this is fast, but we'll simulate longer training)
    model.fit(X_train, y_train)
    actual_training_time = time.time() - start_time
    
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
    
    # Log initial metrics
    tracker.log(
        train_mse=train_mse,
        test_mse=test_mse,
        train_r2=train_r2,
        test_r2=test_r2,
        train_mae=train_mae,
        test_mae=test_mae,
        training_time=actual_training_time,
        cv_r2_mean=cv_mean,
        cv_r2_std=cv_std,
        n_features=len(feature_importance),
        feature_importance=feature_importance,
        model_trained=True,
        phase="initial_training"
    )
    
    print(f"  Initial Training Complete:")
    print(f"    Train MSE: {train_mse:.4f}")
    print(f"    Test MSE:  {test_mse:.4f}")
    print(f"    Train R-squared:   {train_r2:.4f}")
    print(f"    Test R-squared:    {test_r2:.4f}")
    print(f"    CV R-squared:      {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"    Time:       {actual_training_time:.4f}s")
    
    # Simulate longer training with iterative updates
    print(f"\nSimulating extended training (5 minutes)...")
    
    # Simulate different training phases
    phases = [
        ("feature_engineering", 60),      # 1 minute
        ("hyperparameter_tuning", 120),    # 2 minutes  
        ("model_validation", 60),          # 1 minute
        ("final_evaluation", 60)           # 1 minute
    ]
    
    total_simulation_time = 300  # 5 minutes
    iteration = 0
    
    for phase_name, phase_duration in phases:
        print(f"    Phase: {phase_name} ({phase_duration}s)")
        
        # Simulate work in this phase
        phase_start = time.time()
        while time.time() - phase_start < phase_duration:
            # Simulate some work and log progress
            elapsed = time.time() - start_time
            progress = elapsed / total_simulation_time
            
            # Simulate improving metrics over time
            improvement_factor = 1.0 + (progress * 0.1)  # 10% improvement over time
            
            simulated_train_mse = train_mse * (1.0 / improvement_factor)
            simulated_test_mse = test_mse * (1.0 / improvement_factor)
            simulated_train_r2 = min(1.0, train_r2 * improvement_factor)
            simulated_test_r2 = min(1.0, test_r2 * improvement_factor)
            
            # Add some noise to make it realistic
            noise_factor = 1.0 + np.random.normal(0, 0.02)
            
            # # Log progress
            # tracker.log(
            #     iteration=iteration,
            #     phase=phase_name,
            #     elapsed_time=elapsed,
            #     progress_percent=progress * 100,
            #     simulated_train_mse=simulated_train_mse * noise_factor,
            #     simulated_test_mse=simulated_test_mse * noise_factor,
            #     simulated_train_r2=simulated_train_r2 * noise_factor,
            #     simulated_test_r2=simulated_test_r2 * noise_factor,
            #     cpu_usage=np.random.uniform(20, 80),
            #     memory_usage=np.random.uniform(0.1, 0.5),
            #     learning_rate=0.001 * (1.0 - progress * 0.5)  # Decay over time
            # )
            
            iteration += 1
            
            # Print progress every 30 seconds
            if iteration % 30 == 0:
                print(f"      {elapsed:.0f}s elapsed, {progress*100:.1f}% complete")
            
            time.sleep(1)  # Update every second
    
    # Final metrics
    final_elapsed = time.time() - start_time
    
    # Log final results
    tracker.log(
        final_train_mse=train_mse,
        final_test_mse=test_mse,
        final_train_r2=train_r2,
        final_test_r2=test_r2,
        final_cv_r2_mean=cv_mean,
        final_cv_r2_std=cv_std,
        total_training_time=final_elapsed,
        total_iterations=iteration,
        training_completed=True,
        phase="completed"
    )
    
    print(f"\nExtended training simulation completed!")
    print(f"    Total time: {final_elapsed:.0f}s")
    print(f"    Total iterations: {iteration}")
    
    return {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'training_time': final_elapsed,
        'feature_importance': feature_importance,
        'total_iterations': iteration
    }

def main():
    """Main training function."""
    
    # Initialize tracker
    tracker = Tracker(
        project="decision_tree_long_training",
        run_id="dt_5min_001",
        log_dir="./logs"
    )
    
    print("Starting Decision Tree Long Training with mlterm...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print(f"Expected duration: ~5 minutes")
    print("\nIn another terminal, run: mlterm dashboard --project decision_tree_long_training")
    print("=" * 80)
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Train Decision Tree with extended simulation
    result = train_decision_tree_iterative(X_train, y_train, X_test, y_test, tracker)
    
    # Save model
    model_filename = "decision_tree_5min_model.pkl"
    joblib.dump(result['model'], model_filename)
    
    # Log artifacts
    tracker.log_artifact(
        name="decision_tree_model",
        path=model_filename,
        description=f"Decision Tree model trained for {result['training_time']:.0f}s"
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
        total_iterations=result['total_iterations'],
        training_completed=True
    )
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nLong training completed!")
    print(f"Decision Tree - Test MSE: {result['test_mse']:.4f}, Test R-squared: {result['test_r2']:.4f}")
    print(f"Cross-validation R-squared: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    print(f"Total training time: {result['training_time']:.0f}s")
    print(f"Total iterations: {result['total_iterations']}")
    print(f"Model saved: {model_filename}")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")
    
    # Print feature importance
    print(f"\nFeature Importance:")
    for i, importance in enumerate(result['feature_importance']):
        print(f"    Feature {i+1}: {importance:.4f}")

if __name__ == "__main__":
    main()
