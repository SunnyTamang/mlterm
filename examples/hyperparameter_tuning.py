"""
Advanced sklearn hyperparameter tuning with mlterm tracking.
Shows how to track hyperparameter optimization experiments.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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

def hyperparameter_tuning_experiment(model_class, model_name, param_grid, X_train, y_train, X_test, y_test, tracker, search_type='grid'):
    """Perform hyperparameter tuning for a model."""
    
    print(f"\nHyperparameter tuning for {model_name}...")
    
    # Choose search method
    if search_type == 'grid':
        search = GridSearchCV(
            model_class(), 
            param_grid, 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
    else:  # random
        search = RandomizedSearchCV(
            model_class(), 
            param_grid, 
            n_iter=20,  # Number of parameter settings sampled
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    # Record tuning start time
    start_time = time.time()
    
    # Perform search
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    # Get best parameters and results
    best_params = search.best_params_
    best_score = search.best_score_
    best_model = search.best_estimator_
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Log hyperparameter tuning results
    tracker.log(
        model_name=model_name,
        tuning_method=search_type,
        best_params=best_params,
        best_cv_score=best_score,
        test_mse=test_mse,
        test_r2=test_r2,
        tuning_time=tuning_time,
        n_params_tested=len(search.cv_results_['params']),
        best_estimator=str(type(best_model).__name__)
    )
    
    # Log individual parameter combinations
    for i, (params, score) in enumerate(zip(search.cv_results_['params'], search.cv_results_['mean_test_score'])):
        tracker.log(
            model_name=f"{model_name}_param_combination",
            combination_index=i,
            params=params,
            cv_score=score,
            tuning_method=search_type
        )
    
    print(f"  Best params: {best_params}")
    print(f"  Best CV R-squared:  {best_score:.4f}")
    print(f"  Test R-squared:     {test_r2:.4f}")
    print(f"  Tuning time: {tuning_time:.2f}s")
    
    return {
        'model': best_model,
        'best_params': best_params,
        'best_cv_score': best_score,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'tuning_time': tuning_time
    }

def main():
    """Main hyperparameter tuning function."""
    
    # Initialize tracker
    tracker = Tracker(
        project="hyperparameter_tuning",
        run_id="tuning_001",
        log_dir="./logs"
    )
    
    print("Starting Hyperparameter Tuning with mlterm...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project hyperparameter_tuning")
    print("=" * 80)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_data()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log experiment setup
    tracker.log_hyperparameters(
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        n_features=X_train.shape[1],
        data_source="synthetic_regression",
        preprocessing="standardization",
        cv_folds=5,
        scoring_metric="r2"
    )
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Define hyperparameter grids for different models
    tuning_experiments = [
        {
            'model_class': Ridge,
            'model_name': 'Ridge Regression',
            'param_grid': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'search_type': 'grid'
        },
        {
            'model_class': Lasso,
            'model_name': 'Lasso Regression',
            'param_grid': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'search_type': 'grid'
        },
        {
            'model_class': ElasticNet,
            'model_name': 'ElasticNet',
            'param_grid': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'search_type': 'grid'
        },
        {
            'model_class': RandomForestRegressor,
            'model_name': 'Random Forest',
            'param_grid': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'search_type': 'random'  # Use random search for large grid
        },
        {
            'model_class': GradientBoostingRegressor,
            'model_name': 'Gradient Boosting',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'search_type': 'random'
        },
        {
            'model_class': SVR,
            'model_name': 'Support Vector Regression',
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'search_type': 'random'
        }
    ]
    
    results = []
    best_overall_model = None
    best_overall_score = float('inf')
    
    print(f"\nRunning {len(tuning_experiments)} hyperparameter tuning experiments...")
    
    for i, experiment in enumerate(tuning_experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(tuning_experiments)}: {experiment['model_name']}")
        print(f"{'='*60}")
        
        # Use scaled data for models that benefit from it
        if experiment['model_name'] in ['Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Support Vector Regression']:
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        else:
            X_train_use, X_test_use = X_train, X_test
        
        # Perform hyperparameter tuning
        result = hyperparameter_tuning_experiment(
            experiment['model_class'],
            experiment['model_name'],
            experiment['param_grid'],
            X_train_use, y_train, X_test_use, y_test,
            tracker,
            experiment['search_type']
        )
        
        results.append((experiment['model_name'], result))
        
        # Track best overall model
        if result['test_mse'] < best_overall_score:
            best_overall_score = result['test_mse']
            best_overall_model = result['model']
            best_overall_name = experiment['model_name']
        
        # Small delay to see updates in dashboard
        time.sleep(1)
    
    # Log final results
    tracker.log(
        best_overall_model=best_overall_name,
        best_overall_mse=best_overall_score,
        total_experiments=len(tuning_experiments),
        tuning_complete=True
    )
    
    # Save best model
    best_model_filename = f"best_tuned_model_{best_overall_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_overall_model, best_model_filename)
    
    # Log artifacts
    tracker.log_artifact(
        name="best_tuned_model",
        path=best_model_filename,
        description=f"Best tuned model: {best_overall_name} with MSE: {best_overall_score:.4f}"
    )
    
    tracker.log_artifact(
        name="scaler",
        path="scaler.pkl",
        description="StandardScaler used for preprocessing"
    )
    
    # Save scaler
    joblib.dump(scaler, "scaler.pkl")
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nHyperparameter tuning completed!")
    print(f"Best overall model: {best_overall_name} with test MSE: {best_overall_score:.4f}")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")
    
    # Print summary table
    print(f"\nTuning Results Summary:")
    print(f"{'Model':<25} {'Best CV R-squared':<12} {'Test R-squared':<10} {'Test MSE':<12} {'Time (s)':<10}")
    print("-" * 80)
    for model_name, result in results:
        print(f"{model_name:<25} {result['best_cv_score']:<12.4f} {result['test_r2']:<10.4f} "
              f"{result['test_mse']:<12.4f} {result['tuning_time']:<10.2f}")

if __name__ == "__main__":
    main()
