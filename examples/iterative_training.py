"""
Iterative training example with mlterm - simulates gradient descent training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from mlterm import Tracker

class SimpleLinearRegression:
    """Simple linear regression implementation for iterative training."""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        """Train the model using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            y_pred = X.dot(self.weights) + self.bias
            
            # Calculate loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
            
            # Calculate gradients
            dw = (2 / n_samples) * X.T.dot(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                break
                
        return self
    
    def predict(self, X):
        """Make predictions."""
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def load_and_preprocess_data():
    """Load and preprocess the data."""
    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target'].values
    X_test = test_df.drop('target', axis=1).values
    y_test = test_df['target'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_with_tracking(X_train, y_train, X_test, y_test, tracker):
    """Train model with detailed tracking."""
    
    # Different learning rates to test
    learning_rates = [0.01, 0.001, 0.1]
    
    for lr_idx, lr in enumerate(learning_rates):
        print(f"\nTraining with learning rate: {lr}")
        
        # Initialize model
        model = SimpleLinearRegression(learning_rate=lr, max_iter=1000)
        
        # Log hyperparameters for this run
        tracker.log_hyperparameters(
            learning_rate=lr,
            max_iterations=1000,
            tolerance=1e-6,
            run_index=lr_idx
        )
        
        # Train model with detailed tracking
        n_samples, n_features = X_train.shape
        model.weights = np.random.normal(0, 0.01, n_features)
        model.bias = 0
        
        for iteration in range(1000):
            # Forward pass
            y_pred = X_train.dot(model.weights) + model.bias
            
            # Calculate loss
            train_loss = np.mean((y_train - y_pred) ** 2)
            
            # Calculate gradients
            dw = (2 / n_samples) * X_train.T.dot(y_pred - y_train)
            db = (2 / n_samples) * np.sum(y_pred - y_train)
            
            # Update parameters
            model.weights -= lr * dw
            model.bias -= lr * db
            
            # Calculate test metrics every 10 iterations
            if iteration % 10 == 0:
                y_test_pred = model.predict(X_test)
                test_loss = np.mean((y_test - y_test_pred) ** 2)
                test_r2 = model.score(X_test, y_test)
                
                # Log metrics
                tracker.log(
                    iteration=iteration,
                    learning_rate=lr,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    test_r2=test_r2,
                    weight_norm=np.linalg.norm(model.weights),
                    bias=model.bias,
                    run_index=lr_idx
                )
            
            # Check convergence
            if iteration > 0 and abs(train_loss - model.loss_history[-1] if model.loss_history else 0) < 1e-6:
                print(f"  Converged at iteration {iteration}")
                break
                
            # Small delay to see real-time updates
            time.sleep(0.01)
        
        # Final evaluation
        final_train_loss = np.mean((y_train - model.predict(X_train)) ** 2)
        final_test_loss = np.mean((y_test - model.predict(X_test)) ** 2)
        final_test_r2 = model.score(X_test, y_test)
        
        # Log final results
        tracker.log(
            iteration=iteration,
            learning_rate=lr,
            train_loss=final_train_loss,
            test_loss=final_test_loss,
            test_r2=final_test_r2,
            converged=True,
            final_iteration=iteration,
            run_index=lr_idx
        )
        
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Test Loss:  {final_test_loss:.4f}")
        print(f"  Final Test R-squared:    {final_test_r2:.4f}")

def main():
    """Main training function."""
    
    # Initialize tracker
    tracker = Tracker(
        project="iterative_linear_regression",
        run_id="iter_lr_001",
        log_dir="./logs"
    )
    
    print("Starting Iterative Linear Regression Training...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project iterative_linear_regression")
    print("=" * 70)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Log data info
    tracker.log_hyperparameters(
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        n_features=X_train.shape[1],
        data_source="synthetic_regression",
        preprocessing="standardization"
    )
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Train with tracking
    train_with_tracking(X_train, y_train, X_test, y_test, tracker)
    
    # Log artifacts
    tracker.log_artifact(
        name="training_data",
        path="data/train.csv",
        description="Training dataset"
    )
    
    tracker.log_artifact(
        name="test_data",
        path="data/test.csv", 
        description="Test dataset"
    )
    
    # Finish the run
    tracker.finish("completed")
    
    print(f"\nIterative training completed!")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")

if __name__ == "__main__":
    main()
