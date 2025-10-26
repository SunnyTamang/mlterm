"""
Advanced example showing mlterm with context manager and multiple experiments.
"""

import time
import random
import math
from mlterm import Tracker


def run_experiment(project_name: str, run_id: str, config: dict):
    """Run a single experiment with the given configuration."""
    
    with Tracker(project=project_name, run_id=run_id, log_dir="./logs") as tracker:
        # Log hyperparameters
        tracker.log_hyperparameters(**config)
        
        print(f"Running experiment: {run_id}")
        print(f"Config: {config}")
        
        # Simulate training with different characteristics based on config
        base_learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        
        for epoch in range(epochs):
            # Simulate different training dynamics based on config
            if config["optimizer"] == "SGD":
                # SGD typically has more oscillatory behavior
                base_loss = 2.0 * math.exp(-epoch / 40) + 0.1
                noise = random.gauss(0, 0.1)
            else:  # Adam
                # Adam typically has smoother convergence
                base_loss = 2.0 * math.exp(-epoch / 25) + 0.05
                noise = random.gauss(0, 0.03)
            
            train_loss = max(0.01, base_loss + noise)
            val_loss = train_loss * (1.1 + random.gauss(0, 0.05))
            
            # Accuracy progression
            base_accuracy = 0.3 + 0.6 * (1 - math.exp(-epoch / 30))
            train_accuracy = min(0.99, base_accuracy + random.gauss(0, 0.01))
            val_accuracy = train_accuracy * (0.98 + random.gauss(0, 0.01))
            
            # Learning rate schedule
            if config["lr_schedule"] == "exponential":
                lr = base_learning_rate * (0.95 ** (epoch // 5))
            else:  # cosine
                lr = base_learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            
            # Log metrics
            tracker.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                learning_rate=lr,
                batch_time=random.uniform(0.05, 0.2),
                # Only log GPU metrics if GPU is actually available
                # gpu_utilization=random.uniform(60, 90),  # Commented out - no real GPU
                process_memory=random.uniform(0.3, 1.5)  # GB - process-specific memory
            )
            
            # Simulate training time
            time.sleep(0.05)
        
        print(f"Experiment {run_id} completed!")


def run_hyperparameter_sweep():
    """Run multiple experiments with different hyperparameters."""
    
    # Define hyperparameter configurations
    configs = [
        {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "optimizer": "Adam",
            "lr_schedule": "exponential"
        },
        {
            "learning_rate": 0.01,
            "batch_size": 64,
            "epochs": 50,
            "optimizer": "SGD",
            "lr_schedule": "cosine"
        },
        {
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 50,
            "optimizer": "Adam",
            "lr_schedule": "cosine"
        }
    ]
    
    project_name = "hyperparameter_sweep"
    
    print("Starting hyperparameter sweep...")
    print("Monitor with: mlterm dashboard --project hyperparameter_sweep")
    print("=" * 60)
    
    for i, config in enumerate(configs):
        run_id = f"sweep_{i+1:02d}"
        run_experiment(project_name, run_id, config)
        
        # Small delay between experiments
        time.sleep(1)
    
    print("\nHyperparameter sweep completed!")
    print("Compare results: mlterm list-runs --project hyperparameter_sweep")


if __name__ == "__main__":
    run_hyperparameter_sweep()
