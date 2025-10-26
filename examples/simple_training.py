"""
Example training script demonstrating mlterm integration.

This script shows how to use mlterm to track a simple machine learning experiment.
"""

import time
import random
import math
from mlterm import Tracker


def simulate_training():
    """Simulate a training process with realistic metrics."""
    
    # Initialize tracker
    tracker = Tracker(
        project="example_classification",
        run_id="demo_run_001",
        log_dir="./logs"
    )
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        model="ResNet50",
        optimizer="Adam",
        loss_function="CrossEntropy"
    )
    
    print("Starting training with mlterm tracking...")
    print(f"Project: {tracker.project}")
    print(f"Run ID: {tracker.run_id}")
    print(f"Log file: {tracker.log_file}")
    print("\nIn another terminal, run: mlterm dashboard --project example_classification")
    print("=" * 70)
    
    # Simulate training loop
    for epoch in range(100):
        # Simulate realistic training metrics
        base_loss = 2.0 * math.exp(-epoch / 30) + 0.1
        noise = random.gauss(0, 0.05)
        train_loss = max(0.01, base_loss + noise)
        
        # Validation loss (usually higher than training)
        val_loss = train_loss * (1.2 + random.gauss(0, 0.1))
        
        # Accuracy (increases over time)
        base_accuracy = 0.5 + 0.4 * (1 - math.exp(-epoch / 20))
        train_accuracy = min(0.99, base_accuracy + random.gauss(0, 0.02))
        val_accuracy = train_accuracy * (0.95 + random.gauss(0, 0.02))
        
        # Learning rate (decay over time)
        lr = 0.001 * (0.95 ** (epoch // 10))
        
        # Log metrics FIRST
        tracker.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=lr,
            batch_time=random.uniform(0.1, 0.3),  # seconds per batch
            # Only log GPU metrics if GPU is actually available
            # gpu_utilization=random.uniform(70, 95),  # Commented out - no real GPU
            process_memory=random.uniform(0.5, 2.0)  # GB - process-specific memory
        )
        
        # Print progress every 10 epochs - use the SAME values that were logged
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}, LR={lr:.6f}")
        
        # Simulate training time
        time.sleep(0.1)
    
    # Log final results
    tracker.log_artifact(
        name="model_checkpoint",
        path="./model_final.pth",
        description="Final trained model"
    )
    
    tracker.log_artifact(
        name="training_plot",
        path="./training_curves.png",
        description="Training and validation curves"
    )
    
    # Finish the run
    tracker.finish("completed")
    
    print("\nTraining completed!")
    print(f"Check the dashboard: mlterm dashboard --project {tracker.project}")


if __name__ == "__main__":
    simulate_training()
