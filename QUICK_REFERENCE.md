# mlterm Tracker - Quick Reference

## Essential Code Template

```python
from mlterm import Tracker

# 1. Initialize tracker (REQUIRED)
tracker = Tracker(
    project="your_project_name",
    run_id="unique_run_id", 
    log_dir="./logs"
)

# 2. Log hyperparameters (REQUIRED)
tracker.log_hyperparameters(
    learning_rate=0.001,
    batch_size=32,
    model_type="RandomForest"
)

# 3. Log metrics during training (REQUIRED)
for epoch in range(num_epochs):
    # Your training code...
    
    tracker.log(
        epoch=epoch,
        train_loss=train_loss,
        test_loss=test_loss,
        accuracy=accuracy
    )

# 4. Finish the run (REQUIRED)
tracker.finish("completed")
```

## Dashboard Commands

```bash
# Start monitoring
mlterm dashboard --project your_project_name

# List all runs
mlterm list-runs --project your_project_name

# Compare runs
mlterm compare --project your_project_name --runs run_001 run_002
```

## Key Functions

| Function | Purpose | Required |
|----------|---------|----------|
| `Tracker()` | Initialize tracking session | ✅ Yes |
| `log_hyperparameters()` | Log model configuration | ✅ Yes |
| `log()` | Log training metrics | ✅ Yes |
| `log_artifact()` | Log files/models | ❌ Optional |
| `finish()` | End tracking session | ✅ Yes |

## Common Patterns

### Sklearn Models
```python
tracker.log(
    train_mse=float(mean_squared_error(y_train, y_pred)),
    test_mse=float(mean_squared_error(y_test, y_test_pred)),
    feature_importance=model.feature_importances_.tolist()
)
```

### Deep Learning
```python
tracker.log(
    epoch=epoch,
    train_loss=float(train_loss),
    test_loss=float(test_loss),
    learning_rate=float(scheduler.get_last_lr()[0])
)
```

### Hyperparameter Tuning
```python
tracker.log(
    best_params=grid_search.best_params_,
    best_score=float(grid_search.best_score_),
    cv_mean=float(cv_scores.mean()),
    cv_std=float(cv_scores.std())
)
```

## Troubleshooting

- **JSON Error**: Convert numpy arrays to lists with `.tolist()`
- **File Not Found**: Check project name spelling
- **Dashboard Not Updating**: Verify log file is being written
- **Performance**: Reduce logging frequency for long training

## File Structure
```
logs/
├── project_name_run_id.jsonl      # Main log file
└── project_name_run_id.jsonl.lock # Lock file
```
