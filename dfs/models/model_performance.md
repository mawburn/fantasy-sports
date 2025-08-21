# Model Performance Log

## Current Best Results

| Position | MAE   | R²     | Best Epoch | Learning Rate | Batch Size | Date       |
| -------- | ----- | ------ | ---------: | ------------: | ---------: | ---------- |
| QB       | 6.337 | 0.156  |         90 |        0.0001 |         64 | 2025-08-21 |
| RB       | 6.337 | -0.029 |         13 |       0.00003 |        128 | 2025-08-21 |
| WR       | 5.399 | 0.250  |        558 |        0.0001 |         64 | 2025-08-21 |
| TE       | 6.164 | -0.886 |          3 |        0.0001 |         64 | 2025-08-21 |
| DST      | 3.905 | 0.008  |          - |             - |          - | 2025-08-21 |

## Experiment History

### QB Model Evolution

| Date       | R²     | MAE    | Learning Rate | Batch Size | Result                |
| ---------- | ------ | ------ | ------------- | ---------- | --------------------- |
| Baseline   | -2.309 | 15.341 | 0.00003       | 128        | ❌ Terrible           |
| 2025-08-21 | 0.156  | 6.337  | 0.0001        | 64         | ✅ Major breakthrough |

**Changes Made:**

- Baseline: Original standard neural network
- 2025-08-21: Added Vegas features + Multi-head architecture + DFS loss + 1000 epochs + No early stopping

**QB Insights:**

- Vegas features + multi-head architecture = +2.46 R² improvement
- Higher learning rate (0.0001) > Lower LR (0.00003)
- Smaller batches (64) > larger batches (128)
- Feature engineering was the breakthrough, not just hyperparameters

### RB Model Evolution

| Date       | R²     | MAE   | Learning Rate | Batch Size | Result                |
| ---------- | ------ | ----- | ------------- | ---------- | --------------------- |
| 2025-08-21 | -0.029 | 6.337 | 0.00003       | 128        | ⚠️ Needs optimization |

**Changes Made:**

- 2025-08-21: Removed early stopping, full 800 epochs

**RB Insights:**

- Best epoch was very early (13/800) - suggests overfitting or poor features
- Needs same feature engineering approach as QB
- Low learning rate might be limiting learning

### WR Model Evolution

| Date       | R²    | MAE   | Learning Rate | Batch Size | Result              |
| ---------- | ----- | ----- | ------------- | ---------- | ------------------- |
| 2025-08-21 | 0.250 | 5.399 | 0.0001        | 64         | ✅ Good performance |

**Changes Made:**

- 2025-08-21: Full 600 epochs, no early stopping, best checkpoint at epoch 558

**WR Insights:**

- Strong performance with R² = 0.250, consistent with previous results
- Late convergence (epoch 558/600) suggests model needs full training time
- Good MAE (5.399) for WR position complexity

### TE Model Evolution

| Date       | R²     | MAE   | Learning Rate | Batch Size | Result              |
| ---------- | ------ | ----- | ------------- | ---------- | ------------------- |
| 2025-08-21 | -0.886 | 6.164 | 0.0001        | 64         | ❌ Poor performance |

**Changes Made:**

- 2025-08-21: Full 600 epochs, no early stopping, best epoch 3

**TE Insights:**

- Severe degradation from previous R² = 0.260 to -0.886
- Very early best epoch (3/600) indicates major overfitting or data issues
- Predictions showing zero variance (all 0.6) - validation warnings in logs
- Needs investigation: feature engineering, data quality, or architecture issues

### DST Model Evolution

| Date       | R²    | MAE   | Learning Rate | Batch Size | Result                      |
| ---------- | ----- | ----- | ------------- | ---------- | --------------------------- |
| 2025-08-21 | 0.008 | 3.905 | -             | -          | ⚠️ Minimal predictive power |

**Changes Made:**

- 2025-08-21: CatBoost model with 1431 samples, 11 features

**DST Insights:**

- Very low R² (0.008) indicates model barely better than random
- Good MAE (3.905) but lacks predictive variance
- CatBoost with basic defensive features insufficient for DST complexity
- Position inherently difficult to predict due to game script variability
