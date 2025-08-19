# NFL DFS Position Model Training Guide

## Overview

This document provides a comprehensive guide to the training parameters, features, and configurations for each position model in the DFS optimization system. Each position has its own specialized neural network architecture and feature engineering pipeline optimized for predicting DraftKings fantasy performance.

## Global Training Configuration

### Device Optimization

- **Apple Silicon M-Series**: Automatically uses MPS (Metal Performance Shaders) for GPU acceleration
- **CUDA**: Falls back to CUDA if available
- **CPU**: Uses all available cores with optimized thread count
- **Deterministic Training**: Fixed seed (42) for reproducible results

### Base Training Parameters (All Positions)

- **Optimizer**: Adam with weight decay (1e-4)
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Early Stopping Patience**: 15 epochs (25 for DST due to smaller dataset)
- **Gradient Clipping**: Max norm of 0.5 to prevent exploding gradients
- **Validation Split**: 80/20 train/validation
- **Loss Function**: MSE (Mean Squared Error)

## Position-Specific Configurations

### 1. Quarterback (QB)

#### Network Architecture (QBNetwork)

- **Input Layer**: Variable based on feature count
- **Feature Layers**:
  - Linear(input_size → 128) + BatchNorm1d + ReLU + Dropout(0.2)
  - Linear(128 → 64) + BatchNorm1d + ReLU + Dropout(0.15)
- **Specialized Branches**:
  - **Passing Branch**: Linear(64 → 32) + ReLU + Dropout(0.1)
  - **Rushing Branch**: Linear(64 → 16) + ReLU + Dropout(0.1)
- **Output**: Combined branches → Linear(48 → 16) → Linear(16 → 1)
- **Fantasy Point Range**: 0-45 points

#### Training Hyperparameters

- **Learning Rate**: 0.00015
- **Batch Size**: 50
- **Epochs**: 200
- **BatchNorm eps**: 1e-3 (higher for stability)

#### Key Features

- **Correlation Features**:
  - Team rushing/receiving yards per game
  - Number of viable receivers (>75 yards)
  - Max target concentration
  - RB/TE involvement in passing game
  - Top 3 targets (share & consistency)
- **Statistical Features**:
  - Recent passing/rushing performance
  - TD rates and interception rates
  - Home/away splits
- **Defensive Matchup**:
  - Opponent pass defense metrics
  - Pressure/sack rates allowed

### 2. Running Back (RB)

#### Network Architecture (RBNetwork)

- **Input Layer**: Variable based on feature count
- **Feature Layers**:
  - Linear(input_size → 96) + LayerNorm + ReLU + Dropout(0.25)
  - Linear(96 → 48) + LayerNorm + ReLU + Dropout(0.2)
- **Specialized Branches**:
  - **Workload Branch**: Linear(48 → 24) + ReLU + Dropout(0.15)
  - **Efficiency Branch**: Linear(48 → 16) + ReLU + Dropout(0.1)
- **Output**: Combined → Linear(40 → 16) → Linear(16 → 1) + Sigmoid
- **Fantasy Point Range**: 0-35 points (scaled output)

#### Training Hyperparameters

- **Learning Rate**: 0.00001 (lowest among positions)
- **Batch Size**: 50
- **Epochs**: 250 (highest among positions)
- **Normalization**: LayerNorm (better for MPS)

#### Key Features

- **Workload Features**:
  - Average touches (carries + targets)
  - Workhorse rate (>15 carries)
  - Pass involvement rate
  - TD regression indicators
- **Efficiency Metrics**:
  - Yards per carry
  - Yards after catch
  - Red zone usage
- **Game Script Correlation**:
  - Expected game flow
  - Score differential tendencies

### 3. Wide Receiver (WR)

#### Network Architecture (WRNetwork)

- **Main Layers**:
  - Linear(input_size → 112) + ReLU + Dropout(0.3)
  - Linear(112 → 56) + ReLU + Dropout(0.25)
  - Linear(56 → 28) + ReLU + Dropout(0.2)
- **Specialized Branches**:
  - **Target Branch**: Linear(28 → 16) + ReLU + Dropout(0.15)
  - **Big Play Branch**: Linear(28 → 12) + ReLU + Dropout(0.1)
- **Output**: Combined → Linear(28 → 16) → Linear(16 → 1) + Sigmoid
- **Fantasy Point Range**: 0-30 points

#### Training Hyperparameters

- **Learning Rate**: 0.0001
- **Batch Size**: 50
- **Epochs**: 200
- **Higher Dropout**: 0.3 initial (volatility handling)

#### Key Features

- **Target Competition**:
  - Number of receivers on team
  - Target concentration metrics
  - WR vs TE target dominance
- **Matchup Features**:
  - Defense completion rate allowed
  - Defense yards per completion
  - TD rate allowed to WRs
- **Performance Metrics**:
  - Recent target share
  - Catch rate
  - Yards per reception
  - Deep ball rate

### 4. Tight End (TE)

#### Network Architecture (TENetwork)

- **Feature Layers**:
  - Linear(input_size → 80) + LayerNorm + ReLU + Dropout(0.2)
  - Linear(80 → 40) + LayerNorm + ReLU + Dropout(0.15)
  - Linear(40 → 20) + LayerNorm + ReLU + Dropout(0.1)
- **Output**: Linear(20 → 12) → Linear(12 → 1) + Sigmoid
- **Fantasy Point Range**: 0-25 points
- **Simpler Architecture**: Due to more predictable role

#### Training Hyperparameters

- **Learning Rate**: 0.0001
- **Batch Size**: 50
- **Epochs**: 200
- **Note**: Learning rate appears to be 0.000 in code (likely typo, should be 0.0001)

#### Key Features

- **Role Features**:
  - TE involvement percentage
  - Red zone target rate
  - Pass blocking vs route running
- **Competition Features**:
  - WR target dominance on team
  - Overall passing volume
- **Matchup Features**:
  - Defense vs TE metrics
  - Middle of field coverage

### 5. Defense/Special Teams (DST/DEF)

#### Network Architecture (DEFNetwork)

- **Shared Layers**:
  - Linear(input_size → 64) + ReLU + Dropout(0.4)
  - Linear(64 → 32) + ReLU + Dropout(0.3)
- **Triple Branch Architecture**:
  - **Pressure Branch**: Linear(32 → 16) + ReLU + Dropout(0.2)
  - **Turnover Branch**: Linear(32 → 12) + ReLU + Dropout(0.2)
  - **Points Branch**: Linear(32 → 8) + ReLU + Dropout(0.15)
- **Output**: Combined (36) → Linear(36 → 18) → Linear(18 → 1) + Sigmoid
- **Fantasy Point Range**: 0-20 points
- **Highest Dropout**: 0.4 (high variance position)

#### Training Hyperparameters

- **Learning Rate**: 0.001 (highest, small dataset)
- **Batch Size**: 16 (smallest, ~1K samples)
- **Epochs**: 200
- **Patience**: 25 (extended for small dataset)

#### Key Features

- **Opponent Tendencies**:
  - Pass/run ratio
  - Red zone frequency
  - Third down rate
  - Average play efficiency
- **Defensive Performance**:
  - Sack rate
  - Interception rate
  - Pressure rate
  - Stuff rate (runs ≤3 yards)
- **Game Script Features**:
  - Late game exposure
  - Pressure situation frequency
- **Special Teams**:
  - Return TD potential
  - Field position metrics

## Feature Engineering Pipeline

### Common Features (All Positions)

1. **Recent Performance** (4-week lookback default):

   - Average fantasy points
   - Min/max points (floor/ceiling)
   - Consistency score
   - Games played

2. **Game Context**:

   - Home/away indicator
   - Season and week
   - Division/conference matchup
   - Weather conditions (if outdoor)

3. **Statistical Aggregations**:
   - Position-specific stats (yards, TDs, etc.)
   - Per-game averages
   - Efficiency metrics
   - Trend indicators

### Data Preprocessing

1. **NaN Handling**: Convert to 0.0 with safeguards
2. **Outlier Clipping**:
   - Features: [-1000, 1000]
   - Targets: [-10, 50] fantasy points
3. **Feature Scaling**: StandardScaler (mean=0, std=1)
4. **Constant Feature Removal**: Variance threshold
5. **Batch Normalization**: In-network normalization

## Training Process

### Data Collection Flow

1. Load player-game combinations from database
2. Pre-compute defensive matchup features (cached)
3. Extract correlation features (if available)
4. Compute statistical features from historical data
5. Ensure consistent feature space across all samples

### Training Loop

1. **Forward Pass**: Input → Network → Predictions
2. **Loss Calculation**: MSE between predictions and actual
3. **Gradient Computation**: Backpropagation with checks
4. **Gradient Clipping**: Prevent exploding gradients
5. **Parameter Update**: Adam optimizer step
6. **Learning Rate Scheduling**: Reduce on plateau
7. **Early Stopping**: Monitor validation loss

### Model Evaluation Metrics

- **MAE** (Mean Absolute Error): Primary metric
- **RMSE** (Root Mean Squared Error): Penalizes outliers
- **R²** (Coefficient of Determination): Explained variance
- **Training/Validation Split**: Monitor overfitting

## Prediction & Uncertainty Quantification

### Point Estimates

- Direct network output (scaled by position range)

### Uncertainty Estimates

- **Confidence Score**: Fixed at 0.7 (can be improved)
- **Prediction Intervals**: ±1.96 \* residual_std (95% CI)
- **Floor**: point_estimate - 0.8 \* uncertainty
- **Ceiling**: point_estimate + 1.0 \* uncertainty

## Correlated Multi-Position Model

### Architecture Components

1. **GameContextEncoder**:

   - Encodes game-level features affecting all players
   - Multi-head attention (4 heads)
   - Shared context for correlation

2. **PositionSpecificHeads**:

   - Separate prediction head per position
   - Fusion of game context + player features
   - Uncertainty parameter estimation

3. **Stack Factors**:
   - Learnable correlation matrix
   - Models inter-position dependencies
   - Adjusts predictions based on correlations

### Correlation Adjustments

- 10% weight on correlation adjustments
- Captures QB-WR stacking benefits
- RB-DEF negative correlation (game script)
- TE-WR target competition

## Optimization Recommendations

### For Small Datasets (DST)

- Reduce batch size (16)
- Increase patience (25)
- Higher learning rate (0.001)
- More epochs (200+)

### For Volatile Positions (WR)

- Higher dropout rates (0.3+)
- More regularization
- Ensemble approaches
- Wider prediction intervals

### For Stable Positions (RB/TE)

- Lower learning rates
- More training epochs
- Focus on workload features
- Tighter prediction intervals

## Model Storage & Loading

### File Structure

```
dfs/models/
├── qb_model.pth     # QB network weights
├── rb_model.pth     # RB network weights
├── wr_model.pth     # WR network weights
├── te_model.pth     # TE network weights
└── dst_model.pth    # DST network weights
```

### Model Versioning

- Version tracked in ModelConfig
- Backward compatibility maintained
- State dict format (PyTorch standard)

## Future Improvements

1. **Dynamic Learning Rates**: Position-specific schedules
2. **Ensemble Methods**: Multiple models per position
3. **Advanced Features**:
   - Player embeddings
   - Team style factors
   - Coaching tendency models
4. **Uncertainty Quantification**:
   - Bayesian neural networks
   - Monte Carlo dropout
   - Quantile regression
5. **Real-time Updates**:
   - Injury adjustments
   - Weather changes
   - Late scratches
