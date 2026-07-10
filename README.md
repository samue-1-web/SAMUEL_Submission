# SAMUEL SUBMISSION

## Overview

This project predicts the contour of the current frame using the corresponding contours from previous and future frames.

Three independent deep learning models are trained:

- Contour1 Model
- Contour2 Model
- Contour3 Model

Each contour is predicted separately using residual learning. The final contour estimate is obtained by adding the predicted residual contour to a linear estimate computed from the previous and future contours.

---

# Methodology

## Problem Formulation

Given:

- Previous contour frame: C(t−n)
- Future contour frame: C(t+n)

The objective is to predict:

- Current contour frame: C(t)

where n represents the temporal offset.

---

# Training Sample Generation

Training samples are generated from contour sequences contained in the provided MATLAB annotation files.

For every frame i:

- Previous contour = frame (i−n)
- Future contour = frame (i+n)
- Ground truth contour = frame (i)

All valid temporal offsets are used during training.

A sample is included only if:

- i−n ≥ 0
- i+n < total frames

This strategy increases dataset size and enables the models to learn contour dynamics across multiple temporal distances.

---

# Preprocessing

## Contour Resampling

All contours are resampled to:

**100 points**

using arc-length based linear interpolation.

This ensures a consistent contour representation regardless of the original number of contour points.

---

## Normalization

For each sample:

Mean and standard deviation are computed using:

Previous Contour + Future Contour

Normalization:

```text
C_norm = (C - mean) / std
```

where:

- mean = contour mean
- std = contour standard deviation

The same normalization parameters are used for:

- Previous contour
- Future contour
- Ground truth contour

---

# Residual Learning

A linear contour estimate is first computed.

Velocity:

```text
V = (Future - Previous) / 2
```

Linear estimate:

```text
Linear = Previous + V
```

Residual target:

```text
Residual = GroundTruth - Linear
```

The neural network predicts the residual contour rather than the complete contour.

---

# Data Augmentation

During training, contour-specific augmentation is applied.

The following transformations are randomly used:

- Small rotations
- Small translations
- Small scaling operations

These augmentations improve model robustness and generalization.

No augmentation is applied during inference.

---

# Feature Engineering

## Contour2 Features

Additional motion-related features are used.

### Motion Feature

```text
Motion = Future - Previous
```

### Velocity Feature

```text
Velocity = Motion / 2
```

### Temporal Encoding

Temporal offsets are encoded using sinusoidal features:

```text
sin(n/f)
cos(n/f)
```

for multiple frequency scales.

---

# Model Architectures

## Contour1 Model

### Model Name

**ContourModel1_CNN**

### Architecture

Input:

- Previous contour (100×2)
- Future contour (100×2)

Concatenated input channels:

4 channels

Network:

- Conv1D(4 → 64, kernel=5)
- BatchNorm
- GELU

- Conv1D(64 → 128, kernel=5)
- BatchNorm
- GELU

- Conv1D(128 → 128, kernel=5)
- BatchNorm
- GELU

- Conv1D(128 → 64, kernel=3)
- BatchNorm
- GELU

- Conv1D(64 → 2, kernel=1)

Output:

- Residual contour (100 × 2)

---

## Contour2 Model

### Model Name

**ContourModel2_TCN**

### Input Features

- Previous contour
- Future contour
- Motion feature
- Velocity feature

Total channels:

8

### Architecture

Block 1:

- Conv1D(8 → 64)
- BatchNorm
- GELU

Block 2:

- Conv1D(64 → 64)
- Dilation = 2
- BatchNorm
- GELU

Residual Connection

Block 3:

- Conv1D(64 → 128)
- Dilation = 4
- BatchNorm
- GELU
- Dropout(0.1)

Block 4:

- Conv1D(128 → 128)
- Dilation = 8
- BatchNorm
- GELU
- Dropout(0.1)

Residual Connection

Refinement:

- Conv1D(128 → 128)
- BatchNorm
- GELU

- Conv1D(128 → 64)
- BatchNorm
- GELU

Output Layer:

- Conv1D(64 → 2)

Output:

- Residual contour (100 × 2)

---

## Contour3 Model

### Model Name

**ResidualContourModel**

### Architecture

Input:

Flattened previous and future contours

Dimension:

400

Backbone:

- Linear(400 → 512)
- BatchNorm
- ReLU
- Dropout(0.2)

- Linear(512 → 256)
- BatchNorm
- ReLU
- Dropout(0.2)

- Linear(256 → 200)

Shortcut:

- Linear(400 → 200)

Residual Output:

```text
Output = Backbone + Shortcut
```

Final output reshaped to:

100 × 2

---

# Loss Functions

The final training objective combines:

1. Regression loss
2. Contour smoothness loss
3. Soft Dynamic Time Warping (SoftDTW)

General form:

```text
Loss =
Regression Loss +
λs × Smoothness Loss +
λd × SoftDTW Loss
```

---

## Contour1

Regression Loss:

- SmoothL1Loss (β = 0.3)

SoftDTW:

- γ = 0.05
- Weight = 0.05

Smoothness Weight:

- 0.10

---

## Contour2

Regression Loss:

- Huber Loss (δ = 1.0)

SoftDTW:

- γ = 0.02
- Weight = 0.02

Smoothness Weight:

- 0.01

---

## Contour3

Regression Loss:

- SmoothL1Loss (β = 0.5)

SoftDTW:

- γ = 0.10
- Weight = 0.05

Smoothness Weight:

- 0.05

---

# Hyperparameters

| Parameter | Value |
|------------|--------|
| Epochs | 100 |
| Batch Size | 32 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 10 |
| Early Stopping Patience | 10 |

---

## Learning Rates

### Contour1

- Learning Rate = 5e-5
- Weight Decay = 1e-5

### Contour2

- Learning Rate = 1e-4
- Weight Decay = 5e-5

### Contour3

- Learning Rate = 5e-5
- Weight Decay = 1e-5

---

# Training Dataset Statistics

Training Set:

- Contour1 Samples: 4298
- Contour2 Samples: 4298
- Contour3 Samples: 4298

Development Set:

- Contour1 Samples: 5894
- Contour2 Samples: 5894
- Contour3 Samples: 5894

---

# Alpha Tuning

After training, residual scaling coefficients were tuned on the development set using DTW minimization.

Best coefficients:

| Contour | Alpha |
|----------|---------|
| Contour1 | 0.30 |
| Contour2 | 0.40 |
| Contour3 | 0.10 |

Development DTW:

| Contour | Best DEV DTW |
|----------|-------------|
| Contour1 | 47.1425 |
| Contour2 | 117.4699 |
| Contour3 | 36.8784 |

---

# Inference

For a given previous and future contour:

1. Resample contours to 100 points.
2. Normalize contours.
3. Generate model-specific features.
4. Predict residual contour.
5. Reconstruct contour using tuned alpha.
6. Denormalize contour coordinates.
7. Save prediction.

---

# Residual Reconstruction

## Contour1

```text
Prediction = Linear + 0.30 × Residual
```

## Contour2

```text
Prediction = Linear + 0.40 × Residual
```

## Contour3

```text
Prediction = Linear + 0.10 × Residual
```

---

# Final Test Results

## Contour1

- Mean Baseline DTW: 50.2768
- Mean AI DTW: 49.8073
- Model Wins: 3249 / 5894
- Win Rate: 55.1%

## Contour2

- Mean Baseline DTW: 87.8794
- Mean AI DTW: 87.2833
- Model Wins: 3321 / 5894
- Win Rate: 56.3%

## Contour3

- Mean Baseline DTW: 33.0847
- Mean AI DTW: 33.3725
- Model Wins: 2379 / 5894
- Win Rate: 40.4%

---

# Summary of Results

| Contour | Alpha | AI DTW | Win Rate |
|----------|--------|---------|-----------|
| Contour1 | 0.30 | 49.8073 | 55.1% |
| Contour2 | 0.40 | 87.2833 | 56.3% |
| Contour3 | 0.10 | 33.3725 | 40.4% |

---

# Required Package Installation

```bash
pip install torch
pip install numpy
pip install scipy
pip install pandas
pip install fastdtw
pip install pysdtw
```

---

# Execution

## Required Files

```text
training.py

contour1_huber_best.pth
contour2_huber_best.pth
contour3_huber_best.pth

trainingdata/
devdata/
testingdata/
```

## Run Training

```bash
python training.py
```

---

# Output

Generated model checkpoints:

```text
contour1_huber_best.pth
contour2_huber_best.pth
contour3_huber_best.pth
```

Generated latest checkpoints:

```text
contour1_huber_latest.pth
contour2_huber_latest.pth
contour3_huber_latest.pth
```

These models are subsequently used for contour prediction and DTW evaluation.
