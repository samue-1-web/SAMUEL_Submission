# SAMUEL_Submission

# Contour Prediction Using Deep Learning

## Overview

This project predicts the current contour using the previous and future contour frames.

Three independent models are trained:

- Contour1 Model
- Contour2 Model
- Contour3 Model

Each contour is predicted separately and reconstructed using residual learning.

The final prediction is obtained by adding the learned residual to a linear interpolation of the previous and future contours.

---

# Training Sample Generation

Training samples are generated from contour sequences.

For a frame i and temporal offset n:

- Previous contour = frame(i − n)
- Future contour = frame(i + n)
- Ground truth contour = frame(i)

Only valid samples satisfying:

- i − n ≥ 0
- i + n < total_frames

are used.

All valid offsets are included during training.

For each generated sample:

1. Previous contour is resampled to 100 points.
2. Future contour is resampled to 100 points.
3. Ground truth contour is resampled to 100 points.
4. Normalization is applied.
5. Residual targets are computed.

Residual target:

Residual = Ground Truth − Linear Interpolation

where

Linear Interpolation = (Previous + Future) / 2

---

# Model Details

## Overall Approach

For every contour:

1. Previous contour frame is provided.
2. Future contour frame is provided.
3. Both contours are resampled to 100 points.
4. Contours are normalized.
5. A neural network predicts a residual contour.
6. The residual is added to a linear interpolation between previous and future contours.
7. The contour is denormalized and saved.

---

# Contour1 Architecture

Model Name:

ContourModel1Improved

Architecture:

Encoder:

- Linear(400 → 512)
- LayerNorm
- GELU

- Linear(512 → 256)
- LayerNorm
- GELU

- Dropout(0.05)

- Linear(256 → 128)
- LayerNorm
- GELU

Residual Block:

- Linear(128 → 128)
- GELU

- Linear(128 → 128)
- GELU

Decoder:

- Linear(128 → 256)
- LayerNorm
- GELU

- Linear(256 → 512)
- LayerNorm
- GELU

- Linear(512 → 200)

Output:

100 × 2 contour coordinates

---

# Contour2 Architecture

Model Name:

ContourModel2_DeepTCN

Input Features:

- Previous contour
- Future contour
- Motion feature
- Encoded temporal offset

Motion Feature:

Motion = Future − Previous

Temporal Encoding:

Sinusoidal encoding of temporal offset n

Architecture:

Block 1:

- Conv1D(6 → 64)
- BatchNorm
- GELU

Block 2:

- Conv1D(64 → 64)
- Dilation = 2
- BatchNorm
- GELU

Block 3:

- Conv1D(64 → 128)
- Dilation = 4
- BatchNorm
- GELU

Block 4:

- Conv1D(128 → 128)
- Dilation = 8
- BatchNorm
- GELU

Block 5:

- Conv1D(128 → 256)
- Dilation = 16
- BatchNorm
- GELU

Block 6:

- Conv1D(256 → 256)
- Dilation = 32
- BatchNorm
- GELU

Pooling:

- AdaptiveAvgPool1D

Head:

- Linear(256 + encoded_offset → 512)
- GELU
- Dropout(0.2)

- Linear(512 → 256)
- GELU

- Linear(256 → 200)

Output:

100 × 2 contour coordinates

---

# Contour3 Architecture

Model Name:

ResidualContourModel

Architecture:

- Linear(400 → 256)
- BatchNorm
- ReLU
- Dropout(0.2)

- Linear(256 → 128)
- BatchNorm
- ReLU
- Dropout(0.2)

- Linear(128 → 200)

Output:

100 × 2 contour coordinates

---

# Preprocessing

## Interpolation

All contours are resampled to:

100 points

Interpolation Method:

Linear interpolation based on contour arc length.

---

## Normalization

Mean and standard deviation are computed using:

Previous Contour + Future Contour

Normalization:

Normalized = (Contour − Mean) / Standard Deviation

---

## Feature Engineering

### Contour2 Only

Motion:

Motion = Future − Previous

Temporal Encoding:

Sinusoidal encoding:

sin(n/f)
cos(n/f)

where n is the temporal offset.

---

## Data Augmentation

Contour-specific augmentation is used during training.

Examples include:

- Small rotations
- Small translations
- Small scaling
- Smooth contour deformation

No augmentation is applied during inference.

---

# Hyperparameters

## Contour1

Learning Rate:

5e-5

Weight Decay:

1e-5

Batch Size:

32

Epochs:

100

Smoothness Weight:

0.1

Loss:

- Smooth L1 Loss
- SoftDTW

---

## Contour2

Learning Rate:

1e-4

Weight Decay:

1e-5

Batch Size:

32

Epochs:

100

Smoothness Weight:

0.01

Loss:

- Huber Loss
- SoftDTW

---

## Contour3

Learning Rate:

5e-5

Weight Decay:

1e-5

Batch Size:

32

Epochs:

100

Smoothness Weight:

0.05

Loss:

- Smooth L1 Loss
- SoftDTW

---

# Optimizer

Adam

---

# Learning Rate Scheduler

ReduceLROnPlateau

Factor:

0.5

Patience:

10

---

# Early Stopping

Patience: 30 epochs

Training stops when validation loss does not improve for 30 consecutive epochs.

# Residual Scaling Factors

The following residual scaling factors were selected using development-set DTW tuning.

Contour1:

α = 0.30

Contour2:

α = 0.10

Contour3:

α = 0.10

---

# Postprocessing

## Linear Interpolation

Linear estimate:

Linear = (Previous + Future) / 2

---

## Residual Reconstruction

Contour1:

Prediction = Linear + 0.30 × Residual

Contour2:

Prediction = Linear + 0.10 × Residual

Contour3:

Prediction = Linear + 0.10 × Residual

---

## Denormalization

Predicted contour coordinates are transformed back to original coordinate space:

Prediction = Prediction × Standard Deviation + Mean

---

## Visualization Generation

After contour reconstruction and denormalization:

1. Previous contour is plotted.
2. Future contour is plotted.
3. Predicted contour is plotted.

Separate PNG files are generated for:

- Contour1
- Contour2
- Contour3

These visualizations are produced automatically during inference and saved locally.

# Required Package Installation

Install:

```bash
pip install torch
pip install numpy
pip install scipy
pip install fastdtw
pip install pysdtw
```

# Execution

## Required Files

The following files should be present in the same folder:

```
inference.py

contour1_model.py
contour2_model.py
contour3_model.py

contour1_huber_best.pth
contour2_huber_best.pth
contour3_huber_best.pth

prev.mat
fut.mat
```

## Run

```bash
python inference.py
```

Inference generates:

1. predicted_contour.mat
2. contour1_prediction.png
3. contour2_prediction.png
4. contour3_prediction.png

# Input Files

prev.mat

Contains contour coordinate data corresponding to the previous frame.

fut.mat

Contains contour coordinate data corresponding to the future frame.

The exact field names are determined by the provided evaluation data.

Both files contain contour coordinates for:

- Contour1
- Contour2
- Contour3

# Output Files

## Contour Predictions

predicted_contour.mat

Contains:

- contour1
- contour2
- contour3

Predicted contour coordinates in original coordinate space.

---

## Visualization Files

The inference pipeline also generates visualizations for qualitative inspection.

Generated files:

- contour1_prediction.png
- contour2_prediction.png
- contour3_prediction.png

Each visualization contains:

- Previous contour
- Future contour
- Predicted contour

These plots are generated automatically during inference and are intended for visual verification of contour reconstruction quality.
