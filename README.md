
# Sunspot Forecasting: Hybrid Ridge + LightGBM + EVT

## Overview

This project implements an advanced time-series forecasting pipeline to predict **Daily International Sunspot Numbers**. The goal is to accurately model the 11-year solar cycle while improving the detection of extreme solar activity events, which are critical for space weather monitoring.

We employ a **Hybrid Machine Learning Architecture** that decomposes the problem into three distinct layers:

1. **Trend & Seasonality (Linear Baseline)**:
   - **Ridge Regression** captures the broad auto-regressive trends and dominant periodicities of the time series.

2. **Non-Linear Residual Correction**:
   - **LightGBM** (default) or XGBoost is trained on the *residuals* of the linear model, learning the complex short-term non-linear dynamics that the linear model misses.

3. **Tail Calibration (Extreme Value Theory)**:
   - **EVT (Peaks-Over-Threshold)** models the tail distribution of forecast errors, adjusting predictions for extreme solar events.

## Current Performance

Evaluated with expanding-window walk-forward validation (1965–2026):

| Metric | Value |
|--------|-------|
| RMSE   | 19.24 |
| MAE    | 17.54 |

## Project Structure

```
Sunspots/
│
├── data/
│   ├── raw/             # Immutable source data (SIDC CSVs)
│   └── processed/       # Generated features
│
├── src/                 # Source code module
│   ├── data.py          # Data ingestion and cleaning
│   ├── features.py      # Feature engineering
│   ├── model.py         # The HybridEVTModel class
│   ├── train.py         # Multi-step forecasting logic
│   └── utils.py         # Plotting and configuration
│
├── models/              # Exported model data and configuration
│
├── notebooks/
│   └── 01-Analysis_and_Modeling.ipynb  # Primary training and analysis
│
├── 02-Gradio_Predictions.ipynb         # Interactive Gradio App (Toni's Predictor)
│
├── config.yaml          # Hyperparameters (Random State = 7)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup & Usage

### 1. Create & Activate a Virtual Environment

```powershell
# From the project root
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Register the Jupyter Kernel (optional, recommended)

So the notebooks pick up the venv automatically:

```bash
python -m ipykernel install --user --name=sunspots --display-name "Sunspots Venv"
```

### 4. Training the Model

Open `notebooks/01-Analysis_and_Modeling.ipynb` to:
1. Load data and engineer features.
2. Evaluate the hybrid model using walk-forward validation.
3. **Export the model data**: the last cell saves the required files to `models/`.

### 5. Interactive Predictions (Toni's Predictor)

Run `02-Gradio_Predictions.ipynb` to launch an interactive Gradio web interface. You can:
- Select the number of future days to forecast (up to 30).
- View a table of predicted sunspot counts.
- See a trend plot combining recent history and future forecasts.
- Inspect the historical backtesting performance.

## Methodology

### Feature Engineering

| Feature group | Details |
|---|---|
| Auto-regressive lags | t-1, t-2, t-3, t-4, t-5, t-7, t-30, t-365, t-4015 (log scale) |
| Rolling statistics | Mean, std, min, max over windows of 7, 30, 365, 4015 days |
| Momentum | `lag_1 − lag_30` (short vs. long-term trend direction) |
| Annual seasonality | sin/cos encoding of day-of-year |
| Solar cycle phase | sin/cos encoding anchored to the 2008-12 solar minimum (~4015-day cycle) |

### Validation Strategy

**Expanding Window Walk-Forward Validation** strictly prevents data leakage:
- **Initial training window**: 11 years (~4015 days)
- **Step size**: 90 days (quarterly advances for more reliable estimates)
- **Forecast horizon**: 5 days (direct multi-step)

### Reproducibility

The model uses a fixed **Random State of 7** (Toni's lucky number) in `config.yaml` to ensure reproducible results.

### Data

- **Source**: [SILSO](https://www.sidc.be/silso/datafiles) (Sunspot Index and Long-term Solar Observations, Belgium)
- **Preprocessing**: `log(1+x)` transformation to stabilise variance on the heavy-tailed distribution.

## Authors
[Jordi Roselló / Toni Majà]
