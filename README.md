
# Sunspot Forecasting: Hybrid Ridge + LightGBM + EVT

## Overview

This project implements an advanced time-series forecasting pipeline to predict **Daily International Sunspot Numbers**. The goal is to accurately model the 11-year solar cycle while improving the detection of extreme solar activity events, which are critical for space weather monitoring.

**Why it matters:** geomagnetic storms triggered by high sunspot activity disrupt satellite operations, GPS accuracy, power grids, and HF radio communications. Accurate short-term forecasts help operators take preventive action.

We employ a **Hybrid Machine Learning Architecture** that decomposes the problem into three layers:

1. **Trend & Seasonality (Ridge Regression)** — captures broad auto-regressive trends and dominant periodicities
2. **Non-Linear Residual Correction (LightGBM)** — trained on Ridge residuals to learn short-term non-linear dynamics
3. **Tail Calibration (EVT / Peaks-Over-Threshold)** — adjusts predictions for extreme solar events

## Current Performance

Evaluated with expanding-window walk-forward validation (1965–2026):

| Model | Granularity | RMSE | MAE |
|---|---|---|---|
| Naive (lag-1) | daily | 22.04 | 19.59 |
| Rolling mean (30d) | daily | 28.47 | 26.16 |
| McNish-Lincoln (13m smooth) | monthly | 32.36 | 23.64 |
| ARIMA(5,1,0) | monthly | 24.33 | 17.51 |
| **Hybrid (ours)** | **daily** | **19.24** | **17.54** |

Our hybrid beats the best daily baseline (naive lag-1) by **−12.7% RMSE** and **−10.4% MAE**.

Notable: naive lag-1 outperforms longer rolling means, confirming the strong short-lag autocorrelation visible in the ACF. McNish-Lincoln performs poorly at a 5-day horizon — expected, as it is designed for monthly/yearly cycle forecasting. ARIMA(5,1,0) is competitive on MAE but operates at monthly granularity.

## Project Structure

```
Sunspots/
│
├── data/
│   ├── raw/             # Immutable source data (SIDC CSV)
│   └── processed/       # Generated features
│
├── src/                 # Source code modules
│   ├── data.py          # Data ingestion and cleaning
│   ├── features.py      # Feature engineering
│   ├── model.py         # HybridEVTModel class
│   ├── train.py         # Walk-forward validation and forecasting
│   └── utils.py         # Plotting and config utilities
│
├── models/              # Exported model data and config (joblib)
│
├── notebooks/
│   ├── 00-EDA.ipynb                # ← Start here: data download + exploration
│   └── 01-Analysis_and_Modeling.ipynb  # Training, baselines, evaluation, export
│
├── 02-Gradio_Predictions.ipynb     # Interactive app (Toni's Predictor)
│
├── config.yaml          # Hyperparameters
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup & Usage

### 1. Create & Activate a Virtual Environment

```powershell
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Register the Jupyter Kernel (recommended)

```bash
python -m ipykernel install --user --name=sunspots --display-name "Sunspots Venv"
```

### 4. Run the Notebooks in Order

| Step | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/00-EDA.ipynb` | Downloads data from SILSO, explores distributions, autocorrelation, solar cycles, and baseline comparisons |
| 2 | `notebooks/01-Analysis_and_Modeling.ipynb` | Trains the hybrid model, compares against baselines (including ARIMA and McNish-Lincoln), exports artefacts to `models/` |
| 3 | `02-Gradio_Predictions.ipynb` | Launches the interactive Gradio app using the exported artefacts |

> `data/raw/sunspots.csv` is cached after the first download — subsequent runs skip re-downloading automatically.

## Methodology

### Feature Engineering

| Feature group | Details |
|---|---|
| Auto-regressive lags | t-1, t-2, t-3, t-4, t-5, t-7, t-30, t-365, t-4015 (log scale) |
| Rolling statistics | Mean, std, min, max over windows of 7, 30, 365, 4015 days |
| Momentum | `lag_1 − lag_30` (short vs. long-term trend direction) |
| Annual seasonality | sin/cos encoding of day-of-year |
| Solar cycle phase | sin/cos anchored to the 2008-12 solar minimum (~4015-day cycle) |

### Validation Strategy

**Expanding Window Walk-Forward** — no data leakage:
- Initial training window: 11 years (~4015 days)
- Step size: 90 days (quarterly advances)
- Forecast horizon: 5 days ahead (direct multi-step)

### Data

- **Source:** [SILSO](https://www.sidc.be/silso/datafiles) (Sunspot Index and Long-term Solar Observations, Belgium)
- **Preprocessing:** `log(1+x)` transformation to stabilise variance on the heavy-tailed distribution

### Reproducibility

Fixed **Random State 7** (Toni's lucky number) in `config.yaml`.

## Authors
[Jordi Roselló / Toni Majà]
