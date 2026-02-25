
# Sunspot Forecasting: Hybrid Ridge + XGBoost + EVT

## Overview

This project implements an advanced time-series forecasting pipeline to predict **Daily International Sunspot Numbers**. The goal is to accurately model the 11-year solar cycle while improving the detection of extreme solar activity events, which are critical for space weather monitoring.

To achieve this, we employ a **Hybrid Machine Learning Architecture** that decomposes the problem into three distinct layers:

1.  **Trend & Seasonality (Linear Baseline)**: 
    *   **Ridge Regression** is used to capture the broad, auto-regressive trends and the dominant periodicities of the time series.
    
2.  **Non-Linear Residual Correction**: 
    *   **Gradient Boosting (XGBoost/LightGBM)** is trained on the *residuals* of the linear model. It learns the complex, short-term non-linear dynamics that the linear model misses.
    
3.  **Tail Calibration (Extreme Value Theory)**:
    *   We apply **EVT (Peaks-Over-Threshold)** to model the tail distribution of the forecast errors, adjusting predictions for extreme solar events.

## Project Structure

The codebase is organized to separate data processing, modeling, and experimentation:

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

### 2. Training the Model
Open `notebooks/01-Analysis_and_Modeling.ipynb` to:
1.  Load data and engineer features.
2.  Evaluate the hybrid model using walk-forward validation.
3.  **Export the model data**: The last cell saves the required data to the `models/` folder.

### 3. Interactive Predictions (Toni's Predictor)
Run `02-Gradio_Predictions.ipynb` to launch an interactive Gradio web interface. You can:
- Select the number of future days to forecast.
- View a table of predicted sunspot counts.
- See a trend plot combining recent history and future forecasts.

## Methodology

### Reproducibility
The model is configured with a fixed **Random State of 7** (Toni's lucky number) in `config.yaml` to ensure consistent and reproducible results across different runs.

### Data Handling
- **Source**: SILSO (Sunspot Index and Long-term Solar Observations).
- **Preprocessing**: Log-transformation `log(1+x)` is applied to stabilize variance.

### Validation Strategy
We use **Expanding Window Walk-Forward Validation** to strictly prevent data leakage.
- **Initial Training**: 11 years (approx. 4015 days).
- **Validation Horizon**: Multi-day forecasts.

## Authors
[Jordi Roselló / Toni Majà]
