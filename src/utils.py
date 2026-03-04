
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else np.nan

def plot_sunspots_series(df, col='SUNSPOTS', start_date=None, end_date=None, title="Daily International Sunspot Number", figsize=(14, 4), lw=0.5, save_path=None):
    """
    Plots the sunspots time series.
    """
    if start_date or end_date:
        plot_df = df.loc[start_date:end_date][col]
    else:
        plot_df = df[col]

    plt.figure(figsize=figsize)
    plt.plot(plot_df.index, plot_df, lw=lw)
    plt.title(title)
    plt.ylabel("Sunspot Count")
    plt.xlabel("Year")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_predictions(y_true, y_pred, title="Model Predictions vs Actual", figsize=(14, 4), save_path=None):
    """
    Plots true values vs predicted values.
    """
    plt.figure(figsize=figsize)
    plt.plot(y_true, label="Actual", color="blue", alpha=0.7)
    plt.plot(y_pred, label="Predicted", color="red", alpha=0.7, linestyle="--")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
