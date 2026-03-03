
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model import HybridEVTModel, evt_tail_correction

def expanding_walk_forward_splits(X, y, initial_train_size, val_size, step_size):
    start_val_idx = initial_train_size
    while start_val_idx + val_size <= len(X):
        train_idx = slice(0, start_val_idx)
        val_idx = slice(start_val_idx, start_val_idx + val_size)
        yield X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx]
        start_val_idx += step_size

def train_evaluate(X, y, config):
    initial_train_size = config['evaluation']['initial_train_size']
    val_size = config['evaluation']['val_size']
    step_size = config['evaluation']['step_size']
    
    model_params = config['model']
    
    hybrid_rmse, hybrid_mae = [], []
    all_preds = []
    all_actuals = []
    
    resid_model_type = model_params.get('resid_model_type', 'xgb')
    resid_params_key = 'lgb_params' if resid_model_type == 'lgb' else 'xgb_params'

    model = HybridEVTModel(
        ridge_alpha=model_params['ridge_alpha'],
        resid_model_type=resid_model_type,
        resid_params=model_params[resid_params_key],
        max_lag_resid=config['features']['max_lag_resid']
    )

    for i, (Xtr, ytr, Xval, yval) in enumerate(expanding_walk_forward_splits(X, y, initial_train_size, val_size, step_size)):
        if len(Xtr) < 365: continue

        # ytr and yval are LOG transformed target
        hybrid_pred = model.fit_predict_val(Xtr, ytr, Xval)

        yval_linear = np.expm1(yval)

        # EVT tail correction using training residuals
        hybrid_pred = evt_tail_correction(hybrid_pred, yval_linear.values)

        # Collect for metrics and plotting
        hybrid_rmse.append(np.sqrt(mean_squared_error(yval_linear, hybrid_pred)))
        hybrid_mae.append(mean_absolute_error(yval_linear, hybrid_pred))
        
        all_preds.extend(hybrid_pred)
        all_actuals.extend(yval_linear.values)
        
    return {
        "hybrid_rmse": np.mean(hybrid_rmse),
        "hybrid_mae": np.mean(hybrid_mae),
        "predictions": np.array(all_preds),
        "actuals": np.array(all_actuals)
    }

def run_future_forecast(df, steps, config):
    """
    Train on all available data and forecast 'steps' into the future using a Direct Multi-step approach.
    Returns a dictionary of {Timestamp: prediction}.
    """
    from src.features import create_features
    
    # 1. Prepare base features
    df_feat = create_features(df, config['features']['lags'], config['features']['rolling_windows'])
    X_all = df_feat.drop(columns=['SUNSPOTS', 'LOG_SUNSPOTS'])
    X_last = X_all.iloc[-1:].copy()
    
    last_date = df.index.max()
    model_params = config['model']
    predictions = {}

    # 2. Iterate through each step ahead
    for s in range(1, steps + 1):
        df_train = df_feat.copy()
        # Direct forecasting: predict t+s using features at t
        df_train['target'] = df_train['LOG_SUNSPOTS'].shift(-s)
        df_train = df_train.dropna()
        
        X_train = df_train.drop(columns=['target', 'SUNSPOTS', 'LOG_SUNSPOTS'])
        y_train = df_train['target']
        
        resid_model_type = model_params.get('resid_model_type', 'xgb')
        resid_params_key = 'lgb_params' if resid_model_type == 'lgb' else 'xgb_params'
        model = HybridEVTModel(
            ridge_alpha=model_params['ridge_alpha'],
            resid_model_type=resid_model_type,
            resid_params=model_params[resid_params_key],
            max_lag_resid=config['features']['max_lag_resid']
        )
        
        # Fit and predict for the last available features
        pred = model.fit_predict_val(X_train, y_train, X_last)
        
        forecast_date = last_date + pd.Timedelta(days=s)
        predictions[forecast_date] = pred[0]
        
    return predictions


if __name__ == "__main__":
    pass
