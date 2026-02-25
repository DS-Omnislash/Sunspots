
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import genpareto, genextreme as gev
from src.features import build_residual_lags

def evt_tail_correction(forecast, actual, upper_q=80, lower_q=20):
    residuals = actual - forecast
    if len(residuals) == 0:
        return forecast

    upper_thresh = np.percentile(residuals, upper_q)
    lower_thresh = np.percentile(residuals, lower_q)

    upper_excess = residuals[residuals > upper_thresh] - upper_thresh
    lower_excess = -(residuals[residuals < lower_thresh] - lower_thresh)

    E_upper = 0.0
    E_lower = 0.0

    if len(upper_excess) > 0:
        shape_u, _, scale_u = genpareto.fit(upper_excess, floc=0)
        if shape_u < 1:
            E_upper = scale_u / (1 - shape_u)

    if len(lower_excess) > 0:
        shape_l, _, scale_l = genpareto.fit(lower_excess, floc=0)
        if shape_l < 1:
            E_lower = scale_l / (1 - shape_l)

    corrected = forecast.copy()
    corrected[residuals > upper_thresh] += E_upper
    corrected[residuals < lower_thresh] -= E_lower
    return corrected

class HybridEVTModel:
    def __init__(self, ridge_alpha=1.0, resid_model_type='xgb', resid_params=None, max_lag_resid=5):
        self.ridge = Ridge(alpha=ridge_alpha)
        self.max_lag_resid = max_lag_resid
        self.resid_model_type = resid_model_type
        
        default_xgb = {
            'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 
            'verbosity': 0, 'n_jobs': -1
        }
        default_lgb = {
            'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 
            'verbosity': -1, 'n_jobs': -1
        }
        
        if resid_model_type == 'xgb':
            self.resid_params = resid_params if resid_params else default_xgb
            self.resid_model = XGBRegressor(**self.resid_params)
        else:
            self.resid_params = resid_params if resid_params else default_lgb
            self.resid_model = LGBMRegressor(**self.resid_params)
            
    def fit_predict_val(self, X_train, y_train, X_val):
        # 1. Ridge
        self.ridge.fit(X_train, y_train)
        pred_train_log = self.ridge.predict(X_train)
        pred_val_log = self.ridge.predict(X_val)
        
        # 2. Residuals (on log scale or original? Original usually works better for EVT)
        # But here the ridge is on log target. The residuals are calc on linear scale usually in the notebook.
        
        # Let's stick to the notebook logic: 
        # ridge trained on LOG.
        # residuals = exp(y_train) - exp(pred_train_log)
        
        pred_train_linear = np.expm1(pred_train_log)
        y_train_linear = np.expm1(y_train)
        residuals_train = y_train_linear - pred_train_linear
        
        # 3. Residual Lags
        resid_df = build_residual_lags(residuals_train, max_lag=self.max_lag_resid)
        
        if len(resid_df) == 0:
            return np.expm1(pred_val_log) # Fallback to ridge

        X_resid = resid_df.drop(columns=['resid'])
        y_resid = resid_df['resid']
        
        self.resid_model.fit(X_resid, y_resid)
        
        # 4. Recursive prediction for validation
        start_resid_lags = list(residuals_train[-self.max_lag_resid:])
        # Pad if not enough
        if len(start_resid_lags) < self.max_lag_resid:
             start_resid_lags = [0]*(self.max_lag_resid - len(start_resid_lags)) + start_resid_lags

        resid_preds = []
        current_lags = start_resid_lags
        
        for _ in range(len(X_val)):
            # Create feature row: lag_1 is last, lag_2 is second last...
            input_feat = {f'resid_lag_{k}': current_lags[-k] for k in range(1, self.max_lag_resid + 1)}
            input_df = pd.DataFrame([input_feat])
            
            pred_res = self.resid_model.predict(input_df)[0]
            resid_preds.append(pred_res)
            current_lags.append(pred_res)
            
        resid_preds = np.array(resid_preds)
        
        # 5. Combine
        hybrid_pred = np.expm1(pred_val_log) + resid_preds
        
        return hybrid_pred
