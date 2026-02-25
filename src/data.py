
import pandas as pd
import requests
from io import StringIO
import os
import numpy as np

def load_data(url, save_path=None):
    """
    Loads sunspot data from the given URL, preprocesses it, and optionally saves it.
    """
    try:
        # Check if local file exists first to avoid re-downloading
        if save_path and os.path.exists(save_path):
            print(f"Loading data from local file: {save_path}")
            df = pd.read_csv(save_path, index_col=0, parse_dates=True)
            return df

        print(f"Downloading data from {url}...")
        r = requests.get(url)
        r.raise_for_status()
        
        print("Processing data...")
        df = pd.read_csv(StringIO(r.text), sep=';', header=None, comment='#',
                         names=['Year', 'Month', 'Day', 'Decimal_Date', 'Ri', 'Std', 'Num_obs', 'Definitive'])

        df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
        df = df.set_index('Date')
        df = df[['Ri']].rename(columns={'Ri': 'SUNSPOTS'})
        
        # Data cleaning: sunspots cannot be negative
        df['SUNSPOTS'] = df['SUNSPOTS'].clip(lower=0)
        
        # log transform is common
        df['LOG_SUNSPOTS'] = np.log1p(df['SUNSPOTS'])

        df = df.sort_index()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)
            print(f"Data saved to {save_path}")
            
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise
