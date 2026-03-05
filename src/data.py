
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
            df['SUNSPOTS'] = df['SUNSPOTS'].clip(lower=0)
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


def load_solar_flux(url, save_path=None):
    """
    Loads the F10.7 solar flux index (Penticton adjusted) from NOAA NGDC.
    Returns a DataFrame with a DatetimeIndex and a single column 'F10.7'.
    Falls back to local cache if available.
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading solar flux from local file: {save_path}")
        return pd.read_csv(save_path, index_col=0, parse_dates=True)

    print(f"Downloading solar flux from {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    lines = r.text.splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith(':'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            # Date is YYYYMMDD or YYYYMMDD.0 in parts[0]
            date_int = int(float(parts[0]))
            date_str = str(date_int)
            if len(date_str) != 8:
                continue
            year  = int(date_str[:4])
            month = int(date_str[4:6])
            day   = int(date_str[6:8])

            # Try each remaining column for a valid flux value (50–500 SFU)
            for candidate in parts[1:]:
                val = float(candidate)
                if 50 <= val <= 500:
                    rows.append((pd.Timestamp(year, month, day), val))
                    break
        except (ValueError, IndexError):
            continue

    if not rows:
        print("load_solar_flux: could not parse any rows. First 20 lines of response:")
        for l in lines[:20]:
            print(repr(l))
        raise ValueError("load_solar_flux: zero rows parsed — see printed output above for format debugging")

    df = pd.DataFrame(rows, columns=['Date', 'F10.7']).set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    print(f"Solar flux loaded: {len(df):,} days  ({df.index.min().date()} → {df.index.max().date()})")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Solar flux saved to {save_path}")

    return df
