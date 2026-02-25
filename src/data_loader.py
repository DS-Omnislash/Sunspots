
import pandas as pd
import requests
from io import StringIO
import os

def load_sunspot_data(url="https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv", save_path=None):
    """
    Loads sunspot data from the given URL, preprocesses it, and optionally saves it.
    
    Args:
        url (str): The URL to fetch the data from.
        save_path (str, optional): Path to save the processed CSV.

    Returns:
        pd.DataFrame: The processed DataFrame with 'Date' index and 'SUNSPOTS' column.
    """
    try:
        print(f"Downloading data from {url}...")
        r = requests.get(url)
        r.raise_for_status()
        
        print("Processing data...")
        # Fit in a df
        df = pd.read_csv(StringIO(r.text), sep=';', header=None, comment='#',
                         names=['Year', 'Month', 'Day', 'Decimal_Date', 'Ri', 'Std', 'Num_obs', 'Definitive'])

        # Convert to datetime
        df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
        df = df.set_index('Date')

        # Select only sunspots column
        df = df[['Ri']].rename(columns={'Ri': 'SUNSPOTS'})

        # Order by date
        df = df.sort_index()
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)
            print(f"Data saved to {save_path}")
            
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    df = load_sunspot_data(save_path=os.path.join("..", "data", "sunspots.csv"))
    print(df.head())
