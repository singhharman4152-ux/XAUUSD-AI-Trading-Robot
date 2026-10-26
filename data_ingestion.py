import pandas as pd
import MetaTrader5 as mt5
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOL, TIMEFRAME, DATA_PATH, LIVE_DATA_PATH
import os

def load_historical_data(file_path=DATA_PATH):
    """Load historical data from CSV or fetch from MT5 if not found."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            if len(df.columns) == 1:
                with open(file_path, 'r', encoding='utf-8') as f:
                    header = f.readline().strip().split('\t')
                df = pd.read_csv(file_path, sep='\t', skiprows=1, names=header, encoding='utf-8')
            print("Successfully loaded historical data from file.")
        except Exception as e:
            print(f"Error reading historical data: {e}")
            return None
    else:
        print(f"Historical data file not found at {file_path}. Fetching from MT5...")
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return None
        
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 1000)
        if rates is None or len(rates) == 0:
            print(f"Failed to fetch historical data: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, sep='\t', index=False)
        print(f"Saved historical data to {file_path}")

    df.columns = [col.strip().replace('<', '').replace('>', '') for col in df.columns]
    column_map = {'DATE': 'date', 'TIME': 'time', 'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 
                  'CLOSE': 'close', 'TICKVOL': 'volume', 'VOL': 'volume', 'SPREAD': 'spread'}
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    if 'volume' in df.columns:
        volume_cols = [col for col in df.columns if col == 'volume']
        if len(volume_cols) > 1:
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    if 'date' in df.columns and 'time' in df.columns:
        df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], utc=True)
        df = df.drop(columns=['date'])
    
    if 'time' not in df.columns:
        raise KeyError(f"'time' column not found. Available columns: {df.columns.tolist()}")
    
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], utc=True)
    
    required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in required_columns if col in df.columns]]
    return df

def fetch_live_data():
    """Fetch real-time data from MT5."""
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None
    
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 50)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch live data: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
    df.tail(1).to_csv(LIVE_DATA_PATH, mode='a', header=not os.path.exists(LIVE_DATA_PATH), index=False)
    print(f"Fetched {len(df)} rows of live data")
    return df

if __name__ == "__main__":
    historical = load_historical_data()
    live = fetch_live_data()
    print("Historical:\n", historical.tail() if historical is not None else "No historical data")
    print("Live:\n", live.tail() if live is not None else "No live data")