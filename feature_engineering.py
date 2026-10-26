import pandas as pd
import pandas_ta as ta

def engineer_features(df):
    required_cols = ['close', 'high', 'low', 'volume', 'time']
    if not all(col in df.columns for col in required_cols):
        print("Error: Required columns missing from DataFrame.")
        return None
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    print(f"DataFrame shape before features: {df.shape}")
    
    if df['close'].isna().sum() > 0:
        print(f"NaN values in 'close': {df['close'].isna().sum()}")
        df['close'].fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
    
    min_len = 26
    if len(df) < min_len:
        print(f"Warning: Only {len(df)} rows available, some features will be limited.")
    
    df['rsi'] = ta.rsi(df['close'], length=14).fillna(0) if len(df) >= 14 else 0
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9'].fillna(0) if len(df) >= min_len else 0
    df['sma_20'] = ta.sma(df['close'], length=20).fillna(0) if len(df) >= 20 else 0
    df['bb_upper'] = ta.bbands(df['close'], length=20, std=2)['BBU_20_2.0'].fillna(0) if len(df) >= 20 else 0
    df['bb_lower'] = ta.bbands(df['close'], length=20, std=2)['BBL_20_2.0'].fillna(0) if len(df) >= 20 else 0
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(0) if len(df) >= 14 else 0
    df['volume_delta'] = df['volume'].diff().fillna(0).clip(lower=-1000, upper=1000)  # Cap volume_delta
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['lag1'] = df['close'].shift(1).fillna(0) if len(df) >= 2 else 0
    df['lag2'] = df['close'].shift(2).fillna(0) if len(df) >= 3 else 0
    
    df['direction'] = 0
    df.loc[df['close'].shift(-1) > df['close'], 'direction'] = 1
    df.loc[df['close'].shift(-1) < df['close'], 'direction'] = 2
    df['direction'] = df['direction'].fillna(0)
    
    print(f"Engineered DataFrame dtypes: {df.dtypes}")
    return df.dropna()

if __name__ == "__main__":
    from data_ingestion import load_historical_data
    df = load_historical_data()
    if df is not None:
        df = engineer_features(df)
        if df is not None:
            print(df.tail())