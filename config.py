MT5_LOGIN = 
MT5_PASSWORD = ""
MT5_SERVER = ""
SYMBOL = "XAUUSDm"
TIMEFRAME = "M5"  # 5-minute timeframe
DATA_PATH = r"C:\Users\victory\user1.0\xegold3\xe\src\data\historical_xauusd.csv"
LIVE_DATA_PATH = r"C:\Users\victory\user1.0\xegold3\xe\src\data\live_xauusd.csv"
SLEEP_INTERVAL = 300  # 5 minutes
MODEL_PATH_XGB = r"C:\Users\victory\user1.0\xegold3\xe\src\models\xgboost_model.pkl"
LOG_PATH_TRADE = r"C:\Users\victory\user1.0\xegold3\xe\src\logs\trade_log.txt"
MODEL_PATH_RL = r"C:\Users\victory\user1.0\xegold3\xe\src\models\rl_model.pkl"

# New risk management parameters
MAX_RISK_PER_TRADE = 5.0  # 1% of account balance per trade
MAX_DRAWDOWN = 0.10  # 10% max drawdown
INITIAL_SL_DEFAULT = 5000  # Default initial SL in points
MIN_TRAIL_DISTANCE = 5000  # Minimum trailing SL distance
RETRAIN_INTERVAL = 24 * 3600  # Retrain every 24 hours (seconds)