import time
import logging
from data_ingestion import load_historical_data, fetch_live_data
from feature_engineering import engineer_features
from hybrid_predictor import hybrid_predict
from mt5_integration import execute_trade, update_trailing_sl
from real_time_charting import update_chart
from config import SLEEP_INTERVAL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_trading_robot():
    logger.info("Starting trading robot...")
    
    historical_df = load_historical_data()
    if historical_df is None:
        logger.error("Failed to load historical data.")
        historical_df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    historical_df = engineer_features(historical_df)
    if historical_df is None or historical_df.empty:
        logger.warning("Feature engineering failed.")
    
    # Trading state
    balance = 10000
    position = 0
    trailing_distance = 5.0
    
    while True:
        live_df = fetch_live_data()
        if live_df is None:
            logger.warning("No live data. Using historical.")
            live_df = historical_df.tail(50) if not historical_df.empty else pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        live_df = engineer_features(live_df)
        if live_df is not None and not live_df.empty:
            logger.info(f"Live data timestamp: {live_df['time'].iloc[-1]}")
            logger.info(f"Live features: {live_df.iloc[-1]}")
            
            # Update balance from MT5
            from mt5_integration import mt5
            if mt5.initialize():
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
            
            probs, action, trailing_distance = hybrid_predict(live_df.iloc[-1], balance, position, trailing_distance)
            logger.info(f"Predicted: Hold={probs[0]:.2f}, Buy={probs[1]:.2f}, Sell={probs[2]:.2f}, Action={action}, Trailing Distance={trailing_distance}")
            
            if action == 1:
                logger.info("Executing BUY order")
                execute_trade("up", trailing_distance)
                position = 1
            elif action == 2:
                logger.info("Executing SELL order")
                execute_trade("down", trailing_distance)
                position = -1
            else:
                logger.info("Holding position")
                update_trailing_sl()  # Automatic trailing SL
                position = 0 if not mt5.positions_total() else position
            
            update_chart(live_df.tail(50), probs)
        else:
            logger.warning("Failed to engineer live data features.")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    run_trading_robot()