import os
import numpy as np
import pandas as pd
from xgboost_training import load_xgboost_model
from rl_training import load_rl_model
from config import MODEL_PATH_XGB, MODEL_PATH_RL
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hybrid_predict(data, balance, position, trailing_distance):
    logger.info(f"Input: {data}, Balance: {balance}, Position: {position}, Trailing Distance: {trailing_distance}")
    xgboost_model = load_xgboost_model()
    rl_model = load_rl_model()
    
    features = data.drop(labels=['time', 'direction'], errors='ignore')
    features = pd.to_numeric(features, errors='coerce').fillna(0)
    logger.info(f"Features after dropping and converting: {features}")
    logger.info(f"Features dtypes: {features.dtype}")
    features_np = features.to_numpy(dtype=np.float32)
    full_features_np = np.concatenate([features_np, [balance, position, trailing_distance]], dtype=np.float32)
    logger.info(f"Full features for RL: {full_features_np}")
    
    # XGBoost (assuming binary for now; update to 3-class if retrained)
    if os.path.exists(MODEL_PATH_XGB):
        xgb_prob = xgboost_model.predict_proba([features_np])[0]
        xgb_action = xgboost_model.predict([features_np])[0]
        logger.info(f"XGBoost: Prob={xgb_prob}, Action={xgb_action}")
        if len(xgb_prob) == 2:  # Binary to 3-class workaround
            xgb_prob = [1 - (xgb_prob[0] + xgb_prob[1]), xgb_prob[0], xgb_prob[1]]
            xgb_action = 1 if xgb_action == 0 else 2 if xgb_action == 1 else 0
    else:
        xgb_prob = [0.33, 0.33, 0.34]
        xgb_action = 0
    
    # RL Prediction
    if rl_model is not None:
        try:
            rl_action_tuple = rl_model.predict(full_features_np)[0]
            rl_action, trail_adjust = rl_action_tuple[0], rl_action_tuple[1] - 2
            rl_prob = [0.33, 0.33, 0.34]  # Placeholder
            logger.info(f"RL: Action={rl_action}, Trail Adjust={trail_adjust}, Prob={rl_prob}")
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            rl_action, trail_adjust = 0, 0
            rl_prob = [0.33, 0.33, 0.34]
    else:
        rl_action, trail_adjust = 0, 0
        rl_prob = [0.33, 0.33, 0.34]
    
    combined_prob = [(x + y) / 2 for x, y in zip(xgb_prob, rl_prob)]
    combined_action = rl_action if rl_model is not None else xgb_action
    new_trailing_distance = max(1.0, trailing_distance + trail_adjust)
    
    logger.info(f"Combined: Prob={combined_prob}, Action={combined_action}, Trailing Distance={new_trailing_distance}")
    return combined_prob, combined_action, new_trailing_distance

if __name__ == "__main__":
    dummy_data = pd.Series({
        'time': '2025-03-19', 'open': 3031, 'high': 3032, 'low': 3029, 
        'close': 3030, 'volume': 721, 'rsi': 47, 'macd': -1.2, 'sma_20': 3029,
        'bb_upper': 3032, 'bb_lower': 3026, 'atr': 2.5, 'volume_delta': 10,
        'hour': 15, 'minute': 15, 'lag1': 3031, 'lag2': 3031
    })
    probs, action, trail = hybrid_predict(dummy_data, 10000, 0, 5.0)
    logger.info(f"Probs: {probs}, Action: {action}, Trailing Distance: {trail}")