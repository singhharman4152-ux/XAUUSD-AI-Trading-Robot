import xgboost as xgb
import os
from config import MODEL_PATH_XGB
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_xgboost_model(X, y):
    """Train an XGBoost model for three-class prediction and save it."""
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # Multi-class classification
        num_class=3,               # 0: Hold, 1: Buy, 2: Sell
        eval_metric='mlogloss',    # Multi-class log loss
        use_label_encoder=False
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH_XGB), exist_ok=True)
    model.save_model(MODEL_PATH_XGB)
    logger.info(f"XGBoost model saved to {MODEL_PATH_XGB}")
    return model

def load_xgboost_model():
    """Load the trained XGBoost model or return None if not found."""
    model = xgb.XGBClassifier()
    if os.path.exists(MODEL_PATH_XGB):
        model.load_model(MODEL_PATH_XGB)
        logger.info(f"Loaded XGBoost model from {MODEL_PATH_XGB}")
    else:
        logger.warning(f"XGBoost model file {MODEL_PATH_XGB} not found.")
    return model

if __name__ == "__main__":
    from data_ingestion import load_historical_data
    from feature_engineering import engineer_features
    df = load_historical_data()
    if df is not None:
        df = engineer_features(df)
        if df is not None:
            X = df.drop(columns=['time', 'direction'])
            y = df['direction']  # Ensure y has values 0, 1, 2
            model = train_xgboost_model(X, y)