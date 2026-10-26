import os
import pandas as pd
from config import MODEL_PATH_RL, DATA_PATH
from gymnasium import Env, spaces
from stable_baselines3 import PPO
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnv(Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        feature_cols = [col for col in self.df.columns if col not in ['time', 'direction']]
        self.df[feature_cols] = self.df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.trailing_distance = 5.0
        
        self.action_space = spaces.MultiDiscrete([3, 5])  # trade_action (0-2), trail_adjust (0-4)
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'), 
            shape=(len(feature_cols) + 3,),  # 16 features + 3 state variables
            dtype=np.float32
        )
        logger.info(f"Initialized observation space: {self.observation_space}")
    
    def reset(self, seed=None, **kwargs):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.trailing_distance = 5.0
        obs = self._get_observation()
        info = {"balance": self.balance, "position": self.position, "trailing_distance": self.trailing_distance}
        return obs, info
    
    def step(self, action):
        trade_action = action[0]
        trail_adjust = action[1] - 2  # -2 to +2
        current_price = self.df['close'].iloc[self.current_step]
        next_price = self.df['close'].iloc[self.current_step + 1] if self.current_step + 1 < len(self.df) else current_price
        
        reward = 0
        price_change = next_price - current_price
        
        self.trailing_distance = max(1.0, self.trailing_distance + trail_adjust)
        
        if self.position == 1 and next_price <= self.sl_price:
            reward = (self.sl_price - self.entry_price) * 100
            self.balance += reward
            self.position = 0
        elif self.position == -1 and next_price >= self.sl_price:
            reward = (self.entry_price - self.sl_price) * 100
            self.balance += reward
            self.position = 0
        
        if self.position == 0:
            if trade_action == 1:
                self.position = 1
                self.entry_price = current_price
                self.sl_price = current_price - self.trailing_distance
            elif trade_action == 2:
                self.position = -1
                self.entry_price = current_price
                self.sl_price = current_price + self.trailing_distance
        
        if self.position == 1 and next_price > self.entry_price:
            new_sl = next_price - self.trailing_distance
            self.sl_price = max(self.sl_price, new_sl)
            reward += price_change * 100
        elif self.position == -1 and next_price < self.entry_price:
            new_sl = next_price + self.trailing_distance
            self.sl_price = min(self.sl_price, new_sl)
            reward += abs(price_change) * 100
        
        self.balance += reward
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        obs = self._get_observation() if not terminated else self.reset()[0]
        info = {"balance": self.balance, "position": self.position, "trailing_distance": self.trailing_distance}
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        base_obs = self.df.iloc[self.current_step].drop(['time', 'direction']).values
        base_obs = np.array(base_obs, dtype=np.float32)
        return np.concatenate([base_obs, [self.balance, self.position, self.trailing_distance]], dtype=np.float32)

def train_rl_model(df):
    env = TradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    model.learn(total_timesteps=100000)
    os.makedirs(os.path.dirname(MODEL_PATH_RL), exist_ok=True)
    model.save(MODEL_PATH_RL)
    logger.info(f"RL model saved to {MODEL_PATH_RL}")
    return model

def load_rl_model():
    logger.info(f"Checking for RL model at: {MODEL_PATH_RL}")
    if os.path.exists(MODEL_PATH_RL):
        try:
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'sma_20', 'bb_upper', 'bb_lower', 'atr', 'volume_delta',
                'hour', 'minute', 'lag1', 'lag2'
            ]
            df_dummy = pd.DataFrame(columns=['time'] + feature_columns + ['direction'])
            env = TradingEnv(df_dummy)
            logger.info(f"Environment observation space: {env.observation_space}")
            model = PPO.load(MODEL_PATH_RL, env=env)
            logger.info(f"Loaded RL model from {MODEL_PATH_RL}")
            return model
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return None
    else:
        logger.warning(f"RL model file {MODEL_PATH_RL} not found.")
        return None

if __name__ == "__main__":
    from data_ingestion import load_historical_data
    from feature_engineering import engineer_features
    df = load_historical_data()
    if df is not None:
        df = engineer_features(df)
        if df is not None:
            train_rl_model(df)  # Retrain with 19 features
            model = load_rl_model()
            if model is not None:
                logger.info("RL model loaded successfully")
            else:
                logger.error("Failed to load RL model")