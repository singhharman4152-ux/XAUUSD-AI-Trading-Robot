import MetaTrader5 as mt5
import os
import pandas as pd
import time
from config import LOG_PATH_TRADE, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOL
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

open_position = {"order": None, "position_ticket": None, "type": None, "entry_price": 0, "sl_price": 0, "trailing_distance": 5.0}

def execute_trade(prediction, trailing_distance):
    """Execute a trade with automatic trailing SL."""
    global open_position
    
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    if prediction.lower() not in ["up", "down"]:
        logger.error(f"Invalid prediction: {prediction}")
        return
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to retrieve account info: {mt5.last_error()}")
        return
    logger.info(f"Current account balance: {account_info.balance}")
    
    if open_position["position_ticket"] and ((prediction.lower() == "up" and open_position["type"] == "sell") or 
                                              (prediction.lower() == "down" and open_position["type"] == "buy")):
        close_position(open_position["position_ticket"])
    
    if not open_position["position_ticket"]:
        action = 1 if prediction.lower() == "up" else 0
        lot = 0.10
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error(f"Failed to get tick data: {mt5.last_error()}")
            return
        
        price = tick.ask if action == 1 else tick.bid
        sl_price = price - trailing_distance if action == 1 else price + trailing_distance
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Automated trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trade executed: {'Buy' if action == 1 else 'Sell'} order {result.order}")
            time.sleep(0.5)
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions:
                position = positions[0]
                open_position.update({
                    "order": result.order,
                    "position_ticket": position.ticket,
                    "type": "buy" if action == 1 else "sell",
                    "entry_price": price,
                    "sl_price": sl_price,
                    "trailing_distance": trailing_distance
                })
                log_trade(result.order, "open", price, sl_price)
            else:
                logger.error("No position found after execution")
        else:
            logger.error(f"Trade failed: {result.comment if result else 'None'}")
    
    update_trailing_sl()  # Automatic trailing SL

def update_trailing_sl():
    """Update trailing SL automatically."""
    global open_position
    if not open_position["position_ticket"]:
        return
    
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        logger.error(f"Failed to update SL: No tick data")
        return
    
    current_price = tick.ask if open_position["type"] == "buy" else tick.bid
    new_sl = current_price - open_position["trailing_distance"] if open_position["type"] == "buy" else current_price + open_position["trailing_distance"]
    
    if open_position["type"] == "buy" and new_sl > open_position["sl_price"]:
        modify_position(open_position["position_ticket"], new_sl)
    elif open_position["type"] == "sell" and new_sl < open_position["sl_price"]:
        modify_position(open_position["position_ticket"], new_sl)

def modify_position(position_ticket, new_sl):
    """Modify SL of an open position."""
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position_ticket,
        "symbol": SYMBOL,
        "sl": new_sl,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"SL updated for position {position_ticket} to {new_sl}")
        open_position["sl_price"] = new_sl
        log_trade(open_position["order"], "modify", open_position["entry_price"], new_sl)
    else:
        logger.error(f"SL update failed: {result.comment if result else 'Position doesn’t exist'}")

def close_position(position_ticket):
    """Close an open position."""
    global open_position
    if not position_ticket:
        return
    
    position = mt5.positions_get(ticket=position_ticket)
    if not position:
        logger.warning(f"Position {position_ticket} not found")
        open_position["position_ticket"] = None
        return
    
    position = position[0]
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(SYMBOL).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "Close trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Position {position_ticket} closed")
        log_trade(open_position["order"], "close", open_position["entry_price"], open_position["sl_price"])
        open_position.update({"order": None, "position_ticket": None, "type": None, "entry_price": 0, "sl_price": 0})
    else:
        logger.error(f"Close failed: {result.comment if result else 'None'}")

def log_trade(order, action, price, sl):
    os.makedirs(os.path.dirname(LOG_PATH_TRADE), exist_ok=True)
    with open(LOG_PATH_TRADE, "a") as f:
        f.write(f"{action.capitalize()}: Order {order}, Type: {open_position['type']}, Price: {price}, SL: {sl}, Time: {pd.Timestamp.now(tz='UTC')}\n")

if __name__ == "__main__":
    execute_trade("up", 5.0)
    execute_trade("down", 5.0)