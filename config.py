"""
ftmo_agent/config.py — Multi-symbol configuration V4.
Each symbol has its own pip size, digits, contract size, and volatility profile.
Now includes variable spreads, slippage, commission, and curriculum learning.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class SymbolSpec:
    symbol: str
    pip_size: float           # 1 pip in price terms
    digits: int               # MT5 digits
    contract_size: float      # 1 lot = X units
    pip_value_per_lot: float  # $ per pip per lot
    min_volume: float = 0.01
    max_volume: float = 0.5
    spread_points_mean: float = 20   # mean spread in points
    spread_points_std: float = 5     # std of spread variation
    spread_news_mult: float = 5.0    # multiplier during news/volatile
    volatility_pct: float = 0.01     # daily volatility estimate
    commission_per_lot: float = 7.0  # $7 per standard lot (MT5 typical)
    slippage_pct_mean: float = 0.3   # 30% of spread as typical slippage
    slippage_pct_std: float = 0.2    # variability of slippage

SYMBOLS: dict[str, SymbolSpec] = {
    "XAUUSD":    SymbolSpec("XAUUSD",    0.01,   2, 100,    1.0,  0.01, 1.0, 30, 10, 6.0, 0.015, 15.0, 0.4, 0.25),
    "EURUSD":    SymbolSpec("EURUSD",    0.0001, 5, 100000, 10.0,  0.01, 1.0, 12,  4, 4.0, 0.006, 7.0,  0.3, 0.20),
    "GBPUSD":    SymbolSpec("GBPUSD",    0.0001, 5, 100000, 10.0,  0.01, 1.0, 27,  8, 5.0, 0.007, 7.0,  0.3, 0.20),
    "US30.cash": SymbolSpec("US30.cash", 1.0,    2, 1,      1.0,   0.01, 0.5, 263, 50, 5.0, 0.010, 3.0,  0.4, 0.25),
    "GER40.cash":SymbolSpec("GER40.cash", 1.0,   2, 1,      1.0,   0.01, 0.5, 203, 40, 5.0, 0.011, 3.0,  0.4, 0.25),
    "US500.cash":SymbolSpec("US500.cash", 0.1,   2, 1,      0.1,   0.01, 0.5, 63,  15, 4.0, 0.009, 3.0,  0.3, 0.20),
    "US100.cash":SymbolSpec("US100.cash", 1.0,   2, 1,      1.0,   0.01, 0.5, 213, 50, 5.0, 0.012, 3.0,  0.4, 0.25),
    "BTCUSD":    SymbolSpec("BTCUSD",    1.0,    2, 1,      1.0,   0.01, 0.3, 100, 30, 8.0, 0.030, 2.0,  0.5, 0.30),
}

# Active symbols for training and trading
ACTIVE_SYMBOLS = ["XAUUSD", "EURUSD", "US30.cash", "GER40.cash", "US500.cash", "US100.cash", "BTCUSD"]

# Timeframes (multi-TF feature extraction)
TIMEFRAMES = {
    "m15": {"bars": 48, "resample": None},
    "h1":  {"bars": 24, "resample": "4T"},
    "h4":  {"bars": 12, "resample": "16T"},
    "d1":  {"bars": 5,  "resample": "96T"},
}

# Action space
HOLD = 0
BUY = 1
SELL = 2
CLOSE = 3
SPLIT_BUY = 4
SPLIT_SELL = 5
PYRAMID = 6
PARTIAL_CLOSE = 7
N_ACTIONS = 8

ACTION_NAMES = ["HOLD", "BUY", "SELL", "CLOSE", "SPLIT_BUY", "SPLIT_SELL", "PYRAMID", "PARTIAL_CLOSE"]

# FTMO rules
FTMO_CONFIG = {
    "account_size": 10000,
    "daily_dd_limit": 0.05,
    "total_dd_limit": 0.10,
    "profit_target": 0.10,
    "max_trades_per_day": 8,
    "max_concurrent_positions": 3,
    "max_hold_bars": 48,           # V5: 12h max hold (was 96/24h)
    "min_hold_bars": 4,            # 1h min hold
    "cooldown_after_losses": 3,
    "cooldown_bars": 8,
}

# Risk management V4
RISK_CONFIG = {
    "risk_per_trade": 0.005,       # 0.5% risk per trade (was 1%)
    "max_risk_total": 0.02,       # 2% total risk (was 3%)
    "slbe_trigger_usd": 5.0,      # SLBE trigger at $5 (was $3)
    "slbe_offset_usd": 1.0,       # SLBE offset $1 (was $0.5)
    "pyramid_min_profit_usd": 10.0,
    "pyramid_risk_reduction": 0.5,
    "partial_close_trigger": 2.0,
    "partial_close_pct": 0.5,
    # TP/SL ratios
    "sl_atr_mult": 2.0,           # SL at 2 ATR (was 1.5)
    "tp_atr_mult": 4.0,           # TP at 4 ATR (was 3.0) — more breathing room
}

# Anti-bias
ANTI_BIAS_CONFIG = {
    "enabled": True,
    "penalty_weight": 1.5,         # reduced penalty (was 2.0)
    "min_trades_for_bias_check": 10,
    "max_buy_ratio": 0.65,
    "max_sell_ratio": 0.65,
}

# Curriculum learning V4
CURRICULUM_CONFIG = {
    "enabled": True,
    # Phase 1: no frictions, easy entry (0-200 episodes)
    "phase1_episodes": 200,
    "phase1_spread_mult": 0.0,     # no spread
    "phase1_slippage_mult": 0.0,   # no slippage
    "phase1_commission_mult": 0.0, # no commission
    "phase1_max_trades": 20,       # lots of attempts
    "phase1_exploration_bonus": 0.20,  # 20% bonus for opening trades
    "phase1_pnl_scale": 100.0,         # amplify PnL ×100 in phase 1
    # Phase 2: low frictions (200-500 episodes)
    "phase2_episodes": 300,
    "phase2_spread_mult": 0.3,     # 30% of real spread
    "phase2_slippage_mult": 0.3,
    "phase2_commission_mult": 0.0,
    "phase2_max_trades": 12,
    "phase2_exploration_bonus": 0.01,  # was 0.005, doublé
    "phase2_pnl_scale": 10.0,          # scaling réduit
    # Phase 3: full frictions (500+ episodes)
    "phase3_spread_mult": 1.0,
    "phase3_slippage_mult": 1.0,
    "phase3_commission_mult": 1.0,
    "phase3_max_trades": 8,
    # Global reward shaping
    "trade_completion_bonus": 0.01,    # +1% du compte pour trade fermé en profit
    "trade_completion_penalty": 0.01,  # -1% pour trade fermé en perte
    "time_decay_max": 0.02,            # -2%/step max d'inactivité
    "holding_bonus": 0.005,            # +0.5%/step en position
}

# Cross-symbol correlation window (bars)
CORRELATION_WINDOW = 96  # ~1 day of M15 data
CORRELATION_SYMBOLS = ["XAUUSD", "EURUSD", "US30.cash", "GER40.cash", "US500.cash", "US100.cash", "BTCUSD"]