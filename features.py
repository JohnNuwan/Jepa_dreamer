"""
ftmo_agent/features.py — Feature engineering for XAUUSD trading.
Computes 35+ technical features from OHLCV data.
"""
import numpy as np
import pandas as pd

def compute_features(df, lookback=48):
    """Compute all features. Returns DataFrame with features column."""
    df = df.copy()
    df.index = pd.to_datetime(df['time'])
    df = df.sort_index()
    
    # Returns
    df['ret'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Multi-timeframe returns
    for p in [4, 8, 16, 32, 48]:
        df[f'ret_{p}'] = df['close'].pct_change(p)
    
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs))) / 100.0
    
    # EMA crossover
    df['ema_fast'] = df['close'].ewm(span=12).mean()
    df['ema_slow'] = df['close'].ewm(span=26).mean()
    df['ema_cross'] = (df['ema_fast'] - df['ema_slow']) / df['close']
    
    # MACD
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_pos'] = (df['close'] - sma20) / (2 * std20 + 1e-10)
    
    # ATR (volatility)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    df['atr_norm'] = df['atr'] / df['close']
    
    # ADX (trend strength)
    plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
    minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr_sm = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / (atr_sm + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / (atr_sm + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    df['adx'] = dx.ewm(alpha=1/14).mean() / 100.0
    
    # Volume features
    vol_sma = df['tick_volume'].rolling(20).mean()
    df['vol_ratio'] = df['tick_volume'] / (vol_sma + 1e-10)
    df['vol_z'] = (df['tick_volume'] - vol_sma) / (df['tick_volume'].rolling(20).std() + 1e-10)
    
    # VWAP
    typical = (df['high'] + df['low'] + df['close']) / 3
    cum_vp = (typical * df['tick_volume']).rolling(48).sum()
    cum_v = df['tick_volume'].rolling(48).sum()
    df['vwap'] = cum_vp / (cum_v + 1e-10)
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['close']
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['momentum_norm'] = df['momentum'] / (df['atr'] + 1e-10)
    
    # Stochastic
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = (df['close'] - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Higher timeframe context (using rolling on M15)
    df['h1_trend'] = df['close'].pct_change(4)  # ~1h on M15
    df['h4_trend'] = df['close'].pct_change(16)  # ~4h on M15
    df['d1_trend'] = df['close'].pct_change(96)  # ~1d on M15
    
    # Price levels
    rolling_max = df['high'].rolling(48).max()
    rolling_min = df['low'].rolling(48).min()
    df['range_pos'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    # Candles
    df['body'] = (df['close'] - df['open']) / df['close']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Time features (cyclical encoding)
    hour = df.index.hour
    day = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day / 5)
    df['day_cos'] = np.cos(2 * np.pi * day / 5)
    
    # Session flags
    df['london'] = ((hour >= 8) & (hour < 17)).astype(float)
    df['ny'] = ((hour >= 13) & (hour < 22)).astype(float)
    df['overlap'] = ((hour >= 13) & (hour < 17)).astype(float)
    df['asian'] = ((hour >= 0) & (hour < 9)).astype(float)
    
    # Select feature columns
    feature_cols = [
        'ret', 'log_ret', 'ret_4', 'ret_8', 'ret_16', 'ret_32', 'ret_48',
        'rsi', 'ema_cross', 'macd', 'macd_hist', 'bb_pos',
        'atr_norm', 'adx', 'vol_ratio', 'vol_z', 'vwap_dist',
        'momentum_norm', 'stoch_k', 'stoch_d',
        'h1_trend', 'h4_trend', 'd1_trend', 'range_pos',
        'body', 'upper_wick', 'lower_wick',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'london', 'ny', 'overlap', 'asian',
    ]
    
    # Normalize to [-1, 1] using rolling z-score (no look-ahead)
    for col in feature_cols:
        roll_mean = df[col].rolling(lookback * 4, min_periods=10).mean()
        roll_std = df[col].rolling(lookback * 4, min_periods=10).std()
        df[col] = ((df[col] - roll_mean) / (roll_std + 1e-8)).clip(-5, 5)
    
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols
