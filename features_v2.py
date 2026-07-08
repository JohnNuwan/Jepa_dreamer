"""
ftmo_agent/features_v2.py — Multi-symbol multi-timeframe features V4.
Now includes: HV, MACD/RSI slopes, lag features, spread estimation,
inter-symbol correlations, session breakdown, price action features.
132 base features + ~30 new = 162 per TF → after 4 TFs = ~648 total.
"""
import numpy as np
import pandas as pd
from config import SYMBOLS, SymbolSpec, CORRELATION_WINDOW, CORRELATION_SYMBOLS


def compute_single_tf_features(df, lookback=48, symbol=None):
    """Compute features for a single timeframe. V4: added slopes, HV, spread, lag."""
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']
    df = df.sort_index()

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    opn = df['open'].astype(float)
    vol = df.get('tick_volume', df.get('volume', pd.Series(0, index=df.index))).astype(float)
    spread_col = df.get('spread', None)  # real spread if available in data

    # ── 1. Returns & Log Returns ──
    df['ret'] = close.pct_change()
    df['log_ret'] = np.log(close / close.shift(1))
    for p in [4, 8, 16, 32, 48]:
        df[f'ret_{p}'] = close.pct_change(p)

    # ── 2. RSI + Slope ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    df['rsi'] = (100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))) / 100.0
    df['rsi_slope'] = df['rsi'].diff(3) / 3.0  # NEW: RSI momentum

    # ── 3. EMA + MACD + Slopes ──
    df['ema_fast'] = close.ewm(span=12).mean()
    df['ema_slow'] = close.ewm(span=26).mean()
    df['ema_cross'] = (df['ema_fast'] - df['ema_slow']) / close
    df['ema_cross_slope'] = df['ema_cross'].diff(3) / 3.0  # NEW

    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_slope'] = df['macd'].diff(3) / 3.0  # NEW: MACD momentum
    df['macd_hist_slope'] = df['macd_hist'].diff(3) / 3.0  # NEW

    # ── 4. Bollinger + Position ──
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pos'] = (close - sma20) / (2 * std20 + 1e-10)
    df['bb_width'] = (2 * std20) / close  # NEW: Bollinger width (volatility proxy)

    # ── 5. ATR + HV ──
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    df['atr_norm'] = df['atr'] / close
    df['atr_slope'] = df['atr_norm'].diff(3) / 3.0  # NEW

    # NEW: Historical Volatility (10, 20, 30 bars)
    for p in [10, 20, 30]:
        df[f'hv_{p}'] = df['log_ret'].rolling(p).std() * np.sqrt(p)
    # HV ratio (short-term vs long-term volatility)
    df['hv_ratio'] = df['hv_10'] / (df['hv_30'] + 1e-10)

    # ── 6. ADX ──
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    both_dm = plus_dm.copy()
    both_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    atr_sm = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di = 100 * (both_dm.ewm(alpha=1/14).mean() / (atr_sm + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / (atr_sm + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    df['adx'] = dx.ewm(alpha=1/14).mean() / 100.0
    df['di_diff'] = (plus_di - minus_di) / 100.0
    df['di_diff_slope'] = df['di_diff'].diff(3) / 3.0  # NEW

    # ── 7. Volume ──
    vol_sma = vol.rolling(20).mean()
    df['vol_ratio'] = vol / (vol_sma + 1e-10)
    df['vol_z'] = (vol - vol_sma) / (vol.rolling(20).std() + 1e-10)

    # ── 8. VWAP ──
    typical = (high + low + close) / 3
    cum_vp = (typical * vol).rolling(48).sum()
    cum_v = vol.rolling(48).sum()
    df['vwap_dist'] = (close - cum_vp / (cum_v + 1e-10)) / close

    # ── 9. Stochastic ──
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['stoch_k'] = (close - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_slope'] = df['stoch_k'].diff(2) / 2.0  # NEW

    # ── 10. Price Action ── NEW: more robust
    body = (close - opn)
    wick_high = high - df[['open', 'close']].max(axis=1)
    wick_low = df[['open', 'close']].min(axis=1) - low
    total_range = (high - low)
    df['body_ratio'] = body.abs() / (total_range + 1e-10)
    df['upper_wick_ratio'] = wick_high / (total_range + 1e-10)
    df['lower_wick_ratio'] = wick_low / (total_range + 1e-10)
    df['range_norm'] = total_range / close
    # Candlestick patterns
    df['doji'] = (body.abs() / close < 0.001).astype(float)
    df['hammer'] = ((wick_low > body.abs() * 2) & (wick_high < body.abs() * 0.3)).astype(float)
    df['shooting_star'] = ((wick_high > body.abs() * 2) & (wick_low < body.abs() * 0.3)).astype(float)
    df['engulfing'] = ((body.shift(1) * body < 0) & (body.abs() > body.shift(1).abs() * 1.5)).astype(float)

    # ── 11. Range Position with multiple windows ──
    for w in [8, 24, 48]:
        rmax = high.rolling(w).max()
        rmin = low.rolling(w).min()
        df[f'range_pos_{w}'] = (close - rmin) / (rmax - rmin + 1e-10)

    # ── 12. Spread estimation ── NEW
    if spread_col is not None:
        df['spread_est'] = spread_col.astype(float) / close
    else:
        # Estimate spread from OHLC (Roll's spread estimator approximation)
        # Covariance of price changes gives an estimate
        dp = close.diff()
        df['spread_est'] = (-dp.shift(1) * dp).rolling(20).mean().clip(lower=0).fillna(0)
        df['spread_est'] = df['spread_est'] / (close + 1e-10)
    df['spread_z'] = (df['spread_est'] - df['spread_est'].rolling(48).mean()) / (df['spread_est'].rolling(48).std() + 1e-10)

    # ── 13. Momentum composite ── NEW
    df['mom_composite'] = (df['rsi'] - 0.5) * 0.3 + df['di_diff'] * 0.3 + df['stoch_k'] * 0.2 + df['range_pos_8'] * 0.2

    # ── 14. Session Features (improved) ──
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
        minute = df.index.minute if hasattr(df.index, 'minute') else 0
        day = df.index.dayofweek

        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * day / 5)
        df['day_cos'] = np.cos(2 * np.pi * day / 5)

        # Session breakdown (M15 bars)
        time_num = hour + minute / 60.0
        df['session_asia'] = ((time_num >= 0) & (time_num < 9)).astype(float)
        df['session_london_open'] = ((time_num >= 8) & (time_num < 10)).astype(float)
        df['session_london'] = ((time_num >= 9) & (time_num < 17)).astype(float)
        df['session_ny_open'] = ((time_num >= 13) & (time_num < 15)).astype(float)
        df['session_ny'] = ((time_num >= 15) & (time_num < 22)).astype(float)
        df['session_overlap'] = ((time_num >= 13) & (time_num < 17)).astype(float)
        df['session_close'] = ((time_num >= 21) | (time_num < 1)).astype(float)

        # Session volatility profile (NEW: which session is most volatile now)
        df['session_volatile'] = 0.0
        df.loc[df['session_london_open'] == 1, 'session_volatile'] = 1.0
        df.loc[df['session_ny_open'] == 1, 'session_volatile'] = 1.0
        df.loc[df['session_overlap'] == 1, 'session_volatile'] = 0.5
    else:
        for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                   'session_asia', 'session_london_open', 'session_london',
                   'session_ny_open', 'session_ny', 'session_overlap',
                   'session_close', 'session_volatile']:
            df[c] = 0.0

    # ── 15. Lag features ── NEW (shift selected features by 1, 2 bars)
    for feature in ['ret', 'rsi', 'macd_hist', 'atr_norm', 'stoch_k']:
        for lag in [1, 2]:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

    return df


def get_feature_columns_v4():
    """Return list of feature column names (V4 expanded)."""
    base = [
        # Returns (7)
        'ret', 'log_ret', 'ret_4', 'ret_8', 'ret_16', 'ret_32', 'ret_48',
        # RSI (2)
        'rsi', 'rsi_slope',
        # EMA/MACD (6)
        'ema_cross', 'ema_cross_slope', 'macd', 'macd_signal', 'macd_hist',
        'macd_slope', 'macd_hist_slope',
        # Bollinger (2)
        'bb_pos', 'bb_width',
        # ATR (3)
        'atr_norm', 'atr_slope',
        # HV (4)
        'hv_10', 'hv_20', 'hv_30', 'hv_ratio',
        # ADX (3)
        'adx', 'di_diff', 'di_diff_slope',
        # Volume (2)
        'vol_ratio', 'vol_z',
        # VWAP (1)
        'vwap_dist',
        # Stochastic (3)
        'stoch_k', 'stoch_d', 'stoch_slope',
        # Price Action (9)
        'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'range_norm',
        'doji', 'hammer', 'shooting_star', 'engulfing',
        # Range Position (3)
        'range_pos_8', 'range_pos_24', 'range_pos_48',
        # Spread (2)
        'spread_est', 'spread_z',
        # Momentum (1)
        'mom_composite',
        # Session (12)
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'session_asia', 'session_london_open', 'session_london',
        'session_ny_open', 'session_ny', 'session_overlap',
        'session_close', 'session_volatile',
    ]  # 60 features

    # Lag features: 5 features × 2 lags = 10 more
    lagged = [f'{f}_lag{l}' for f in ['ret', 'rsi', 'macd_hist', 'atr_norm', 'stoch_k']
              for l in [1, 2]]

    return base + lagged  # 70 features per TF


def compute_multi_tf_features(df_m15, lookback=48, symbol=None):
    """
    Compute multi-timeframe features from M15 data. V4 expanded.
    Returns: (features_array, feature_names, df_processed)
    """
    # Primary TF (M15) — V4 features
    df_m15 = compute_single_tf_features(df_m15, lookback, symbol)
    m15_cols = get_feature_columns_v4()

    # Resample for higher timeframes
    df_m15_idx = df_m15.set_index('time') if 'time' in df_m15.columns else df_m15

    # Helper: resample and compute features for higher TF
    def resample_and_features(df_idx, rule, lookback_tf):
        df_resampled = df_idx.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'tick_volume': 'sum'
        }).dropna().reset_index()
        # Use basic features for higher TFs (lighter)
        df_feat = compute_single_tf_features(df_resampled, lookback_tf, symbol)
        return df_feat.set_index('time') if 'time' in df_feat.columns else df_feat

    # H1
    df_h1 = resample_and_features(df_m15_idx, '1h', 24)
    h1_features = df_h1[m15_cols] if all(c in df_h1.columns for c in m15_cols) else df_h1[[c for c in m15_cols if c in df_h1.columns]]

    # H4
    df_h4 = resample_and_features(df_m15_idx, '4h', 12)
    h4_features = df_h4[m15_cols] if all(c in df_h4.columns for c in m15_cols) else df_h4[[c for c in m15_cols if c in df_h4.columns]]

    # D1
    df_d1 = resample_and_features(df_m15_idx, '1D', 5)
    d1_features = df_d1[m15_cols] if all(c in df_d1.columns for c in m15_cols) else df_d1[[c for c in m15_cols if c in df_d1.columns]]

    # Align to M15 timeline
    m15_time_idx = df_m15['time'] if 'time' in df_m15.columns else df_m15.index

    h1_aligned = h1_features.reindex(m15_time_idx, method='ffill').fillna(0)
    h4_aligned = h4_features.reindex(m15_time_idx, method='ffill').fillna(0)
    d1_aligned = d1_features.reindex(m15_time_idx, method='ffill').fillna(0)

    # Prefix columns
    h1_aligned.columns = [f'h1_{c}' for c in h1_aligned.columns]
    h4_aligned.columns = [f'h4_{c}' for c in h4_aligned.columns]
    d1_aligned.columns = [f'd1_{c}' for c in d1_aligned.columns]

    # Get actual columns present (handle missing ones gracefully)
    m15_feat = df_m15[m15_cols].copy()
    all_features = pd.concat([
        m15_feat.reset_index(drop=True),
        h1_aligned.reset_index(drop=True),
        h4_aligned.reset_index(drop=True),
        d1_aligned.reset_index(drop=True)
    ], axis=1)

    # Normalize
    for col in all_features.columns:
        roll_mean = all_features[col].rolling(lookback * 4, min_periods=10).mean()
        roll_std = all_features[col].rolling(lookback * 4, min_periods=10).std()
        all_features[col] = ((all_features[col] - roll_mean) / (roll_std + 1e-8)).clip(-5, 5)

    all_features = all_features.fillna(0)

    feature_names = list(all_features.columns)
    return all_features.values, feature_names, df_m15


def compute_correlations(data_dict, lookback=96):
    """
    NEW: Pre-compute inter-symbol correlations.
    Returns dict: {symbol: {other_symbol: [correlation_values]}}
    """
    # Get aligned close prices for all symbols
    closes = {}
    for symbol in CORRELATION_SYMBOLS:
        if symbol in data_dict:
            _, _, df = data_dict[symbol]
            closes[symbol] = df['close'].values

    if len(closes) < 2:
        return {}

    min_len = min(len(v) for v in closes.values())
    correlations = {}
    for sym1 in closes:
        correlations[sym1] = {}
        for sym2 in closes:
            if sym1 == sym2:
                correlations[sym1][sym2] = np.zeros(min_len)
                continue
            # Rolling correlation
            s1 = closes[sym1][:min_len]
            s2 = closes[sym2][:min_len]
            corr = pd.Series(s1).rolling(lookback).corr(pd.Series(s2)).fillna(0).values
            correlations[sym1][sym2] = corr
    return correlations


def get_correlation_features(correlations, current_symbol, step):
    """
    NEW: Get correlation features for current symbol at given step.
    Returns array of shape (n_other_symbols+1,) — correlation with each other symbol + market avg.
    """
    if not correlations or current_symbol not in correlations:
        return np.zeros(len(CORRELATION_SYMBOLS))

    step = min(step, len(correlations[current_symbol][list(correlations[current_symbol].keys())[0]]) - 1)
    step = max(0, step)

    feats = []
    for sym in CORRELATION_SYMBOLS:
        if sym == current_symbol:
            continue
        corr_val = correlations[current_symbol].get(sym, [0])[min(step, len(correlations[current_symbol].get(sym, [0])) - 1)]
        feats.append(corr_val)

    # Add average correlation to market
    avg_corr = np.mean(feats) if feats else 0
    feats.append(avg_corr)

    return np.array(feats, dtype=np.float32)


def get_symbol_embedding_v4(symbol, n_dim=8):
    """V4: Enhanced symbol embeddings with volatility and correlation profile."""
    embeddings = {
        'XAUUSD':     [1.0, 0.0, 0.0, 0.0, 0.5, 0.8, 0.3, 0.1],
        'EURUSD':     [0.0, 1.0, 0.0, 0.0, 0.3, 0.2, 0.5, 0.4],
        'GBPUSD':     [0.0, 0.5, 0.5, 0.0, 0.3, 0.3, 0.5, 0.4],
        'US30.cash':  [0.5, 0.0, 0.0, 0.5, 0.8, 0.5, 0.2, 0.8],
        'GER40.cash': [0.5, 0.0, 0.5, 0.0, 0.8, 0.4, 0.2, 0.7],
        'US500.cash': [0.5, 0.0, 0.3, 0.3, 0.7, 0.6, 0.3, 0.7],
        'US100.cash': [0.5, 0.0, 0.3, 0.3, 0.9, 0.5, 0.3, 0.9],
        'BTCUSD':     [0.0, 0.0, 0.5, 0.5, 1.0, 0.9, 0.1, 0.5],
    }
    return embeddings.get(symbol, [0.0] * n_dim)