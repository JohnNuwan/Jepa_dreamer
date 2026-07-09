"""
ftmo_agent/environment_v2.py — Multi-symbol trading env with:
- 8 actions (HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE)
- Anti-directional bias penalty
- Pyramiding, Split positions, SLBE, Partial close
- Multi-symbol support (rotates through symbols during training)
- FTMO rules with multi-position support
"""
import numpy as np
import pandas as pd
from config import (SYMBOLS, SymbolSpec, ACTIVE_SYMBOLS, FTMO_CONFIG,
                     RISK_CONFIG, ANTI_BIAS_CONFIG,
                     HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL,
                     PYRAMID, PARTIAL_CLOSE, N_ACTIONS, ACTION_NAMES)
from features_v2 import compute_multi_tf_features, get_symbol_embedding

class Position:
    """Represents an open position with advanced management."""
    def __init__(self, symbol, direction, entry_price, lots, sl, tp,
                 spec, position_type='full'):
        self.symbol = symbol
        self.direction = direction  # 1=long, -1=short
        self.entry_price = entry_price
        self.lots = lots
        self.initial_lots = lots
        self.sl = sl
        self.tp = tp
        self.spec = spec
        self.position_type = position_type  # 'full', 'split_1', 'split_2'
        self.bars_held = 0
        self.slbe_applied = False
        self.partial_closed = False
        self.pyramid_level = 0
    
    def unrealized_pnl(self, current_price):
        if self.direction == 1:
            return (current_price - self.entry_price) * self.lots * self.spec.contract_size
        else:
            return (self.entry_price - current_price) * self.lots * self.spec.contract_size
    
    def update_slbe(self, current_price):
        """Stop Loss Break Even."""
        if self.slbe_applied:
            return False
        
        profit = self.unrealized_pnl(current_price)
        if profit >= RISK_CONFIG['slbe_trigger_usd']:
            offset = RISK_CONFIG['slbe_offset_usd']
            if self.direction == 1:
                self.sl = self.entry_price + offset / (self.lots * self.spec.contract_size)
            else:
                self.sl = self.entry_price - offset / (self.lots * self.spec.contract_size)
            self.slbe_applied = True
            return True
        return False
    
    def check_sl_tp(self, current_price):
        """Check if SL or TP hit. Returns 'SL', 'TP', or None."""
        if self.direction == 1:
            if current_price <= self.sl:
                return 'SL'
            if current_price >= self.tp:
                return 'TP'
        else:
            if current_price >= self.sl:
                return 'SL'
            if current_price <= self.tp:
                return 'TP'
        return None

class MultiSymbolEnv:
    """
    Multi-symbol trading environment for FTMO.
    
    Features:
    - Rotates through symbols during training (diversification)
    - 8 actions including split, pyramid, partial close
    - Anti-bias penalty (prevents buy-only or sell-only agents)
    - FTMO-compliant risk management
    """
    
    def __init__(self, data_dict, feature_cache=None, lookback=48,
                 max_episodes_per_symbol=500):
        """
        data_dict: {symbol: (features_array, feature_names, df_m15)}
        """
        self.data_dict = data_dict
        self.symbols = list(data_dict.keys())
        self.lookback = lookback
        self.n_features = data_dict[self.symbols[0]][0].shape[1] + 8 + 5  # +symbol_emb +position
        self.max_episodes = max_episodes_per_symbol
        
        self.reset()
    
    def reset(self):
        self.current_symbol = np.random.choice(self.symbols)
        self.features, self.feature_names, self.df = self.data_dict[self.current_symbol]
        self.spec = SYMBOLS[self.current_symbol]
        
        self.current_step = self.lookback + np.random.randint(0, max(1, len(self.df) - self.lookback - 2000))
        
        # FTMO state
        self.balance = FTMO_CONFIG['account_size']
        self.peak_balance = self.balance
        self.daily_start_balance = self.balance
        self.positions = []  # list of Position objects
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.last_trade_day = -1
        self.total_trades = 0
        self.winning_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
        self.episode_pnl = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get observation: (lookback, n_features)"""
        start = self.current_step - self.lookback
        end = self.current_step
        
        feat = self.features[start:end]
        
        # Symbol embedding (8 dims, constant across time)
        sym_emb = get_symbol_embedding(self.current_symbol)
        sym_emb_tiled = np.tile(sym_emb, (self.lookback, 1))
        
        # Position info (5 dims)
        n_positions = len(self.positions)
        total_unrealized = sum(p.unrealized_pnl(self.df.iloc[self.current_step]['close']) 
                               for p in self.positions)
        net_direction = sum(p.direction * p.lots for p in self.positions)
        total_lots = sum(p.lots for p in self.positions)
        avg_bars = np.mean([p.bars_held for p in self.positions]) if self.positions else 0
        
        pos_info = np.array([[
            n_positions / 3.0,  # normalize
            total_unrealized / self.balance,
            net_direction / (total_lots + 1e-8),
            total_lots / self.spec.max_volume,
            avg_bars / FTMO_CONFIG['max_hold_bars'],
        ]])
        pos_tiled = np.tile(pos_info, (self.lookback, 1))
        
        obs = np.hstack([feat, sym_emb_tiled, pos_tiled])
        return obs.astype(np.float32)
    
    def _get_price(self):
        return self.df.iloc[self.current_step]['close']
    
    def _get_spread(self):
        return self.spec.spread_points * self.spec.pip_size
    
    def _check_daily_reset(self):
        current_day = self.df.iloc[self.current_step]['time'].day
        if current_day != self.last_trade_day and self.last_trade_day != -1:
            self.daily_start_balance = self.balance
            self.trades_today = 0
            self.cooldown_until = 0
        self.last_trade_day = current_day
    
    def _calc_position_size(self, entry, sl_price, risk_pct=None):
        """Risk-based position sizing."""
        risk = risk_pct or RISK_CONFIG['risk_per_trade']
        risk_amount = self.balance * risk
        sl_distance = abs(entry - sl_price)
        if sl_distance < 1e-6:
            return self.spec.min_volume
        lots = risk_amount / (sl_distance * self.spec.contract_size * self.spec.pip_value_per_lot / self.spec.pip_size)
        return max(self.spec.min_volume, min(lots, self.spec.max_volume))
    
    def _total_risk(self):
        """Current total risk across all positions."""
        total = 0
        for p in self.positions:
            sl_dist = abs(p.entry_price - p.sl)
            total += sl_dist * p.lots * p.spec.contract_size
        return total / self.balance
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, {}
        
        self._check_daily_reset()
        current_price = self._get_price()
        reward = 0
        info = {}
        
        # Manage existing positions
        positions_to_close = []
        for i, pos in enumerate(self.positions):
            pos.bars_held += 1
            
            # SLBE check
            pos.update_slbe(current_price)
            
            # SL/TP check
            exit_reason = pos.check_sl_tp(current_price)
            if exit_reason:
                positions_to_close.append((i, exit_reason, current_price))
            elif pos.bars_held >= FTMO_CONFIG['max_hold_bars']:
                positions_to_close.append((i, 'TIMEOUT', current_price))
        
        # Close positions (reverse order to maintain indices)
        for i, reason, price in reversed(positions_to_close):
            reward += self._close_position(i, price, reason)
        
        # Execute action
        spread = self._get_spread()
        can_trade = (self.trades_today < FTMO_CONFIG['max_trades_per_day'] and
                     self.current_step >= self.cooldown_until and
                     len(self.positions) < FTMO_CONFIG['max_concurrent_positions'])
        
        if action == BUY and not self.positions and can_trade:
            reward += self._open_position(1, current_price + spread/2, 'full')
        elif action == SELL and not self.positions and can_trade:
            reward += self._open_position(-1, current_price - spread/2, 'full')
        elif action == SPLIT_BUY and not self.positions and can_trade:
            reward += self._open_position(1, current_price + spread/2, 'split_1', split=True)
        elif action == SPLIT_SELL and not self.positions and can_trade:
            reward += self._open_position(-1, current_price - spread/2, 'split_1', split=True)
        elif action == CLOSE and self.positions:
            for i in range(len(self.positions) - 1, -1, -1):
                reward += self._close_position(i, current_price, 'MODEL')
        elif action == PYRAMID and self.positions and can_trade:
            # Add to existing position (same direction)
            pos = self.positions[0]
            if pos.unrealized_pnl(current_price) >= RISK_CONFIG['pyramid_min_profit_usd']:
                risk = RISK_CONFIG['risk_per_trade'] * (RISK_CONFIG['pyramid_risk_reduction'] ** (pos.pyramid_level + 1))
                reward += self._open_position(pos.direction, 
                    current_price + (spread/2 if pos.direction == 1 else -spread/2),
                    f'pyramid_{pos.pyramid_level + 1}', risk_pct=risk, pyramid=True)
        elif action == PARTIAL_CLOSE and self.positions:
            # Close 50% of first position
            pos = self.positions[0]
            if not pos.partial_closed and pos.unrealized_pnl(current_price) > 0:
                close_lots = pos.lots * RISK_CONFIG['partial_close_pct']
                pnl = self._partial_close(pos, close_lots, current_price)
                reward += pnl / self.balance
        
        # Small penalty for inaction
        if action == HOLD and not self.positions:
            reward -= 0.002
        
        # FTMO checks
        equity = self.balance + sum(p.unrealized_pnl(current_price) for p in self.positions)
        daily_dd = (self.daily_start_balance - equity) / FTMO_CONFIG['account_size']
        total_dd = (self.peak_balance - equity) / FTMO_CONFIG['account_size']
        
        # Drawdown penalties
        if daily_dd > FTMO_CONFIG['daily_dd_limit'] * 0.6:
            reward -= (daily_dd - FTMO_CONFIG['daily_dd_limit'] * 0.6) * 15
        if total_dd > FTMO_CONFIG['total_dd_limit'] * 0.6:
            reward -= (total_dd - FTMO_CONFIG['total_dd_limit'] * 0.6) * 25
        
        # Hard violations
        done = False
        if daily_dd >= FTMO_CONFIG['daily_dd_limit']:
            reward -= 5
            done = True
            info['violation'] = 'DAILY_DRAWDOWN'
        if total_dd >= FTMO_CONFIG['total_dd_limit']:
            reward -= 10
            done = True
            info['violation'] = 'TOTAL_DRAWDOWN'
        
        # Profit target
        profit_pct = (self.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size']
        if profit_pct >= FTMO_CONFIG['profit_target']:
            reward += 10
            done = True
            info['success'] = True
        
        # Anti-bias penalty
        if ANTI_BIAS_CONFIG['enabled'] and self.total_trades >= ANTI_BIAS_CONFIG['min_trades_for_bias_check']:
            buy_ratio = self.buy_trades / max(1, self.buy_trades + self.sell_trades)
            sell_ratio = 1 - buy_ratio
            if buy_ratio > ANTI_BIAS_CONFIG['max_buy_ratio']:
                reward -= (buy_ratio - ANTI_BIAS_CONFIG['max_buy_ratio']) * ANTI_BIAS_CONFIG['penalty_weight']
            if sell_ratio > ANTI_BIAS_CONFIG['max_sell_ratio']:
                reward -= (sell_ratio - ANTI_BIAS_CONFIG['max_sell_ratio']) * ANTI_BIAS_CONFIG['penalty_weight']
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        
        reward = np.clip(reward, -5, 5)
        
        info.update({
            'balance': self.balance, 'equity': equity,
            'profit_pct': profit_pct, 'daily_dd': daily_dd, 'total_dd': total_dd,
            'positions': len(self.positions), 'symbol': self.current_symbol,
            'buy_trades': self.buy_trades, 'sell_trades': self.sell_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
        })
        
        return self._get_obs(), reward, done, info
    
    def _open_position(self, direction, entry_price, pos_type, 
                       split=False, risk_pct=None, pyramid=False):
        """Open a new position."""
        atr_norm = self.features[self.current_step, self.feature_names.index('atr_norm')] if 'atr_norm' in self.feature_names else 0.01
        atr = max(abs(atr_norm) * entry_price, self.spec.pip_size * 10)
        
        if direction == 1:
            sl = entry_price - atr * 1.5
            tp = entry_price + atr * 3.0
        else:
            sl = entry_price + atr * 1.5
            tp = entry_price - atr * 3.0
        
        if split:
            risk = (risk_pct or RISK_CONFIG['risk_per_trade']) * 0.5
        else:
            risk = risk_pct or RISK_CONFIG['risk_per_trade']
        
        lots = self._calc_position_size(entry_price, sl, risk)
        
        # Check total risk
        if self._total_risk() + risk > RISK_CONFIG['max_risk_total']:
            return -0.01  # penalty for trying to exceed risk limit
        
        pos = Position(
            symbol=self.current_symbol, direction=direction,
            entry_price=entry_price, lots=lots, sl=sl, tp=tp,
            spec=self.spec, position_type=pos_type,
        )
        if pyramid:
            pos.pyramid_level = len([p for p in self.positions if p.direction == direction])
        
        self.positions.append(pos)
        self.trades_today += 1
        self.total_trades += 1
        if direction == 1:
            self.buy_trades += 1
        else:
            self.sell_trades += 1
        
        return 0.001  # small reward for taking a valid trade
    
    def _close_position(self, idx, exit_price, reason):
        """Close a position and return realized PnL."""
        pos = self.positions[idx]
        pnl = pos.unrealized_pnl(exit_price)
        # Subtract spread cost
        pnl -= self._get_spread() * pos.lots * pos.spec.contract_size
        
        self.balance += pnl
        self.episode_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= FTMO_CONFIG['cooldown_after_losses']:
                self.cooldown_until = self.current_step + FTMO_CONFIG['cooldown_bars']
        
        self.positions.pop(idx)
        return pnl / FTMO_CONFIG['account_size']
    
    def _partial_close(self, pos, close_lots, exit_price):
        """Close part of a position."""
        pnl_per_unit = pos.unrealized_pnl(exit_price) / pos.lots
        pnl = pnl_per_unit * close_lots
        pnl -= self._get_spread() * close_lots * pos.spec.contract_size
        
        self.balance += pnl
        pos.lots -= close_lots
        pos.partial_closed = True
        return pnl
