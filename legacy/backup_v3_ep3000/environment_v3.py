"""
ftmo_agent/environment_v3.py — Environment V3 with FIXED reward shaping:
- Stronger HOLD penalty (cumulative with time)
- Bigger trade rewards (encourage activity)
- Inactivity timeout penalty
- Winning/losing amplification
- Anti-HOLD bias
"""
import numpy as np
import pandas as pd
from config import (SYMBOLS, SymbolSpec, ACTIVE_SYMBOLS, FTMO_CONFIG,
                     RISK_CONFIG, ANTI_BIAS_CONFIG,
                     HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL,
                     PYRAMID, PARTIAL_CLOSE, N_ACTIONS, ACTION_NAMES)
from features_v2 import compute_multi_tf_features, get_symbol_embedding

class Position:
    def __init__(self, symbol, direction, entry_price, lots, sl, tp,
                 spec, position_type='full'):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.lots = lots
        self.initial_lots = lots
        self.sl = sl
        self.tp = tp
        self.spec = spec
        self.position_type = position_type
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

class MultiSymbolEnvV3:
    """
    Environment V3: FIXED reward shaping for active trading.
    
    Key changes:
    - HOLD penalty: -0.02 per bar (was -0.005) + cumulative inactivity
    - Trade open reward: 0.05 (was 0.01)
    - Winning: 15x PnL + 1.0 bonus (was 10x + 0.5)
    - Losing: 8x PnL (was 5x)
    - Inactivity timeout: -0.5 if no trades in 50 bars
    - Bars since last trade tracked
    """
    
    def __init__(self, data_dict, lookback=48):
        self.data_dict = data_dict
        self.symbols = list(data_dict.keys())
        self.lookback = lookback
        self.n_features = data_dict[self.symbols[0]][0].shape[1] + 8 + 5
        self.reset()
    
    def reset(self):
        self.current_symbol = np.random.choice(self.symbols)
        self.features, self.feature_names, self.df = self.data_dict[self.current_symbol]
        self.spec = SYMBOLS[self.current_symbol]
        self.current_step = self.lookback + np.random.randint(0, max(1, len(self.df) - self.lookback - 2000))
        
        self.balance = FTMO_CONFIG['account_size']
        self.peak_balance = self.balance
        self.daily_start_balance = self.balance
        self.positions = []
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.last_trade_day = -1
        self.total_trades = 0
        self.winning_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
        self.episode_pnl = 0
        self.realized_pnl = 0
        self.bars_since_last_trade = 0  # FIX: track inactivity
        
        return self._get_obs()
    
    def _get_obs(self):
        start = self.current_step - self.lookback
        end = self.current_step
        feat = self.features[start:end]
        
        sym_emb = get_symbol_embedding(self.current_symbol)
        sym_emb_tiled = np.tile(sym_emb, (self.lookback, 1))
        
        n_positions = len(self.positions)
        total_unrealized = sum(p.unrealized_pnl(self.df.iloc[self.current_step]['close']) 
                               for p in self.positions)
        net_direction = sum(p.direction * p.lots for p in self.positions)
        total_lots = sum(p.lots for p in self.positions)
        avg_bars = np.mean([p.bars_held for p in self.positions]) if self.positions else 0
        
        pos_info = np.array([[
            n_positions / 3.0,
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
        risk = risk_pct or RISK_CONFIG['risk_per_trade']
        risk_amount = self.balance * risk
        sl_distance = abs(entry - sl_price)
        if sl_distance < 1e-6:
            return self.spec.min_volume
        lots = risk_amount / (sl_distance * self.spec.contract_size * self.spec.pip_value_per_lot / self.spec.pip_size)
        return max(self.spec.min_volume, min(lots, self.spec.max_volume))
    
    def _total_risk(self):
        total = 0
        for p in self.positions:
            sl_dist = abs(p.entry_price - p.sl)
            total += sl_dist * p.lots * p.spec.contract_size
        return total / self.balance
    
    def get_action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        has_position = len(self.positions) > 0
        can_open = (self.trades_today < FTMO_CONFIG['max_trades_per_day'] and
                    self.current_step >= self.cooldown_until and
                    len(self.positions) < FTMO_CONFIG['max_concurrent_positions'])
        
        mask[HOLD] = True
        mask[BUY] = not has_position and can_open
        mask[SELL] = not has_position and can_open
        mask[CLOSE] = has_position
        mask[SPLIT_BUY] = not has_position and can_open
        mask[SPLIT_SELL] = not has_position and can_open
        mask[PYRAMID] = has_position and can_open and self.positions[0].unrealized_pnl(self._get_price()) > RISK_CONFIG['pyramid_min_profit_usd']
        mask[PARTIAL_CLOSE] = has_position and not self.positions[0].partial_closed and self.positions[0].unrealized_pnl(self._get_price()) > 0
        return mask
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, {}
        
        self._check_daily_reset()
        current_price = self._get_price()
        reward = 0
        info = {}
        
        action_mask = self.get_action_mask()
        
        if not action_mask[action]:
            action = HOLD
        
        # Manage existing positions
        positions_to_close = []
        for i, pos in enumerate(self.positions):
            pos.bars_held += 1
            pos.update_slbe(current_price)
            exit_reason = pos.check_sl_tp(current_price)
            if exit_reason:
                positions_to_close.append((i, exit_reason, current_price))
            elif pos.bars_held >= FTMO_CONFIG['max_hold_bars']:
                positions_to_close.append((i, 'TIMEOUT', current_price))
        
        for i, reason, price in reversed(positions_to_close):
            reward += self._close_position(i, price, reason)
        
        spread = self._get_spread()
        can_trade = (self.trades_today < FTMO_CONFIG['max_trades_per_day'] and
                     self.current_step >= self.cooldown_until and
                     len(self.positions) < FTMO_CONFIG['max_concurrent_positions'])
        
        traded_this_step = False
        if action == BUY and not self.positions and can_trade:
            reward += self._open_position(1, current_price + spread/2, 'full')
            traded_this_step = True
        elif action == SELL and not self.positions and can_trade:
            reward += self._open_position(-1, current_price - spread/2, 'full')
            traded_this_step = True
        elif action == SPLIT_BUY and not self.positions and can_trade:
            reward += self._open_position(1, current_price + spread/2, 'split_1', split=True)
            traded_this_step = True
        elif action == SPLIT_SELL and not self.positions and can_trade:
            reward += self._open_position(-1, current_price - spread/2, 'split_1', split=True)
            traded_this_step = True
        elif action == CLOSE and self.positions:
            for i in range(len(self.positions) - 1, -1, -1):
                reward += self._close_position(i, current_price, 'MODEL')
            traded_this_step = True
        elif action == PYRAMID and self.positions and can_trade:
            pos = self.positions[0]
            if pos.unrealized_pnl(current_price) >= RISK_CONFIG['pyramid_min_profit_usd']:
                risk = RISK_CONFIG['risk_per_trade'] * (RISK_CONFIG['pyramid_risk_reduction'] ** (pos.pyramid_level + 1))
                reward += self._open_position(pos.direction, 
                    current_price + (spread/2 if pos.direction == 1 else -spread/2),
                    f'pyramid_{pos.pyramid_level + 1}', risk_pct=risk, pyramid=True)
                traded_this_step = True
        elif action == PARTIAL_CLOSE and self.positions:
            pos = self.positions[0]
            if not pos.partial_closed and pos.unrealized_pnl(current_price) > 0:
                close_lots = pos.lots * RISK_CONFIG['partial_close_pct']
                pnl = self._partial_close(pos, close_lots, current_price)
                reward += pnl / self.balance * 5
                traded_this_step = True
        
        # FIX: Stronger HOLD penalty + cumulative inactivity
        if action == HOLD and not self.positions:
            self.bars_since_last_trade += 1
            # Base penalty
            reward -= 0.05
            # Cumulative penalty: grows with inactivity
            if self.bars_since_last_trade > 20:
                reward -= 0.01 * (self.bars_since_last_trade - 20) / 30.0
            # Hard timeout: big penalty for extended inactivity
            if self.bars_since_last_trade > 50:
                reward -= 1.0
                self.bars_since_last_trade = 0  # reset to avoid infinite penalty
        elif traded_this_step:
            self.bars_since_last_trade = 0
        
        # FTMO checks
        equity = self.balance + sum(p.unrealized_pnl(current_price) for p in self.positions)
        daily_dd = (self.daily_start_balance - equity) / FTMO_CONFIG['account_size']
        total_dd = (self.peak_balance - equity) / FTMO_CONFIG['account_size']
        
        if daily_dd > FTMO_CONFIG['daily_dd_limit'] * 0.5:
            reward -= (daily_dd - FTMO_CONFIG['daily_dd_limit'] * 0.5) * 30
        if total_dd > FTMO_CONFIG['total_dd_limit'] * 0.5:
            reward -= (total_dd - FTMO_CONFIG['total_dd_limit'] * 0.5) * 50
        
        done = False
        if daily_dd >= FTMO_CONFIG['daily_dd_limit']:
            reward -= 10
            done = True
            info['violation'] = 'DAILY_DRAWDOWN'
        if total_dd >= FTMO_CONFIG['total_dd_limit']:
            reward -= 20
            done = True
            info['violation'] = 'TOTAL_DRAWDOWN'
        
        profit_pct = (self.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size']
        if profit_pct >= FTMO_CONFIG['profit_target']:
            reward += 20
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
        
        reward = np.clip(reward, -15, 15)
        
        info.update({
            'balance': self.balance, 'equity': equity,
            'profit_pct': profit_pct, 'daily_dd': daily_dd, 'total_dd': total_dd,
            'positions': len(self.positions), 'symbol': self.current_symbol,
            'buy_trades': self.buy_trades, 'sell_trades': self.sell_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'action_mask': action_mask,
        })
        
        return self._get_obs(), reward, done, info
    
    def _open_position(self, direction, entry_price, pos_type, 
                       split=False, risk_pct=None, pyramid=False):
        atr_norm = self.features[self.current_step, self.feature_names.index('atr_norm')] if 'atr_norm' in self.feature_names else 0.01
        atr = max(abs(atr_norm) * entry_price, self.spec.pip_size * 10)
        
        if direction == 1:
            sl = entry_price - atr * 1.5
            tp = entry_price + atr * 3.0
        else:
            sl = entry_price + atr * 1.5
            tp = entry_price - atr * 3.0
        
        risk = (risk_pct or RISK_CONFIG['risk_per_trade']) * (0.5 if split else 1.0)
        lots = self._calc_position_size(entry_price, sl, risk)
        
        if self._total_risk() + risk > RISK_CONFIG['max_risk_total']:
            return -0.05
        
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
        
        # FIX: Bigger reward for taking a valid trade
        return 0.15  # was 0.01, boosted V3
    
    def _close_position(self, idx, exit_price, reason):
        pos = self.positions[idx]
        pnl = pos.unrealized_pnl(exit_price)
        pnl -= self._get_spread() * pos.lots * pos.spec.contract_size
        
        self.balance += pnl
        self.episode_pnl += pnl
        self.realized_pnl += pnl
        
        # FIX: AMPLIFIED reward signal
        pnl_norm = pnl / FTMO_CONFIG['account_size']
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            # FIX: Winning trade: 15x PnL + 1.0 bonus (was 10x + 0.5)
            reward = pnl_norm * 15 + 1.0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= FTMO_CONFIG['cooldown_after_losses']:
                self.cooldown_until = self.current_step + FTMO_CONFIG['cooldown_bars']
            # FIX: Losing trade: 8x PnL loss (was 5x), no extra penalty
            reward = pnl_norm * 8
        
        self.positions.pop(idx)
        return reward
    
    def _partial_close(self, pos, close_lots, exit_price):
        pnl_per_unit = pos.unrealized_pnl(exit_price) / pos.lots
        pnl = pnl_per_unit * close_lots
        pnl -= self._get_spread() * close_lots * pos.spec.contract_size
        self.balance += pnl
        pos.lots -= close_lots
        pos.partial_closed = True
        return pnl
