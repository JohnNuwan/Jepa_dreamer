"""
ftmo_agent/environment.py — Trading environment with FTMO rules.
Simulates FTMO 10K challenge: daily DD 5%, total DD 10%, profit target 10%.
"""
import numpy as np
import pandas as pd

# Actions
HOLD = 0
BUY = 1
SELL = 2
CLOSE = 3
N_ACTIONS = 4

class FTMOTradingEnv:
    """
    Trading environment for FTMO challenge.
    
    State: (lookback, n_features) — last N bars of features + position info
    Action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    Reward: risk-adjusted PnL with FTMO rule penalties
    
    FTMO rules:
    - Account: $10,000
    - Daily drawdown: 5% ($500)
    - Total drawdown: 10% ($1,000)
    - Profit target: 10% ($1,000)
    """
    
    def __init__(self, df, features, lookback=48, 
                 account_size=10000, risk_per_trade=0.01,
                 daily_dd_limit=0.05, total_dd_limit=0.10,
                 profit_target=0.10, max_hold_bars=96,
                 spread_pips=2, pip_size=0.01):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.lookback = lookback
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.daily_dd_limit = daily_dd_limit
        self.total_dd_limit = total_dd_limit
        self.profit_target = profit_target
        self.max_hold = max_hold_bars
        self.spread = spread_pips * pip_size
        self.pip_size = pip_size
        
        self.n_features = len(features) + 3  # + position info
        self.reset()
    
    def reset(self):
        self.current_step = self.lookback
        self.balance = self.account_size
        self.peak_balance = self.account_size
        self.daily_start_balance = self.account_size
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.position_bars = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.last_trade_day = -1
        self.episode_pnl = 0
        self.daily_pnl = 0
        self.done = False
        return self._get_obs()
    
    def _get_obs(self):
        """Get observation: (lookback, n_features)"""
        feat = self.df.loc[self.current_step - self.lookback:self.current_step - 1, 
                           self.features].values
        # Position info
        pos_info = np.array([[
            self.position,
            self.position_bars / self.max_hold,
            (self.entry_price - self.df.loc[self.current_step, 'close']) / 
            (self.df.loc[self.current_step, 'close'] + 1e-8) if self.position != 0 else 0
        ]])
        # Repeat position info for each timestep (or append)
        pos_repeated = np.tile(pos_info, (self.lookback, 1))
        obs = np.hstack([feat, pos_repeated])
        return obs.astype(np.float32)
    
    def _get_price(self):
        return self.df.loc[self.current_step, 'close']
    
    def _get_bid_ask(self):
        close = self.df.loc[self.current_step, 'close']
        return close - self.spread/2, close + self.spread/2
    
    def _check_daily_reset(self):
        """Check if new day for daily drawdown reset."""
        current_day = self.df.loc[self.current_step, 'time'].day
        if current_day != self.last_trade_day and self.last_trade_day != -1:
            self.daily_start_balance = self.balance
            self.trades_today = 0
            self.daily_pnl = 0
        self.last_trade_day = current_day
    
    def _calculate_position_size(self, entry, sl_price):
        """Risk-based position sizing: risk_per_trade % of balance."""
        risk_amount = self.balance * self.risk_per_trade
        sl_distance = abs(entry - sl_price)
        if sl_distance < 1e-6:
            return 0.01  # minimum
        # For XAUUSD: 1 lot = 100 oz, $1 move = $100
        # position_size = risk_amount / (sl_distance_in_pips * pip_value_per_lot)
        pip_value_per_lot = 100  # $100 per $1 move for 1 lot XAUUSD
        lots = risk_amount / (sl_distance * pip_value_per_lot)
        return max(0.01, min(lots, 0.5))  # cap at 0.5 lots
    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}
        
        self._check_daily_reset()
        
        current_price = self._get_price()
        reward = 0
        info = {}
        
        # Manage existing position
        if self.position != 0:
            self.position_bars += 1
            # Check SL/TP
            if self.position == 1:  # long
                unrealized = (current_price - self.entry_price) * self.lot_size * 100
                if current_price <= self.stop_loss:
                    reward += self._close_position(current_price, 'SL')
                    info['exit'] = 'SL'
                elif current_price >= self.take_profit:
                    reward += self._close_position(current_price, 'TP')
                    info['exit'] = 'TP'
                elif self.position_bars >= self.max_hold:
                    reward += self._close_position(current_price, 'TIMEOUT')
                    info['exit'] = 'TIMEOUT'
            elif self.position == -1:  # short
                unrealized = (self.entry_price - current_price) * self.lot_size * 100
                if current_price >= self.stop_loss:
                    reward += self._close_position(current_price, 'SL')
                    info['exit'] = 'SL'
                elif current_price <= self.take_profit:
                    reward += self._close_position(current_price, 'TP')
                    info['exit'] = 'TP'
                elif self.position_bars >= self.max_hold:
                    reward += self._close_position(current_price, 'TIMEOUT')
                    info['exit'] = 'TIMEOUT'
        
        # Execute action (only if flat)
        if self.position == 0 and action in [BUY, SELL]:
            if self.trades_today < 5:  # KICK: max 5 trades/day
                bid, ask = self._get_bid_ask()
                atr = self.df.loc[self.current_step, 'atr_norm'] * current_price
                
                if action == BUY:
                    entry = ask
                    sl = entry - max(atr * 1.5, 2.0)  # 1.5x ATR or min $2
                    tp = entry + max(atr * 3.0, 4.0)  # 1:2 RR minimum
                    self.position = 1
                else:  # SELL
                    entry = bid
                    sl = entry + max(atr * 1.5, 2.0)
                    tp = entry - max(atr * 3.0, 4.0)
                    self.position = -1
                
                self.entry_price = entry
                self.stop_loss = sl
                self.take_profit = tp
                self.position_bars = 0
                self.lot_size = self._calculate_position_size(entry, sl)
                self.trades_today += 1
                self.total_trades += 1
                info['entry'] = 'BUY' if action == BUY else 'SELL'
                info['entry_price'] = entry
                info['sl'] = sl
                info['tp'] = tp
                info['lots'] = self.lot_size
        
        # Small penalty for holding too long without action
        if self.position == 0 and action == HOLD:
            reward -= 0.001  # tiny penalty to encourage trading
        
        # Check FTMO violations
        equity = self.balance
        if self.position != 0:
            if self.position == 1:
                equity += (current_price - self.entry_price) * self.lot_size * 100
            else:
                equity += (self.entry_price - current_price) * self.lot_size * 100
        
        daily_dd = (self.daily_start_balance - equity) / self.account_size
        total_dd = (self.peak_balance - equity) / self.account_size
        
        # Penalty for approaching drawdown limits
        if daily_dd > self.daily_dd_limit * 0.7:
            reward -= (daily_dd - self.daily_dd_limit * 0.7) * 20
        if total_dd > self.total_dd_limit * 0.7:
            reward -= (total_dd - self.total_dd_limit * 0.7) * 30
        
        # Hard violation
        if daily_dd >= self.daily_dd_limit:
            self.done = True
            reward -= 5  # big penalty
            info['violation'] = 'DAILY_DRAWDOWN'
        if total_dd >= self.total_dd_limit:
            self.done = True
            reward -= 10
            info['violation'] = 'TOTAL_DRAWDOWN'
        
        # Profit target reached
        profit_pct = (self.balance - self.account_size) / self.account_size
        if profit_pct >= self.profit_target:
            self.done = True
            reward += 10  # big reward
            info['success'] = True
        
        # Update peak
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        # Scale reward
        reward = np.clip(reward, -5, 5)
        
        info['balance'] = self.balance
        info['equity'] = equity
        info['profit_pct'] = profit_pct
        info['daily_dd'] = daily_dd
        info['total_dd'] = total_dd
        
        return self._get_obs(), reward, self.done, info
    
    def _close_position(self, exit_price, reason):
        """Close position and return realized PnL."""
        if self.position == 1:
            pnl = (exit_price - self.entry_price) * self.lot_size * 100
        else:
            pnl = (self.entry_price - exit_price) * self.lot_size * 100
        
        # Subtract spread cost
        pnl -= self.spread * self.lot_size * 100
        
        self.balance += pnl
        self.episode_pnl += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        self.position = 0
        self.entry_price = 0
        self.position_bars = 0
        
        return pnl / self.account_size  # normalized reward
    
    @property
    def win_rate(self):
        return self.winning_trades / max(1, self.total_trades)
