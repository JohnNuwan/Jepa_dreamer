"""
ftmo_agent/environment_v4.py — Environment V4 with ALL fixes:
- Variable spread (session/volatility-based)
- Slippage on entry, SL, and TP
- Commission MT5 ($7/lot)
- Pure PnL reward (no shaping)
- Curriculum learning (3 phases)
- Inter-symbol correlation features
- V4 features integration
"""
import numpy as np
import pandas as pd
from config import (SYMBOLS, SymbolSpec, ACTIVE_SYMBOLS, FTMO_CONFIG,
                     RISK_CONFIG, ANTI_BIAS_CONFIG, CURRICULUM_CONFIG, CORRELATION_SYMBOLS,
                     HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL,
                     PYRAMID, PARTIAL_CLOSE, N_ACTIONS, ACTION_NAMES)
from features_v2 import (compute_multi_tf_features, get_symbol_embedding_v4,
                          compute_correlations, get_correlation_features)


class Position:
    def __init__(self, symbol, direction, entry_price, lots, sl, tp,
                 spec, position_type='full', commission_paid=0):
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
        self.commission_paid = commission_paid

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

    def check_sl_tp(self, current_price, current_high, current_low):
        """V4: Check SL/TP with realistic slippage using H/L."""
        # SL: use low for long, high for short (worst case)
        if self.direction == 1:
            if current_low <= self.sl:
                return 'SL'
            if current_high >= self.tp:
                return 'TP'
        else:
            if current_high >= self.sl:
                return 'SL'
            if current_low <= self.tp:
                return 'TP'
        return None


class MultiSymbolEnvV4:
    """
    Environment V4: pure PnL, curriculum learning, realistic frictions.

    KEY CHANGES:
    - Reward = pure PnL (% of account), NO shaping tricks
    - Curriculum: phase 1 (no fees) → phase 2 (low fees) → phase 3 (full fees)
    - Variable spread based on session and volatility
    - Slippage on SL/TP execution
    - Commission per lot ($7/lot standard MT5)
    - Inter-symbol correlations in observation
    - V4 features with 70+ per TF
    """

    def __init__(self, data_dict, lookback=48, curriculum_episode=0):
        self.data_dict = data_dict
        self.symbols = list(data_dict.keys())
        self.lookback = lookback

        # Pre-compute correlations
        self.correlations = compute_correlations(data_dict)

        # Determine feature dimensions from first symbol
        first_sym = self.symbols[0]
        feat_dim = data_dict[first_sym][0].shape[1]
        corr_dim = len(CORRELATION_SYMBOLS)  # from config
        sym_emb_dim = 8
        pos_dim = 5
        self.n_features = feat_dim + corr_dim + sym_emb_dim + pos_dim

        self.curriculum_episode = curriculum_episode
        self.reset()

    def _get_curriculum_phase(self):
        """V4: Determine curriculum phase based on episode count."""
        if not CURRICULUM_CONFIG['enabled']:
            return 3, CURRICULUM_CONFIG['phase3_spread_mult'], 0.0
        cfg = CURRICULUM_CONFIG
        ep = self.curriculum_episode
        if ep < cfg['phase1_episodes']:
            return 1, cfg['phase1_spread_mult'], cfg.get('phase1_exploration_bonus', 0.005)
        elif ep < cfg['phase1_episodes'] + cfg['phase2_episodes']:
            return 2, cfg['phase2_spread_mult'], cfg.get('phase2_exploration_bonus', 0.001)
        else:
            return 3, cfg['phase3_spread_mult'], 0.0

    def _get_spread(self):
        """
        V4: Variable spread based on:
        - Symbol base spread
        - Current session (wider during off-hours)
        - Volatility regime
        - Random component for realism
        """
        spec = self.spec
        phase, mult, _ = self._get_curriculum_phase()

        base = spec.spread_points_mean * spec.pip_size
        base_std = spec.spread_points_std * spec.pip_size

        # Session multiplier
        if hasattr(self.df.index, 'hour'):

            if hasattr(self.df, 'iloc') and self.current_step < len(self.df):
                time_val = self.df.index[self.current_step] if hasattr(self.df, 'index') else pd.Timestamp.now()

            # Use the step to get approximate hour
            # Estimate from data
            time_info = self.df['time'].iloc[self.current_step] if 'time' in self.df.columns else None
        # Default: no session adjustment
        session_mult = 1.0

        # If we have time info
        if self.df is not None:
            try:
                row = self.df.iloc[self.current_step] if self.current_step < len(self.df) else self.df.iloc[-1]
                if 'time' in row:
                    t = pd.to_datetime(row['time'])
                    hour = t.hour
                    # Wider spreads during news / close
                    if hour in [8, 9, 14, 15]:  # London/NY opens
                        session_mult = 0.8  # tighter during liquid hours
                    elif hour in [0, 1, 2, 3, 4, 5, 6]:  # Asian session
                        session_mult = 1.5  # wider during thin hours
                    elif hour in [21, 22, 23]:  # Close
                        session_mult = 2.0  # much wider
            except:
                pass

        # Volatility multiplier (estimated from ATR feature)
        vol_mult = 1.0
        try:
            feat_idx = self.feature_names.index('atr_norm') if hasattr(self, 'feature_names') else -1
            if feat_idx >= 0 and self.current_step < len(self.features):
                atr_val = abs(self.features[self.current_step, feat_idx])
                # If atr > average, spread widens
                mean_atr = np.mean(np.abs(self.features[max(0, self.current_step-48):self.current_step+1, feat_idx]))
                if mean_atr > 1e-6:
                    vol_mult = min(3.0, max(0.5, atr_val / mean_atr))
        except:
            pass

        # Random component
        rnd = np.random.normal(0, base_std * 0.3)

        spread = (base * session_mult * vol_mult + rnd) * mult
        return max(spread * 0.1, min(spread, base * 10))  # clamp

    def _get_slippage(self):
        """
        V4: Slippage proportional to spread and volatility.
        Returns slippage in price units.
        """
        phase, mult, _ = self._get_curriculum_phase()
        if phase == 1:
            return 0.0

        spec = self.spec
        base_slippage = spec.spread_points_mean * spec.pip_size * spec.slippage_pct_mean
        slp_std = spec.spread_points_mean * spec.pip_size * spec.slippage_pct_std

        # More slippage during volatile periods
        vol_factor = 1.0
        try:
            feat_idx = self.feature_names.index('atr_norm')
            if feat_idx >= 0:
                atr_val = abs(self.features[self.current_step, feat_idx]) if self.current_step < len(self.features) else 0
                vol_factor = min(3.0, 1.0 + atr_val * 5)
        except:
            pass

        slippage = abs(np.random.normal(base_slippage, slp_std)) * vol_factor * mult
        return slippage

    def _get_commission(self, lots):
        """V4: Commission per trade."""
        phase, mult, _ = self._get_curriculum_phase()
        if phase == 1:
            return 0.0
        cfg = CURRICULUM_CONFIG
        if phase == 2 and cfg['phase2_commission_mult'] == 0.0:
            return 0.0
        return self.spec.commission_per_lot * lots * mult

    def reset(self):
        self.current_symbol = np.random.choice(self.symbols)
        self.features, self.feature_names, self.df = self.data_dict[self.current_symbol]
        self.spec = SYMBOLS[self.current_symbol]

        # Random start within data
        max_start = max(1, len(self.df) - self.lookback - 2000)
        self.current_step = self.lookback + np.random.randint(0, max_start)

        # Account state
        self.balance = FTMO_CONFIG['account_size']
        self.peak_balance = self.balance
        self.daily_start_balance = self.balance
        self.prev_equity = self.balance
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
        self.bars_since_last_trade = 0
        self.max_dd_exceeded = False
        self.peak_equity = self.balance

        # Episode metrics
        self.episode_reward = 0.0

        return self._get_obs()

    def _get_obs(self):
        """V4: Build observation with correlation features."""
        start = max(0, self.current_step - self.lookback)
        end = max(start + 1, self.current_step)
        end = min(end, len(self.features))

        if end - start < self.lookback:
            # Pad if not enough history
            feat = np.zeros((self.lookback, self.features.shape[1]))
            pad = self.lookback - (end - start)
            feat[pad:] = self.features[start:end]
        else:
            feat = self.features[end - self.lookback:end]

        # Correlation features
        corr_feat = get_correlation_features(self.correlations, self.current_symbol, self.current_step)
        corr_tiled = np.tile(corr_feat, (self.lookback, 1))

        # Symbol embedding
        sym_emb = get_symbol_embedding_v4(self.current_symbol)
        sym_emb_tiled = np.tile(sym_emb, (self.lookback, 1))

        # Position info
        n_positions = len(self.positions)
        # Use get_price or estimated
        current_price = self.df.iloc[min(self.current_step, len(self.df)-1)]['close'] if hasattr(self.df, 'iloc') else 0
        total_unrealized = sum(p.unrealized_pnl(current_price) for p in self.positions)
        net_direction = sum(p.direction * p.lots for p in self.positions)
        total_lots = sum(p.lots for p in self.positions)
        avg_bars = np.mean([p.bars_held for p in self.positions]) if self.positions else 0
        max_pos = FTMO_CONFIG['max_concurrent_positions']

        pos_info = np.array([[
            n_positions / max_pos,
            total_unrealized / max(1, self.balance),
            net_direction / max(1, total_lots),
            total_lots / max(1, self.spec.max_volume),
            avg_bars / max(1, FTMO_CONFIG['max_hold_bars']),
        ]])
        pos_tiled = np.tile(pos_info, (self.lookback, 1))

        obs = np.hstack([feat, corr_tiled, sym_emb_tiled, pos_tiled])
        return obs.astype(np.float32)

    def _get_price(self):
        """Get current price from data."""
        if hasattr(self.df, 'iloc'):
            idx = min(self.current_step, len(self.df) - 1)
            return self.df.iloc[idx]['close']
        return 0.0

    def _get_hilo(self):
        """Get current high/low for SL/TP checks."""
        if hasattr(self.df, 'iloc'):
            idx = min(self.current_step, len(self.df) - 1)
            return float(self.df.iloc[idx]['high']), float(self.df.iloc[idx]['low'])
        return self._get_price(), self._get_price()

    def _check_daily_reset(self):
        if 'time' in self.df.columns and self.current_step < len(self.df):
            try:
                current_day = pd.to_datetime(self.df.iloc[self.current_step]['time']).day
                if current_day != self.last_trade_day and self.last_trade_day != -1:
                    self.daily_start_balance = self.balance
                    self.trades_today = 0
                    self.cooldown_until = 0
                self.last_trade_day = current_day
            except:
                pass

    def _calc_position_size(self, entry, sl_price, risk_pct=None):
        """V4: Position sizing with tighter risk."""
        risk = risk_pct or RISK_CONFIG['risk_per_trade']
        risk_amount = self.balance * risk
        sl_distance = abs(entry - sl_price)
        if sl_distance < 1e-6:
            return self.spec.min_volume
        contract_val = self.spec.contract_size
        lots = risk_amount / (sl_distance * contract_val)
        return max(self.spec.min_volume, min(lots, self.spec.max_volume))

    def _total_risk(self):
        total = 0
        for p in self.positions:
            sl_dist = abs(p.entry_price - p.sl)
            total += sl_dist * p.lots * p.spec.contract_size
        return total / max(1, self.balance)

    def get_action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        has_position = len(self.positions) > 0
        phase, _, _ = self._get_curriculum_phase()
        max_trades = CURRICULUM_CONFIG[f'phase{phase}_max_trades']
        can_open = (self.trades_today < max_trades and
                    self.current_step >= self.cooldown_until and
                    len(self.positions) < FTMO_CONFIG['max_concurrent_positions'])

        mask[HOLD] = True
        mask[BUY] = not has_position and can_open
        mask[SELL] = not has_position and can_open
        mask[CLOSE] = has_position
        mask[SPLIT_BUY] = not has_position and can_open
        mask[SPLIT_SELL] = not has_position and can_open
        if has_position and can_open:
            upnl = self.positions[0].unrealized_pnl(self._get_price())
            mask[PYRAMID] = upnl > RISK_CONFIG['pyramid_min_profit_usd']
        else:
            mask[PYRAMID] = False
        if has_position:
            upnl = self.positions[0].unrealized_pnl(self._get_price())
            mask[PARTIAL_CLOSE] = not self.positions[0].partial_closed and upnl > 0
        return mask

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, {}

        self._check_daily_reset()
        current_price = self._get_price()
        current_high, current_low = self._get_hilo()
        reward = 0.0
        info = {}

        action_mask = self.get_action_mask()
        if not action_mask[action]:
            action = HOLD

        # ── 1. Manage existing positions (SL/TP checks with H/L) ──
        positions_to_close = []
        for i, pos in enumerate(self.positions):
            pos.bars_held += 1
            pos.update_slbe(current_price)
            exit_reason = pos.check_sl_tp(current_price, current_high, current_low)
            if exit_reason:
                positions_to_close.append((i, exit_reason))
            elif pos.bars_held >= FTMO_CONFIG['max_hold_bars']:
                positions_to_close.append((i, 'TIMEOUT'))
            # V5 ES: auto-close positions in profit after 20 bars
            elif pos.unrealized_pnl(current_price) > 0 and pos.bars_held >= 20:
                positions_to_close.append((i, 'AUTO_PROFIT'))

        for i, reason in reversed(positions_to_close):
            reward += self._close_position(i, current_price, reason)

        # ── 2. Execute action ──
        spread = self._get_spread()
        slippage = self._get_slippage()
        phase, _, _ = self._get_curriculum_phase()
        max_trades = CURRICULUM_CONFIG[f'phase{phase}_max_trades']
        can_trade = (self.trades_today < max_trades and
                     self.current_step >= self.cooldown_until and
                     len(self.positions) < FTMO_CONFIG['max_concurrent_positions'])

        traded_this_step = False
        opened_position = False
        if action == BUY and not self.positions and can_trade:
            entry = current_price + spread / 2
            reward += self._open_position(1, entry, 'full')
            traded_this_step = True
            opened_position = True
        elif action == SELL and not self.positions and can_trade:
            entry = current_price - spread / 2
            reward += self._open_position(-1, entry, 'full')
            traded_this_step = True
            opened_position = True
        elif action == SPLIT_BUY and not self.positions and can_trade:
            entry = current_price + spread / 2
            reward += self._open_position(1, entry, 'split_1', split=True)
            traded_this_step = True
            opened_position = True
        elif action == SPLIT_SELL and not self.positions and can_trade:
            entry = current_price - spread / 2
            reward += self._open_position(-1, entry, 'split_1', split=True)
            traded_this_step = True
            opened_position = True
        elif action == CLOSE and self.positions:
            for i in range(len(self.positions) - 1, -1, -1):
                reward += self._close_position(i, current_price, 'MODEL')
            traded_this_step = True
        elif action == PYRAMID and self.positions and can_trade:
            pos = self.positions[0]
            upnl = pos.unrealized_pnl(current_price)
            if upnl >= RISK_CONFIG['pyramid_min_profit_usd']:
                risk = RISK_CONFIG['risk_per_trade'] * \
                       (RISK_CONFIG['pyramid_risk_reduction'] ** (pos.pyramid_level + 1))
                entry = current_price + (spread / 2 if pos.direction == 1 else -spread / 2)
                reward += self._open_position(pos.direction, entry,
                                              f'pyramid_{pos.pyramid_level + 1}',
                                              risk_pct=risk, pyramid=True)
                traded_this_step = True
        elif action == PARTIAL_CLOSE and self.positions:
            pos = self.positions[0]
            upnl = pos.unrealized_pnl(current_price)
            if not pos.partial_closed and upnl > 0:
                close_lots = pos.lots * RISK_CONFIG['partial_close_pct']
                pnl = self._partial_close(pos, close_lots, current_price)
                reward += pnl / max(1, FTMO_CONFIG['account_size'])
                traded_this_step = True

        # ── 3. DENSE REWARD SHAPING ──
        #    Principe : 100% des steps reçoivent un reward non-nul
        #    pour que le WM apprenne une fonction reward non-triviale.
        phase, _, exp_bonus = self._get_curriculum_phase()
        cfg = CURRICULUM_CONFIG

        # 3a. Current equity
        current_equity = self.balance + sum(
            p.unrealized_pnl(current_price) for p in self.positions
        )
        equity_change = (current_equity - self.prev_equity) / FTMO_CONFIG['account_size']
        self.prev_equity = current_equity

        # 3b. Scaled PnL (×100 phase 1 pour que le WM le voie)
        pnl_scale = cfg.get(f'phase{phase}_pnl_scale', 1.0)
        reward += equity_change * pnl_scale

        # 3c. Time-decay : plus l'agent reste sans trade, plus il paie
        self.bars_since_last_trade += 1
        decay = -0.02 * (self.bars_since_last_trade / 48.0)
        decay = max(-0.02, decay)  # clamp at -2%/step
        reward += decay

        # 3d. Holding bonus : +0.005/step si position ouverte
        if self.positions:
            reward += 0.005

        # 3e. Opening bonus + reset time-decay
        if opened_position:
            reward += exp_bonus  # phase 1 = 0.20
            self.bars_since_last_trade = 0

        # 3f. Close resets counter too
        if action == CLOSE and self.positions:
            self.bars_since_last_trade = 0

        # Track peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # ── 4. FTMO checks ──
        daily_dd = (self.daily_start_balance - current_equity) / FTMO_CONFIG['account_size']
        total_dd = (self.peak_equity - current_equity) / FTMO_CONFIG['account_size']

        done = False
        if daily_dd >= FTMO_CONFIG['daily_dd_limit']:
            done = True
            info['violation'] = 'DAILY_DRAWDOWN'
            self.max_dd_exceeded = True
        if total_dd >= FTMO_CONFIG['total_dd_limit']:
            done = True
            info['violation'] = 'TOTAL_DRAWDOWN'
            self.max_dd_exceeded = True

        profit_pct = (self.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size']
        if profit_pct >= FTMO_CONFIG['profit_target'] and not self.max_dd_exceeded:
            done = True
            info['success'] = True

        # ── 5. Anti-bias (light touch) ──
        if ANTI_BIAS_CONFIG['enabled'] and self.total_trades >= ANTI_BIAS_CONFIG['min_trades_for_bias_check']:
            total = self.buy_trades + self.sell_trades
            if total > 0:
                buy_ratio = self.buy_trades / total
                if buy_ratio > ANTI_BIAS_CONFIG['max_buy_ratio']:
                    reward -= (buy_ratio - ANTI_BIAS_CONFIG['max_buy_ratio']) * \
                              ANTI_BIAS_CONFIG['penalty_weight'] * 0.01
                elif buy_ratio < 1 - ANTI_BIAS_CONFIG['max_sell_ratio']:
                    reward -= ((1 - buy_ratio) - ANTI_BIAS_CONFIG['max_sell_ratio']) * \
                              ANTI_BIAS_CONFIG['penalty_weight'] * 0.01

        # Update balance tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        self.episode_reward += reward

        info.update({
            'balance': self.balance,
            'equity': current_equity,
            'profit_pct': (self.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'],
            'daily_dd': daily_dd,
            'total_dd': total_dd,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'buy_trades': self.buy_trades,
            'sell_trades': self.sell_trades,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'action_mask': action_mask,
            'phase': self._get_curriculum_phase()[0],
        })

        return self._get_obs(), reward, done, info

    def _open_position(self, direction, entry_price, pos_type,
                       split=False, risk_pct=None, pyramid=False):
        """V4: Open position with spread and commission."""
        # SL/TP based on ATR
        try:
            feat_idx = self.feature_names.index('atr_norm') if hasattr(self, 'feature_names') else -1
            if feat_idx >= 0 and self.current_step < len(self.features):
                atr_norm = abs(self.features[self.current_step, feat_idx])
            else:
                atr_norm = 0.01
        except:
            atr_norm = 0.01

        atr = max(atr_norm * entry_price, self.spec.pip_size * 10)

        if direction == 1:
            sl = entry_price - atr * RISK_CONFIG['sl_atr_mult']
            tp = entry_price + atr * RISK_CONFIG['tp_atr_mult']
        else:
            sl = entry_price + atr * RISK_CONFIG['sl_atr_mult']
            tp = entry_price - atr * RISK_CONFIG['tp_atr_mult']

        risk = (risk_pct or RISK_CONFIG['risk_per_trade']) * (0.5 if split else 1.0)
        lots = self._calc_position_size(entry_price, sl, risk)

        if self._total_risk() + risk > RISK_CONFIG['max_risk_total']:
            return 0.0

        commission = self._get_commission(lots)
        self.balance -= commission  # Commission paid immediately

        pos = Position(
            symbol=self.current_symbol, direction=direction,
            entry_price=entry_price, lots=lots, sl=sl, tp=tp,
            spec=self.spec, position_type=pos_type, commission_paid=commission,
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

        # No reward bonus for opening — pure PnL will handle it
        return 0.0

    def _close_position(self, idx, exit_price, reason):
        """V4: Close position with slippage."""
        pos = self.positions[idx]

        # Slippage on exit
        slippage = self._get_slippage()
        if pos.direction == 1:
            exit_price -= slippage
        else:
            exit_price += slippage

        pnl = pos.unrealized_pnl(exit_price)
        self.balance += pnl
        self.episode_pnl += pnl
        self.realized_pnl += pnl

        pnl_norm = pnl / FTMO_CONFIG['account_size']
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= FTMO_CONFIG['cooldown_after_losses']:
                self.cooldown_until = self.current_step + FTMO_CONFIG['cooldown_bars']

        self.positions.pop(idx)

        # Trade completion bonus/penalty for better reward signal
        if pnl > 0:
            reward_extra = CURRICULUM_CONFIG.get('trade_completion_bonus', 0.01)
        else:
            reward_extra = -CURRICULUM_CONFIG.get('trade_completion_penalty', 0.01)

        return pnl_norm + reward_extra

    def _partial_close(self, pos, close_lots, exit_price):
        pnl_per_unit = pos.unrealized_pnl(exit_price) / max(1, pos.lots)
        pnl = pnl_per_unit * close_lots
        slippage = self._get_slippage()
        if pos.direction == 1:
            pnl -= slippage * close_lots * pos.spec.contract_size
        else:
            pnl -= slippage * close_lots * pos.spec.contract_size
        self.balance += pnl
        pos.lots -= close_lots
        pos.partial_closed = True
        return pnl