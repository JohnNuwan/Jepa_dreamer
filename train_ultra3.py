"""
ftmo_agent/train_ultra3.py — Ultra-fast DreamerV3 V3 with all fixes.
- Cyclic temperature restart (never below 0.7)
- Dynamic entropy floor
- More WM early, more AC later
- Multi-temperature validation
- Entropy collapse detection
"""
import sys, os, json, random, time, math, threading
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'octopus'))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS,
                     HOLD, BUY, SELL, CLOSE)
from features_v2 import compute_multi_tf_features
from environment_v3 import MultiSymbolEnvV3
from dreamer_trainer_v2 import DreamerV3AgentV2, ReplayBuffer


def load_all_symbols():
    data_dict = {}
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    for symbol in ACTIVE_SYMBOLS:
        path = os.path.join(data_dir, f'{symbol}_m15.csv')
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48)
        data_dict[symbol] = (features, feat_names, df_processed)
        print(f"  {symbol}: {len(df)} bars, {features.shape[1]} features")
    return data_dict


class UltraTrainer3:
    """V3 trainer with all entropy/JEPA/reward fixes."""
    
    def __init__(self, n_episodes=3000, save_dir='checkpoints_ultra3'):
        self.n_episodes = n_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print("Loading data...")
        self.data_dict = load_all_symbols()
        print(f"Loaded {len(self.data_dict)} symbols")
        
        env = MultiSymbolEnvV3(self.data_dict, lookback=48)
        n_features = env.n_features
        print(f"Creating agent (n_features={n_features})...")
        self.agent = DreamerV3AgentV2(
            input_dim=n_features, seq_len=48, embedding_dim=128,
            stoch_size=32, stoch_classes=32, deter_size=512,
            hidden_dim=512, action_dim=N_ACTIONS,
            horizon=30, gamma=0.997, lambda_=0.95,
        )
        
        self.replay = ReplayBuffer(capacity=500000)
        self.device_wm = self.agent.device_wm
        
        self.wm_batch_size = 512
        self.ac_batch_size = 256
        
        self.collect_running = False
        self.collect_lock = threading.Lock()
        
        print("Ultra3: all fixes applied (entropy floor, JEPA stability, reward shaping)")
    
    def fill_replay(self):
        print("\n=== PHASE 1: Random collection ===")
        symbols = list(self.data_dict.keys())
        total = 0
        
        from concurrent.futures import ThreadPoolExecutor
        
        def collect_random(symbol, n_steps=500):
            env = MultiSymbolEnvV3(self.data_dict, lookback=48)
            env.current_symbol = symbol
            env.features, env.feature_names, env.df = self.data_dict[symbol]
            env.spec = SYMBOLS[symbol]
            env.reset()
            
            count = 0
            for _ in range(n_steps):
                if env.current_step >= len(env.df) - 2:
                    env.reset()
                mask = env.get_action_mask()
                valid = np.where(mask)[0]
                action = np.random.choice(valid) if len(valid) > 0 else HOLD
                obs = env._get_obs().copy()
                next_obs, reward, done, _ = env.step(action)
                with self.collect_lock:
                    self.replay.add(obs, np.eye(N_ACTIONS)[action], 
                                    reward, next_obs.copy(), float(done))
                count += 1
            return count
        
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(collect_random, sym) for sym in symbols]
            for f in futures:
                total += f.result()
        
        print(f"  Collected {total} transitions (replay: {len(self.replay)})")
    
    def background_collect(self, temperature=1.0, n_steps=200):
        self.collect_running = True
        symbols = list(self.data_dict.keys())
        
        def collect_one(symbol):
            env = MultiSymbolEnvV3(self.data_dict, lookback=48)
            env.current_symbol = symbol
            env.features, env.feature_names, env.df = self.data_dict[symbol]
            env.spec = SYMBOLS[symbol]
            env.reset()
            
            for _ in range(n_steps):
                if env.current_step >= len(env.df) - 2:
                    env.reset()
                mask = env.get_action_mask()
                action, _, _ = self.agent.get_action(
                    env._get_obs(), action_mask=mask,
                    temperature=temperature, deterministic=False)
                obs = env._get_obs().copy()
                next_obs, reward, done, _ = env.step(action)
                with self.collect_lock:
                    self.replay.add(obs, np.eye(N_ACTIONS)[action],
                                    reward, next_obs.copy(), float(done))
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=7) as executor:
            list(executor.map(collect_one, symbols))
        
        self.collect_running = False
    
    def train(self):
        self.fill_replay()
        
        best_val_pnl = -float('inf')
        metrics_log = []
        low_entropy_count = 0
        
        print(f"\n=== PHASE 2: Ultra training V3 ({self.n_episodes} episodes) ===")
        
        for episode in range(self.n_episodes):
            t0 = time.time()
            
            # FIX: Cyclic temperature (never below 0.7)
            # Cycle: start at 2.0, decay to 0.7 over 500 episodes, then restart
            cycle_pos = (episode % 300) / 300.0
            temp = 1.0 + 1.0 * (1.0 - cycle_pos)  # 2.0 → 0.7, then restart
            
            # FIX: Entropy coeff — starts high, never below 0.02
            entropy_progress = min(1.0, episode / (self.n_episodes * 0.3))
            base_entropy = max(0.05, 0.15 * (1.0 - 0.8 * entropy_progress))
            self.agent.base_entropy_coeff = base_entropy
            
            # FIX: More WM early, more AC later
            wm_progress = min(1.0, episode / (self.n_episodes * 0.2))
            n_wm_batches = int(20 + 10 * (1.0 - wm_progress))  # 30→15
            n_ac_batches = int(15 + 25 * wm_progress)          # 10→25
            
            # 1. Background collection
            collect_thread = threading.Thread(
                target=self.background_collect,
                args=(temp, 150),
                daemon=True
            )
            collect_thread.start()
            
            # 2. Train WM
            wm_metrics = {}
            for _ in range(n_wm_batches):
                if len(self.replay) >= self.wm_batch_size:
                    batch = self.replay.sample(self.wm_batch_size)
                    wm_metrics = self.agent.train_world_model(batch)
            
            # 3. Train AC
            ac_metrics = {}
            for _ in range(n_ac_batches):
                if len(self.replay) >= self.ac_batch_size:
                    batch = self.replay.sample(self.ac_batch_size)
                    ac_metrics = self.agent.train_actor_critic(batch)
            
            collect_thread.join(timeout=15)
            
            dt = time.time() - t0
            current_entropy = ac_metrics.get('entropy', 0)
            
            # FIX: Entropy collapse detection
            if current_entropy < 0.1:
                low_entropy_count += 1
            else:
                low_entropy_count = max(0, low_entropy_count - 1)
            
            # 4. Validate
            if episode % 25 == 0 or episode == self.n_episodes - 1:
                # FIX: Multi-temperature validation
                val_low = self.validate(temperature=0.5)
                val_high = self.validate(temperature=1.0)
                val = val_low if val_low['val_trades'] > 0 else val_high
                val['val_pnl_low_t'] = val_low['val_pnl']
                val['val_pnl_high_t'] = val_high['val_pnl']
                
                metrics = {
                    'episode': episode,
                    'time_sec': round(dt, 2),
                    'temp': round(temp, 2),
                    'entropy_coeff': round(self.agent.actor_critic.entropy_coeff, 4),
                    'replay_size': len(self.replay),
                    'wm_loss': wm_metrics.get('wm_loss', 0),
                    'jepa_loss': wm_metrics.get('jepa_loss', 0),
                    'reward_loss': wm_metrics.get('reward_loss', 0),
                    'kl_loss': wm_metrics.get('kl_loss', 0),
                    'ac_loss': ac_metrics.get('ac_loss', 0),
                    'ac_entropy': current_entropy,
                    'critic_loss': ac_metrics.get('critic_loss', 0),
                    'advantage': ac_metrics.get('advantage', 0),
                    'lambda_return': ac_metrics.get('lambda_return', 0),
                    'low_entropy_count': low_entropy_count,
                    **val,
                    'timestamp': datetime.now().isoformat(),
                }
                metrics_log.append(metrics)
                
                print(f"Ep {episode:4d} | {dt:.2f}s | replay={len(self.replay):6d} T={temp:.1f} | "
                      f"wm={wm_metrics.get('wm_loss',0):.4f} jepa={wm_metrics.get('jepa_loss',0):.2f} "
                      f"ac={ac_metrics.get('ac_loss',0):.4f} ent={current_entropy:.3f} | "
                      f"val={val['val_pnl']:+.2%} trades={val['val_trades']} "
                      f"wr={val['val_wr']:.0%} dd={val['val_dd']:.2%} "
                      f"B/S={val['val_buy']}/{val['val_sell']}")
                
                if val['val_pnl'] > best_val_pnl and val['val_trades'] > 0:
                    best_val_pnl = val['val_pnl']
                    self.agent.save(os.path.join(self.save_dir, 'best_model.pt'))
                    print(f"  🏆 NEW BEST: val_pnl={best_val_pnl:+.2%}")
                
                if episode % 200 == 0:
                    self.agent.save(os.path.join(self.save_dir, f'ckpt_ep{episode}.pt'))
            
            if episode % 50 == 0 and metrics_log:
                with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics_log, f, indent=2)
        
        self.agent.save(os.path.join(self.save_dir, 'final_model.pt'))
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_log, f, indent=2)
        print(f"\nDone! Best val PnL: {best_val_pnl:+.2%}")
    
    def validate(self, temperature=0.5):
        all_pnl, all_wr, all_dd, all_trades = [], [], [], []
        all_buys, all_sells = 0, 0
        
        for symbol in list(self.data_dict.keys())[:4]:
            env = MultiSymbolEnvV3(self.data_dict, lookback=48)
            env.reset()
            # FIX: override symbol AFTER reset (reset picks random symbol)
            env.current_symbol = symbol
            env.features, env.feature_names, env.df = self.data_dict[symbol]
            env.spec = SYMBOLS[symbol]
            env.current_step = env.lookback + np.random.randint(0, max(1, len(env.df) - env.lookback - 2000))
            
            obs = env._get_obs()
            done = False
            max_steps = min(1000, len(env.df) - env.current_step - 1)
            peak = env.balance
            max_dd = 0
            
            while not done and env.current_step < max_steps:
                mask = env.get_action_mask()
                action, _, _ = self.agent.get_action(
                    obs, action_mask=mask, temperature=temperature)
                obs, reward, done, info = env.step(action)
                if env.balance > peak: peak = env.balance
                dd = (peak - env.balance) / FTMO_CONFIG['account_size']
                max_dd = max(max_dd, dd)
            
            pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size']
            all_pnl.append(pnl)
            all_trades.append(env.total_trades)
            all_wr.append(env.winning_trades)
            all_dd.append(max_dd)
            all_buys += env.buy_trades
            all_sells += env.sell_trades
        
        return {
            'val_pnl': float(np.mean(all_pnl)),
            'val_wr': float(np.mean(all_wr) / max(1, np.mean(all_trades))),
            'val_dd': float(np.max(all_dd)),
            'val_trades': int(np.mean(all_trades)),
            'val_buy': all_buys,
            'val_sell': all_sells,
        }


if __name__ == '__main__':
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    UltraTrainer3(n_episodes=n_ep).train()
