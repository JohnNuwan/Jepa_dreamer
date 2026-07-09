"""
ftmo_agent/train_ultra2.py — Ultra-fast DreamerV3 V2.
No prefetch overhead. Direct GPU batches. Async collection in background.
"""
import sys, os, json, random, time, math, threading
import numpy as np
import pandas as pd
import torch
from datetime import datetime

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


class UltraTrainer2:
    """Direct GPU training + background async collection."""
    
    def __init__(self, n_episodes=3000, save_dir='checkpoints_ultra2'):
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
        
        # Pre-allocate GPU tensors for max efficiency
        self.wm_batch_size = 512
        self.ac_batch_size = 256
        
        # Async collection state
        self.collect_running = False
        self.collect_lock = threading.Lock()
        
        print(f"Ultra2: direct GPU batches, background collection")
    
    def fill_replay(self):
        """Fill replay buffer with random data."""
        print("\n=== PHASE 1: Random collection ===")
        symbols = list(self.data_dict.keys())
        total = 0
        
        # Use threads for parallel collection
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
        """Background thread: collect experience with model."""
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
    
    def sample_gpu_batch(self, batch_size, device):
        """Sample batch and move directly to GPU."""
        batch = self.replay.sample(batch_size)
        return {
            'obs': torch.as_tensor(batch['obs'], device=device, dtype=torch.float32),
            'actions': torch.as_tensor(batch['actions'], device=device, dtype=torch.float32),
            'rewards': torch.as_tensor(batch['rewards'], device=device, dtype=torch.float32),
            'next_obs': torch.as_tensor(batch['next_obs'], device=device, dtype=torch.float32),
            'dones': torch.as_tensor(batch['dones'], device=device, dtype=torch.float32),
        }
    
    def train(self):
        self.fill_replay()
        
        best_val_pnl = -float('inf')
        metrics_log = []
        
        print(f"\n=== PHASE 2: Ultra training ({self.n_episodes} episodes) ===")
        
        for episode in range(self.n_episodes):
            t0 = time.time()
            
            temp = max(0.5, 2.0 - 1.5 * min(1.0, episode / (self.n_episodes * 0.3)))
            entropy_coeff = max(0.001, 0.05 - 0.049 * min(1.0, episode / (self.n_episodes * 0.5)))
            self.agent.actor_critic.entropy_coeff = entropy_coeff
            
            # 1. Background collection (7 threads, non-blocking)
            collect_thread = threading.Thread(
                target=self.background_collect,
                args=(temp, 150),
                daemon=True
            )
            collect_thread.start()
            
            # 2. Train WM (20 batches, direct GPU)
            wm_metrics = {}
            for _ in range(20):
                if len(self.replay) >= self.wm_batch_size:
                    batch = self.replay.sample(self.wm_batch_size)
                    wm_metrics = self.agent.train_world_model(batch)
            
            # 3. Train AC (20 batches, direct GPU)
            ac_metrics = {}
            for _ in range(20):
                if len(self.replay) >= self.ac_batch_size:
                    batch = self.replay.sample(self.ac_batch_size)
                    ac_metrics = self.agent.train_actor_critic(batch)
            
            # Wait for collection
            collect_thread.join(timeout=15)
            
            dt = time.time() - t0
            
            # 4. Validate
            if episode % 25 == 0 or episode == self.n_episodes - 1:
                val = self.validate(temperature=0.5)
                
                metrics = {
                    'episode': episode, 'time_sec': round(dt, 2),
                    'temp': round(temp, 2), 'replay_size': len(self.replay),
                    'wm_loss': wm_metrics.get('wm_loss', 0),
                    'jepa_loss': wm_metrics.get('jepa_loss', 0),
                    'ac_loss': ac_metrics.get('ac_loss', 0),
                    'ac_entropy': ac_metrics.get('entropy', 0),
                    **val,
                    'timestamp': datetime.now().isoformat(),
                }
                metrics_log.append(metrics)
                
                print(f"Ep {episode:4d} | {dt:.2f}s | replay={len(self.replay):6d} T={temp:.1f} | "
                      f"wm={wm_metrics.get('wm_loss',0):.4f} jepa={wm_metrics.get('jepa_loss',0):.2f} "
                      f"ac={ac_metrics.get('ac_loss',0):.4f} ent={ac_metrics.get('entropy',0):.3f} | "
                      f"val={val['val_pnl']:+.2%} trades={val['val_trades']} "
                      f"wr={val['val_wr']:.0%} dd={val['val_dd']:.2%} "
                      f"B/S={val['val_buy']}/{val['val_sell']}")
                
                if val['val_pnl'] > best_val_pnl:
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
            env.current_symbol = symbol
            env.features, env.feature_names, env.df = self.data_dict[symbol]
            env.spec = SYMBOLS[symbol]
            env.current_step = env.lookback + 100
            env.reset()
            
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
            'val_buy': all_buys, 'val_sell': all_sells,
        }


if __name__ == '__main__':
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    UltraTrainer2(n_episodes=n_ep).train()
