"""
ftmo_agent/train_fast.py — Parallel DreamerV3 training.
Uses 16 CPU workers for experience collection + big GPU batches.
Target: 80%+ GPU utilization on both GPUs.
"""
import sys, os, json, random, time, math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'octopus'))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, RISK_CONFIG,
                     N_ACTIONS, ACTION_NAMES, HOLD, BUY, SELL, CLOSE)
from features_v2 import compute_multi_tf_features, get_symbol_embedding
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


def collect_worker(data_dict, symbol, num_steps, result_queue, worker_id):
    """Worker process: collect experience from one symbol."""
    env = MultiSymbolEnvV3(data_dict, lookback=48)
    env.current_symbol = symbol
    env.features, env.feature_names, env.df = data_dict[symbol]
    env.spec = SYMBOLS[symbol]
    
    collected = []
    obs = env.reset()
    done = False
    
    for step in range(num_steps):
        if done:
            obs = env.reset()
        
        # Random action (exploration phase) or simple heuristic
        action_mask = env.get_action_mask()
        valid = np.where(action_mask)[0]
        action = np.random.choice(valid) if len(valid) > 0 else HOLD
        
        next_obs, reward, done, info = env.step(action)
        collected.append((obs.copy(), np.eye(N_ACTIONS)[action], reward, 
                          next_obs.copy(), float(done)))
        obs = next_obs
    
    result_queue.put((worker_id, collected))


class FastTrainer:
    """Parallel DreamerV3 trainer with maximum hardware utilization."""
    
    def __init__(self, n_workers=16, n_episodes=3000, save_dir='checkpoints_fast'):
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print("Loading data...")
        self.data_dict = load_all_symbols()
        print(f"Loaded {len(self.data_dict)} symbols")
        
        # Create agent
        env = MultiSymbolEnvV3(self.data_dict, lookback=48)
        n_features = env.n_features
        print(f"Creating DreamerV3 V2 agent (n_features={n_features})...")
        self.agent = DreamerV3AgentV2(
            input_dim=n_features, seq_len=48, embedding_dim=128,
            stoch_size=32, stoch_classes=32, deter_size=512,
            hidden_dim=512, action_dim=N_ACTIONS,
            horizon=30, gamma=0.997, lambda_=0.95,
        )
        
        self.replay = ReplayBuffer(capacity=500000)
        
        # Pre-allocate large GPU tensors for batch training
        self.device_wm = self.agent.device_wm
        self.device_ac = self.agent.device_ac
        self.batch_size = 512  # 4x bigger than before
        
        print(f"Fast trainer: {self.n_workers} workers, batch_size={self.batch_size}")
    
    def collect_parallel(self, steps_per_worker=500):
        """Collect experience from all symbols in parallel using multiprocessing."""
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        processes = []
        
        symbols = list(self.data_dict.keys())
        for i, symbol in enumerate(symbols):
            p = ctx.Process(
                target=collect_worker,
                args=(self.data_dict, symbol, steps_per_worker, result_queue, i)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        total_collected = 0
        for _ in range(len(symbols)):
            worker_id, collected = result_queue.get(timeout=120)
            for item in collected:
                obs, action_oh, reward, next_obs, done = item
                self.replay.add(obs, action_oh, reward, next_obs, done)
                total_collected += 1
        
        # Wait for all processes
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        return total_collected
    
    def collect_with_model(self, steps=1000, temperature=1.0):
        """Collect experience using the model (not random) — single process but efficient."""
        symbols = list(self.data_dict.keys())
        symbol = np.random.choice(symbols)
        
        env = MultiSymbolEnvV3(self.data_dict, lookback=48)
        env.current_symbol = symbol
        env.features, env.feature_names, env.df = self.data_dict[symbol]
        env.spec = SYMBOLS[symbol]
        
        obs = env.reset()
        done = False
        collected = 0
        
        for step in range(steps):
            if done:
                obs = env.reset()
            
            action_mask = env.get_action_mask()
            
            # Use model for action selection
            if len(self.replay) > 500:
                action, _, _ = self.agent.get_action(
                    obs, action_mask=action_mask, 
                    temperature=temperature, deterministic=False)
            else:
                valid = np.where(action_mask)[0]
                action = np.random.choice(valid) if len(valid) > 0 else HOLD
            
            next_obs, reward, done, info = env.step(action)
            self.replay.add(obs, np.eye(N_ACTIONS)[action], reward, next_obs, done)
            obs = next_obs
            collected += 1
        
        return collected
    
    def train_wm_batch(self):
        """Train World Model on a large batch — GPU intensive."""
        if len(self.replay) < self.batch_size:
            return {}
        
        batch = self.replay.sample(self.batch_size)
        return self.agent.train_world_model(batch)
    
    def train_ac_batch(self):
        """Train Actor-Critic on imagined trajectories — GPU intensive."""
        if len(self.replay) < 256:
            return {}
        
        # Use smaller batch for AC (imagination is expensive)
        batch = self.replay.sample(256)
        return self.agent.train_actor_critic(batch)
    
    def validate(self, temperature=0.5):
        """Validate on 4 symbols."""
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
                action_mask = env.get_action_mask()
                action, _, _ = self.agent.get_action(
                    obs, action_mask=action_mask,
                    temperature=temperature, deterministic=False)
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
    
    def train(self):
        best_val_pnl = -float('inf')
        metrics_log = []
        
        # Phase 1: Parallel random collection to fill replay buffer
        print("\n=== PHASE 1: Parallel random collection ===")
        for round_num in range(3):
            n = self.collect_parallel(steps_per_worker=500)
            print(f"  Round {round_num}: collected {n} transitions (total: {len(self.replay)})")
        
        # Phase 2: Training loop
        print(f"\n=== PHASE 2: Training ({self.n_episodes} episodes) ===")
        
        for episode in range(self.n_episodes):
            t0 = time.time()
            
            # Temperature and entropy schedules
            temp = max(0.5, 2.0 - 1.5 * min(1.0, episode / (self.n_episodes * 0.3)))
            entropy_coeff = max(0.001, 0.05 - 0.049 * min(1.0, episode / (self.n_episodes * 0.5)))
            self.agent.actor_critic.entropy_coeff = entropy_coeff
            
            # 1. Collect with model (1 symbol, 500 steps)
            n_collected = self.collect_with_model(steps=500, temperature=temp)
            
            # 2. Train World Model (20 batches of 512)
            wm_metrics = {}
            for _ in range(20):
                wm_metrics = self.train_wm_batch()
            
            # 3. Train Actor-Critic (20 batches of 256)
            ac_metrics = {}
            for _ in range(20):
                ac_metrics = self.train_ac_batch()
            
            dt = time.time() - t0
            
            # 4. Validate every 25 episodes
            if episode % 25 == 0 or episode == self.n_episodes - 1:
                val = self.validate(temperature=0.5)
                
                metrics = {
                    'episode': episode,
                    'time_sec': round(dt, 1),
                    'temp': round(temp, 2),
                    'entropy_coeff': round(entropy_coeff, 4),
                    'replay_size': len(self.replay),
                    'wm_loss': wm_metrics.get('wm_loss', 0),
                    'jepa_loss': wm_metrics.get('jepa_loss', 0),
                    'ac_loss': ac_metrics.get('ac_loss', 0),
                    'ac_entropy': ac_metrics.get('entropy', 0),
                    'ac_value': ac_metrics.get('value', 0),
                    **val,
                    'timestamp': datetime.now().isoformat(),
                }
                metrics_log.append(metrics)
                
                print(f"Ep {episode:4d} | {dt:.1f}s | replay={len(self.replay):6d} T={temp:.1f} | "
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
            
            # Save metrics
            if episode % 50 == 0 and metrics_log:
                with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics_log, f, indent=2)
            
            # Periodic parallel re-collection (every 100 episodes)
            if episode % 100 == 0 and episode > 0:
                n = self.collect_parallel(steps_per_worker=300)
                print(f"  📥 Re-collected {n} transitions (replay: {len(self.replay)})")
        
        self.agent.save(os.path.join(self.save_dir, 'final_model.pt'))
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_log, f, indent=2)
        print(f"\nDone! Best val PnL: {best_val_pnl:+.2%}")


if __name__ == '__main__':
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    trainer = FastTrainer(n_workers=16, n_episodes=n_ep)
    trainer.train()
