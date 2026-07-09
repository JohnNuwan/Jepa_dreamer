"""
ftmo_agent/train.py — Training script for the FTMO trading agent.
Trains PPO with LSTM on XAUUSD M15 data with FTMO rules.
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import random
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import compute_features
from environment import FTMOTradingEnv, BUY, SELL, HOLD, CLOSE
from agent import PPOTrainer

def load_data():
    """Load XAUUSD M15 data."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'xauusd_m15.csv')
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Compute features
    df, feature_cols = compute_features(df, lookback=48)
    
    # Split: train (80%), val (20%)
    n = len(df)
    train_end = int(n * 0.8)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:].reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} bars")
    print(f"Train: {len(train_df)} ({train_df['time'].iloc[0]} → {train_df['time'].iloc[-1]})")
    print(f"Val: {len(val_df)} ({val_df['time'].iloc[0]} → {val_df['time'].iloc[-1]})")
    print(f"Features: {len(feature_cols)}")
    
    return train_df, val_df, feature_cols

def train_agent(n_episodes=500, save_dir='checkpoints'):
    """Train the PPO agent."""
    os.makedirs(save_dir, exist_ok=True)
    
    train_df, val_df, feature_cols = load_data()
    
    # Create environments
    env = FTMOTradingEnv(train_df, feature_cols, lookback=48)
    val_env = FTMOTradingEnv(val_df, feature_cols, lookback=48)
    
    # Create agent
    n_features = len(feature_cols) + 3  # + position info
    trainer = PPOTrainer(n_features=n_features, n_actions=4, 
                         lr=3e-4, batch_size=128, n_epochs=10,
                         device='cuda')
    
    print(f"Device: {trainer.device}")
    print(f"Model params: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    best_val_profit = -float('inf')
    metrics_log = []
    
    for episode in range(n_episodes):
        # === TRAINING ===
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        max_steps = min(2000, len(train_df) - env.lookback - 1)
        
        # Random start for diverse experience
        if episode > 10:
            start = np.random.randint(env.lookback, len(train_df) - max_steps - 1)
            env.current_step = start
        
        while not done and ep_steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            action, log_prob, value, entropy = trainer.model.get_action(obs_tensor)
            
            next_obs, reward, done, info = env.step(action)
            trainer.collect(obs, action, reward, log_prob, value, done, info)
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            
            # Update every 256 steps
            if len(trainer.rollout) >= 256:
                trainer.update(last_value=value if not done else 0)
        
        # Final update
        if len(trainer.rollout) >= 32:
            trainer.update(last_value=0)
        
        # === VALIDATION (every 10 episodes) ===
        if episode % 10 == 0 or episode == n_episodes - 1:
            val_profit, val_trades, val_winrate, val_dd = validate(trainer, val_env)
            
            metrics = {
                'episode': episode,
                'train_reward': ep_reward,
                'train_steps': ep_steps,
                'train_trades': env.total_trades,
                'train_profit_pct': (env.balance - env.account_size) / env.account_size,
                'train_winrate': env.win_rate,
                'val_profit_pct': val_profit,
                'val_trades': val_trades,
                'val_winrate': val_winrate,
                'val_max_dd': val_dd,
                'timestamp': datetime.now().isoformat(),
            }
            metrics_log.append(metrics)
            
            print(f"Ep {episode:4d} | reward={ep_reward:7.2f} | "
                  f"train_pnl={metrics['train_profit_pct']:+.2%} | "
                  f"val_pnl={val_profit:+.2%} | val_trades={val_trades} | "
                  f"val_wr={val_winrate:.1%} | val_dd={val_dd:.2%}")
            
            # Save best model
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                trainer.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  🏆 New best! val_pnl={val_profit:+.2%}")
            
            # Save periodic checkpoint
            if episode % 50 == 0:
                trainer.save(os.path.join(save_dir, f'checkpoint_ep{episode}.pt'))
    
    # Save final model
    trainer.save(os.path.join(save_dir, 'final_model.pt'))
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"\nTraining complete. Best val PnL: {best_val_profit:+.2%}")
    print(f"Models saved in {save_dir}/")

def validate(trainer, env):
    """Run validation episode."""
    obs = env.reset()
    done = False
    max_steps = len(env.df) - env.lookback - 1
    
    max_dd = 0
    peak = env.account_size
    
    while not done and env.current_step < max_steps:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        action, _, _, _ = trainer.model.get_action(obs_tensor, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if env.balance > peak:
            peak = env.balance
        dd = (peak - env.balance) / env.account_size
        if dd > max_dd:
            max_dd = dd
    
    profit_pct = (env.balance - env.account_size) / env.account_size
    return profit_pct, env.total_trades, env.win_rate, max_dd

if __name__ == '__main__':
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    train_agent(n_episodes=n_ep)
