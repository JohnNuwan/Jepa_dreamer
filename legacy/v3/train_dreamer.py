"""
ftmo_agent/train_dreamer.py — DreamerV3 training with multi-symbol support.
Uses both GPUs: World Model on GPU0, Actor-Critic on GPU1.
"""
import sys, os, json, random, numpy as np, pandas as pd, torch
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, RISK_CONFIG,
                     ANTI_BIAS_CONFIG, N_ACTIONS, ACTION_NAMES,
                     HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL,
                     PYRAMID, PARTIAL_CLOSE)
from features_v2 import compute_multi_tf_features, get_symbol_embedding
from environment_v2 import MultiSymbolEnv
from dreamer_trainer import DreamerV3Agent, ReplayBuffer

def load_all_symbols():
    data_dict = {}
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    for symbol in ACTIVE_SYMBOLS:
        path = os.path.join(data_dir, f'{symbol}_m15.csv')
        if not os.path.exists(path):
            print(f"SKIP {symbol}: no data")
            continue
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48)
        data_dict[symbol] = (features, feat_names, df_processed)
        print(f"  {symbol}: {len(df)} bars, {features.shape[1]} features")
    return data_dict

def train_dreamer(n_episodes=500, save_dir='checkpoints_dreamer'):
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data...")
    data_dict = load_all_symbols()
    print(f"Loaded {len(data_dict)} symbols")
    
    env = MultiSymbolEnv(data_dict, lookback=48)
    n_features = env.n_features
    
    print(f"Creating DreamerV3 agent (n_features={n_features})...")
    agent = DreamerV3Agent(input_dim=n_features, seq_len=48, embedding_dim=128,
                           stoch_size=32, stoch_classes=32, deter_size=512,
                           hidden_dim=512, action_dim=N_ACTIONS,
                           horizon=15, gamma=0.997, lambda_=0.95)
    
    replay = ReplayBuffer(capacity=50000)
    
    best_val_pnl = -float('inf')
    metrics_log = []
    
    for episode in range(n_episodes):
        # === COLLECT EXPERIENCE ===
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        max_steps = min(2000, len(env.df) - env.current_step - 1)
        
        while not done and ep_steps < max_steps:
            # Get action from DreamerV3 (every 4 steps, else random for exploration)
            if episode < 10 or ep_steps % 4 == 0:
                action = np.random.randint(N_ACTIONS)
            else:
                action, value = agent.get_action(obs, deterministic=False)
            
            next_obs, reward, done, info = env.step(action)
            
            # Store in replay buffer
            replay.add(obs, np.eye(N_ACTIONS)[action], reward, next_obs, done)
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
        
        # === TRAIN WORLD MODEL + ACTOR-CRITIC ===
        wm_metrics = {}
        ac_metrics = {}
        
        if len(replay) >= 128:
            # Train World Model (multiple batches)
            for _ in range(3):
                batch = replay.sample(128)
                wm_metrics = agent.train_world_model(batch)
            
            # Train Actor-Critic on imagined trajectories
            for _ in range(3):
                batch = replay.sample(64)
                ac_metrics = agent.train_actor_critic(batch)
        
        # === VALIDATE (every 20 episodes) ===
        if episode % 20 == 0 or episode == n_episodes - 1:
            val_results = validate_dreamer(agent, data_dict)
            
            buy_ratio = env.buy_trades / max(1, env.buy_trades + env.sell_trades)
            metrics = {
                'episode': episode,
                'train_reward': ep_reward,
                'train_pnl': (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'],
                'train_trades': env.total_trades,
                'train_buy': env.buy_trades,
                'train_sell': env.sell_trades,
                'train_symbol': env.current_symbol,
                'wm_loss': wm_metrics.get('wm_loss', 0),
                'jepa_loss': wm_metrics.get('jepa_loss', 0),
                'ac_loss': ac_metrics.get('ac_loss', 0),
                'ac_entropy': ac_metrics.get('entropy', 0),
                **val_results,
                'timestamp': datetime.now().isoformat(),
            }
            metrics_log.append(metrics)
            
            print(f"Ep {episode:4d} | rew={ep_reward:7.1f} | "
                  f"train={metrics['train_pnl']:+.2%} [{env.current_symbol:10s}] "
                  f"B/S={env.buy_trades}/{env.sell_trades} | "
                  f"wm={wm_metrics.get('wm_loss', 0):.3f} "
                  f"jepa={wm_metrics.get('jepa_loss', 0):.3f} "
                  f"ac={ac_metrics.get('ac_loss', 0):.3f} | "
                  f"val={val_results['val_pnl']:+.2%} "
                  f"wr={val_results['val_wr']:.0%} "
                  f"dd={val_results['val_dd']:.2%}")
            
            if val_results['val_pnl'] > best_val_pnl:
                best_val_pnl = val_results['val_pnl']
                agent.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  🏆 NEW BEST: val_pnl={best_val_pnl:+.2%}")
            
            if episode % 100 == 0:
                agent.save(os.path.join(save_dir, f'ckpt_ep{episode}.pt'))
        
        # Save metrics periodically
        if episode % 50 == 0 and metrics_log:
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_log, f, indent=2)
    
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nDone! Best val PnL: {best_val_pnl:+.2%}")

def validate_dreamer(agent, data_dict, n_symbols=4):
    all_pnl, all_wr, all_dd, all_trades = [], [], [], []
    all_buys, all_sells = 0, 0
    
    for symbol in list(data_dict.keys())[:n_symbols]:
        env = MultiSymbolEnv(data_dict, lookback=48)
        env.current_symbol = symbol
        env.features, env.feature_names, env.df = data_dict[symbol]
        env.spec = SYMBOLS[symbol]
        env.current_step = env.lookback + 100
        env.reset()
        
        obs = env._get_obs()
        done = False
        max_steps = min(1000, len(env.df) - env.current_step - 1)
        peak = env.balance
        max_dd = 0
        
        while not done and env.current_step < max_steps:
            action, _ = agent.get_action(obs, deterministic=True)
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
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    train_dreamer(n_episodes=n_ep)
