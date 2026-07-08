"""
ftmo_agent/train_dreamer_v3.py — Training V3: long run with all fixes.
- 3000 episodes
- 10 WM + 10 AC updates per episode
- Temperature-based validation (temp=0.5, not argmax)
- Entropy decay schedule
- Action masking in validation
"""
import sys, os, json, random, numpy as np, pandas as pd, torch, math
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, RISK_CONFIG,
                     ANTI_BIAS_CONFIG, N_ACTIONS, ACTION_NAMES,
                     HOLD, BUY, SELL, CLOSE)
from features_v2 import compute_multi_tf_features, get_symbol_embedding
from environment_v3 import MultiSymbolEnvV3
from dreamer_trainer_v2 import DreamerV3AgentV2, ReplayBuffer

def load_all_symbols():
    data_dict = {}
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    for symbol in ACTIVE_SYMBOLS:
        path = os.path.join(data_dir, f'{symbol}_m15.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48)
        data_dict[symbol] = (features, feat_names, df_processed)
        print(f"  {symbol}: {len(df)} bars, {features.shape[1]} features")
    return data_dict

def get_entropy_coeff(episode, max_episodes=3000):
    """Entropy decay: high early, low late."""
    start, end = 0.05, 0.001
    progress = min(1.0, episode / (max_episodes * 0.7))
    return start + (end - start) * progress

def get_temperature(episode, max_episodes=3000):
    """Temperature decay for exploration."""
    start, end = 2.0, 0.5
    progress = min(1.0, episode / (max_episodes * 0.5))
    return start + (end - start) * progress

def train(n_episodes=3000, save_dir='checkpoints_v3'):
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data...")
    data_dict = load_all_symbols()
    print(f"Loaded {len(data_dict)} symbols")
    
    env = MultiSymbolEnvV3(data_dict, lookback=48)
    n_features = env.n_features
    
    print(f"Creating DreamerV3 V2 agent...")
    agent = DreamerV3AgentV2(input_dim=n_features, seq_len=48, embedding_dim=128,
                              stoch_size=32, stoch_classes=32, deter_size=512,
                              hidden_dim=512, action_dim=N_ACTIONS,
                              horizon=30, gamma=0.997, lambda_=0.95)
    
    replay = ReplayBuffer(capacity=200000)
    
    best_val_pnl = -float('inf')
    metrics_log = []
    
    for episode in range(n_episodes):
        # === COLLECT EXPERIENCE ===
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        max_steps = min(2000, len(env.df) - env.current_step - 1)
        
        temp = get_temperature(episode, n_episodes)
        entropy_coeff = get_entropy_coeff(episode, n_episodes)
        agent.actor_critic.entropy_coeff = entropy_coeff
        
        while not done and ep_steps < max_steps:
            # Get action with temperature sampling
            action_mask = env.get_action_mask()
            
            # Exploration: random for first 20 episodes, then model
            if episode < 20:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else HOLD
            else:
                action, value, probs = agent.get_action(
                    obs, action_mask=action_mask, 
                    temperature=temp, deterministic=False)
            
            next_obs, reward, done, info = env.step(action)
            
            # Store in replay buffer
            action_oh = np.eye(N_ACTIONS)[action]
            replay.add(obs, action_oh, reward, next_obs, done, action_mask)
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
        
        # === TRAIN WORLD MODEL (10 batches) ===
        wm_metrics = {}
        if len(replay) >= 128:
            for _ in range(10):
                batch = replay.sample(128)
                wm_metrics = agent.train_world_model(batch)
        
        # === TRAIN ACTOR-CRITIC (10 batches) ===
        ac_metrics = {}
        if len(replay) >= 64:
            for _ in range(10):
                batch = replay.sample(64)
                ac_metrics = agent.train_actor_critic(batch)
        
        # === VALIDATE (every 50 episodes) ===
        if episode % 50 == 0 or episode == n_episodes - 1:
            val_results = validate(agent, data_dict, temperature=0.5)
            
            buy_ratio = env.buy_trades / max(1, env.buy_trades + env.sell_trades)
            metrics = {
                'episode': episode,
                'train_reward': ep_reward,
                'train_pnl': (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'],
                'train_trades': env.total_trades,
                'train_buy': env.buy_trades,
                'train_sell': env.sell_trades,
                'train_symbol': env.current_symbol,
                'temperature': temp,
                'entropy_coeff': entropy_coeff,
                'wm_loss': wm_metrics.get('wm_loss', 0),
                'jepa_loss': wm_metrics.get('jepa_loss', 0),
                'ac_loss': ac_metrics.get('ac_loss', 0),
                'ac_entropy': ac_metrics.get('entropy', 0),
                'replay_size': len(replay),
                **val_results,
                'timestamp': datetime.now().isoformat(),
            }
            metrics_log.append(metrics)
            
            print(f"Ep {episode:4d} | rew={ep_reward:7.1f} | "
                  f"train={metrics['train_pnl']:+.2%} [{env.current_symbol:10s}] "
                  f"B/S={env.buy_trades}/{env.sell_trades} "
                  f"T={temp:.1f} ent={entropy_coeff:.4f} | "
                  f"wm={wm_metrics.get('wm_loss', 0):.4f} "
                  f"jepa={wm_metrics.get('jepa_loss', 0):.2f} "
                  f"ac={ac_metrics.get('ac_loss', 0):.4f} | "
                  f"val={val_results['val_pnl']:+.2%} "
                  f"trades={val_results['val_trades']} "
                  f"wr={val_results['val_wr']:.0%} "
                  f"dd={val_results['val_dd']:.2%} "
                  f"B/S={val_results.get('val_buy',0)}/{val_results.get('val_sell',0)}")
            
            if val_results['val_pnl'] > best_val_pnl:
                best_val_pnl = val_results['val_pnl']
                agent.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  🏆 NEW BEST: val_pnl={best_val_pnl:+.2%}")
            
            if episode % 200 == 0:
                agent.save(os.path.join(save_dir, f'ckpt_ep{episode}.pt'))
        
        # Save metrics every 100 episodes
        if episode % 100 == 0 and metrics_log:
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_log, f, indent=2)
    
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nDone! Best val PnL: {best_val_pnl:+.2%}")

def validate(agent, data_dict, temperature=0.5, n_symbols=4):
    all_pnl, all_wr, all_dd, all_trades = [], [], [], []
    all_buys, all_sells = 0, 0
    
    for symbol in list(data_dict.keys())[:n_symbols]:
        env = MultiSymbolEnvV3(data_dict, lookback=48)
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
            action_mask = env.get_action_mask()
            action, _, _ = agent.get_action(
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

if __name__ == '__main__':
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    train(n_episodes=n_ep)
