"""
ftmo_agent/train_v2.py — Training V2: multi-symbol, multi-TF, anti-bias.
"""
import sys, os, json, random, numpy as np, pandas as pd, torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ACTIVE_SYMBOLS, SYMBOLS, FTMO_CONFIG, ACTION_NAMES, N_ACTIONS
from features_v2 import compute_multi_tf_features
from environment_v2 import MultiSymbolEnv
from agent_v2 import PPOTrainerV2

def load_all_symbols():
    """Load and compute features for all symbols."""
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
        
        # Compute multi-TF features
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48)
        data_dict[symbol] = (features, feat_names, df_processed)
        
        print(f"  {symbol}: {len(df)} bars, {features.shape[1]} features")
    
    return data_dict

def train(n_episodes=1000, save_dir='checkpoints_v2'):
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data for all symbols...")
    data_dict = load_all_symbols()
    print(f"Loaded {len(data_dict)} symbols")
    
    # Create environment
    env = MultiSymbolEnv(data_dict, lookback=48)
    print(f"Environment: n_features={env.n_features}, n_actions={N_ACTIONS}")
    
    # Create trainer
    trainer = PPOTrainerV2(
        n_features=env.n_features, n_actions=N_ACTIONS,
        lr=3e-4, batch_size=128, n_epochs=10, device='cuda',
    )
    
    best_val_pnl = -float('inf')
    metrics_log = []
    
    for episode in range(n_episodes):
        # === TRAIN ===
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        max_steps = min(2000, len(env.df) - env.current_step - 1)
        
        while not done and ep_steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            action, log_prob, value, entropy = trainer.model.get_action(obs_tensor)
            
            next_obs, reward, done, info = env.step(action)
            trainer.collect(obs, action, reward, log_prob, value, done, info)
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            
            if len(trainer.rollout) >= 256:
                trainer.update(last_value=value if not done else 0)
        
        if len(trainer.rollout) >= 32:
            trainer.update(last_value=0)
        
        # === VALIDATE (every 20 episodes) ===
        if episode % 20 == 0 or episode == n_episodes - 1:
            val_results = validate(trainer, data_dict)
            
            metrics = {
                'episode': episode,
                'train_reward': ep_reward,
                'train_profit_pct': (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'],
                'train_trades': env.total_trades,
                'train_winrate': env.winning_trades / max(1, env.total_trades),
                'train_buy': env.buy_trades,
                'train_sell': env.sell_trades,
                'train_symbol': env.current_symbol,
                **val_results,
                'timestamp': datetime.now().isoformat(),
            }
            metrics_log.append(metrics)
            
            buy_ratio = env.buy_trades / max(1, env.buy_trades + env.sell_trades)
            print(f"Ep {episode:4d} | rew={ep_reward:7.1f} | "
                  f"train={metrics['train_profit_pct']:+.2%} [{env.current_symbol:10s}] "
                  f"B/S={env.buy_trades}/{env.sell_trades} | "
                  f"val_pnl={val_results['val_profit_pct']:+.2%} "
                  f"val_trades={val_results['val_trades']} "
                  f"wr={val_results['val_winrate']:.0%} "
                  f"dd={val_results['val_max_dd']:.2%} "
                  f"B/S={val_results.get('val_buy',0)}/{val_results.get('val_sell',0)}")
            
            if val_results['val_profit_pct'] > best_val_pnl:
                best_val_pnl = val_results['val_profit_pct']
                trainer.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  🏆 NEW BEST: val_pnl={best_val_pnl:+.2%}")
            
            if episode % 100 == 0:
                trainer.save(os.path.join(save_dir, f'ckpt_ep{episode}.pt'))
    
    trainer.save(os.path.join(save_dir, 'final_model.pt'))
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"\nDone! Best val PnL: {best_val_pnl:+.2%}")

def validate(trainer, data_dict, n_trials=3):
    """Validate on each symbol and average."""
    all_pnl = []
    all_trades = []
    all_wins = []
    all_dd = []
    all_buys = 0
    all_sells = 0
    
    for symbol in list(data_dict.keys())[:4]:  # validate on 4 symbols
        for trial in range(n_trials):
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
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                action, _, _, _ = trainer.model.get_action(obs_tensor, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                if env.balance > peak:
                    peak = env.balance
                dd = (peak - env.balance) / FTMO_CONFIG['account_size']
                max_dd = max(max_dd, dd)
            
            pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size']
            all_pnl.append(pnl)
            all_trades.append(env.total_trades)
            all_wins.append(env.winning_trades)
            all_dd.append(max_dd)
            all_buys += env.buy_trades
            all_sells += env.sell_trades
    
    return {
        'val_profit_pct': np.mean(all_pnl),
        'val_trades': int(np.mean(all_trades)),
        'val_winrate': np.mean(all_wins) / max(1, np.mean(all_trades)),
        'val_max_dd': np.max(all_dd),
        'val_buy': all_buys,
        'val_sell': all_sells,
    }

if __name__ == '__main__':
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    train(n_episodes=n_ep)
