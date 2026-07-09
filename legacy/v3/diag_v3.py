import sys, os, json
import numpy as np
import torch

sys.path.insert(0, '/home/aza/ftmo_agent')
sys.path.insert(0, '/home/aza/ftmo_agent/octopus')

from config import SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS, ACTION_NAMES, HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE
from features_v2 import compute_multi_tf_features
from environment_v3 import MultiSymbolEnvV3
from dreamer_trainer_v2 import DreamerV3AgentV2
import pandas as pd

# Load data (same as train_ultra3)
print("Loading data...")
data_dir = '/home/aza/ftmo_agent/data'
data_dict = {}
for sym in ACTIVE_SYMBOLS:
    path = os.path.join(data_dir, f'{sym}_m15.csv')
    if not os.path.exists(path):
        print(f"  {sym}: NO DATA at {path}")
        continue
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48)
    data_dict[sym] = (features.astype(np.float32), feat_names, df_processed)
    print(f"  {sym}: {len(df)} bars, {features.shape[1]} features")

print(f"Loaded {len(data_dict)} symbols")

# Create agent
n_features = data_dict[ACTIVE_SYMBOLS[0]][0].shape[1] + 8 + 5
print(f"n_features = {n_features}")
agent = DreamerV3AgentV2(input_dim=n_features)

# Load best model
ckpt_path = '/home/aza/ftmo_agent/checkpoints_ultra3/best_model.pt'
if os.path.exists(ckpt_path):
    agent.load(ckpt_path)
    print(f"Loaded {ckpt_path}")
else:
    print("No checkpoint found!")

# Test action selection on multiple symbols
print("\n=== ACTION PROBABILITIES PER SYMBOL ===")
for sym in list(data_dict.keys())[:4]:
    env = MultiSymbolEnvV3(data_dict, lookback=48)
    env.current_symbol = sym
    env.features, env.feature_names, env.df = data_dict[sym]
    env.spec = SYMBOLS[sym]
    env.current_step = env.lookback + 100
    env.reset()
    
    obs = env._get_obs()
    mask = env.get_action_mask()
    
    # Get action with different temperatures
    for temp in [0.5, 1.0, 2.0]:
        action, value, probs = agent.get_action(obs, action_mask=mask, temperature=temp)
        action_name = ACTION_NAMES[action]
        print(f"  {sym} T={temp:.1f}: action={action_name}({action}) value={value:.4f}")
        print(f"    probs: {['%.4f' % p for p in probs]}")
        print(f"    mask:  {mask}")
    
    # Also test deterministic
    action, value, probs = agent.get_action(obs, action_mask=mask, temperature=0)
    print(f"  {sym} DET: action={ACTION_NAMES[action]}({action}) value={value:.4f}")
    print(f"    probs: {['%.4f' % p for p in probs]}")
    print()

# Run 500 steps of validation to see trade frequency
print("\n=== 500-STEP VALIDATION RUN ===")
for sym in list(data_dict.keys())[:4]:
    env = MultiSymbolEnvV3(data_dict, lookback=48)
    env.current_symbol = sym
    env.features, env.feature_names, env.df = data_dict[sym]
    env.spec = SYMBOLS[sym]
    env.current_step = env.lookback + 100
    env.reset()
    
    trades = 0
    holds = 0
    actions_taken = []
    obs = env._get_obs()
    
    for step in range(500):
        mask = env.get_action_mask()
        action, _, _ = agent.get_action(obs, action_mask=mask, temperature=0.5)
        actions_taken.append(action)
        if action == HOLD:
            holds += 1
        elif action in (BUY, SELL, SPLIT_BUY, SPLIT_SELL):
            trades += 1
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size']
    from collections import Counter
    action_counts = Counter(actions_taken)
    print(f"  {sym}: trades={trades} holds={holds} pnl={pnl:+.2%}")
    print(f"    actions: {[(ACTION_NAMES[a], c) for a, c in sorted(action_counts.items())]}")
