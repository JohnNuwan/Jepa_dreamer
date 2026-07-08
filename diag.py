"""
ftmo_agent/ftmo_diag_v4.py — Diagnostic V4: load best V4 model and show action probs.
"""
import torch, numpy as np, sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "octopus")

from config import SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS, HOLD, BUY, SELL, CLOSE
from features_v2 import compute_multi_tf_features, compute_correlations
from environment import MultiSymbolEnvV4
from dreamer_trainer_v2 import DreamerV3AgentV2
import pandas as pd

action_names = ["HOLD","BUY","SELL","CLOSE","SPLIT_BUY","SPLIT_SELL","PYRAMID","PARTIAL_CLOSE"]

# Load data
data_dict = {}
for symbol in ACTIVE_SYMBOLS:
    path = f"data/{symbol}_m15.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48, symbol=symbol)
    data_dict[symbol] = (features, feat_names, df_processed)

# Create env to get dimensions
env = MultiSymbolEnvV4(data_dict, lookback=48)
n_features = env.n_features
print(f"🧠 Dimensions: {n_features} features")

# Load agent
agent = DreamerV3AgentV2(
    input_dim=n_features, seq_len=48, embedding_dim=128,
    stoch_size=32, stoch_classes=32, deter_size=512,
    hidden_dim=512, action_dim=N_ACTIONS,
    horizon=30, gamma=0.997, lambda_=0.95,
)

model_path = "checkpoints_v4/best_model.pt"
if os.path.exists(model_path):
    agent.load(model_path)
    print(f"✅ Model loaded from {model_path}")
else:
    print(f"⚠️  No model at {model_path}, using untrained agent")

# Diagnostics
print("\n=== ACTION PROBABILITIES ===")
for symbol in list(data_dict.keys())[:3]:
    env.current_symbol = symbol
    env.features, env.feature_names, env.df = data_dict[symbol]
    env.spec = SYMBOLS[symbol]

    for step in [500, 5000, 15000]:
        if step >= len(env.df) - 2:
            continue
        env.current_step = step
        env.reset()
        obs = env._get_obs()
        mask = env.get_action_mask()

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device_wm)
        with torch.no_grad():
            emb = agent.jepa.encoder(obs_t)
            stoch = torch.zeros(1, agent.stoch_size, agent.stoch_classes, device=agent.device_wm)
            deter = torch.zeros(1, agent.deter_size, device=agent.device_wm)
            post, _, deter, _ = agent.world_model.transition(
                prev_stoch=stoch, prev_action=torch.zeros(1, agent.action_dim, device=agent.device_wm),
                prev_embedding=emb, next_embedding=emb)
            state = torch.cat([post.reshape(1, -1), deter], dim=-1).to(agent.device_ac)
            logits, probs = agent.actor_critic.actor(state)

        probs_np = probs.cpu().numpy()[0]
        masked_str = ",".join([action_names[i][:4] for i in range(N_ACTIONS) if mask[i]])

        print(f"\n{symbol} step={step} mask=[{masked_str}] phase={env._get_curriculum_phase()[0]}:")
        for i in range(N_ACTIONS):
            if mask[i]:
                print(f"  {action_names[i]:15s} prob={probs_np[i]:.4f}")

        for temp in [0.5, 1.0, 2.0]:
            action, _, _ = agent.get_action(obs, action_mask=mask, temperature=temp)
            print(f"  T={temp} -> {action_names[action]}")

# Validation episode
print("\n=== FULL VALIDATION ===")
symbol = "XAUUSD"
if symbol in data_dict:
    env.current_symbol = symbol
    env.features, env.feature_names, env.df = data_dict[symbol]
    env.spec = SYMBOLS[symbol]
    env.current_step = env.lookback + len(env.df) - 3000
    env.reset()

    obs = env._get_obs()
    action_counts = np.zeros(N_ACTIONS, dtype=int)
    for step in range(500):
        if env.current_step >= len(env.df) - 1:
            break
        mask = env.get_action_mask()
        action, _, _ = agent.get_action(obs, action_mask=mask, temperature=0.7)
        action_counts[action] += 1
        obs, reward, done, info = env.step(action)
        if done:
            break

    print(f"Steps: {step+1}")
    print(f"Phase: V4 with all fixes")
    print(f"Action distribution:")
    for i in range(N_ACTIONS):
        if action_counts[i] > 0:
            print(f"  {action_names[i]:15s} = {action_counts[i]} ({action_counts[i]/(step+1)*100:.1f}%)")
    print(f"Total trades: {env.total_trades}")
    print(f"Winning trades: {env.winning_trades} ({env.winning_trades/max(1,env.total_trades)*100:.1f}%)")
    print(f"Buy/Sell: {env.buy_trades}/{env.sell_trades}")
    print(f"Balance: ${env.balance:.2f}")
    print(f"PnL: {(env.balance - 10000)/100:.2f}%")