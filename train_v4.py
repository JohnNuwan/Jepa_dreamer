"""
ftmo_agent/train_v4.py — Training V4 with all fixes:
- Curriculum learning (3 phases)
- Pure PnL reward
- Variable spread, slippage, commission
- V4 features with correlations
- Same DreamerV3 architecture
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'octopus'))

from config import SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS
from features_v2 import compute_multi_tf_features, compute_correlations
from environment_v4 import MultiSymbolEnvV4
from dreamer_trainer_v2 import DreamerV3AgentV2, ReplayBuffer


def load_all_symbols():
    """Load data for all symbols and compute V4 features."""
    data_dict = {}
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    for symbol in ACTIVE_SYMBOLS:
        path = os.path.join(data_dir, f'{symbol}_m15.csv')
        if not os.path.exists(path):
            print(f"  ⚠️ {symbol}: pas de fichier {path}")
            continue
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48, symbol=symbol)
        data_dict[symbol] = (features, feat_names, df_processed)
        print(f"  ✅ {symbol}: {len(df)} bars → {features.shape[1]} features")
    return data_dict


class V4Trainer:
    """Trainer V4 with curriculum learning."""

    def __init__(self, n_episodes=3000, save_dir='checkpoints_v4'):
        self.n_episodes = n_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        print("📦 Chargement des données V4...")
        data_dict = load_all_symbols()
        print(f"   {len(data_dict)} symboles chargés")

        # Compute correlations
        print("🔗 Calcul des corrélations inter-symboles...")
        self.correlations = compute_correlations(data_dict)
        print(f"   {len(self.correlations)} symboles corrélés")

        # Create temporary env to get feature dimensions
        temp_env = MultiSymbolEnvV4(data_dict, lookback=48, curriculum_episode=0)
        n_features = temp_env.n_features
        print(f"🧠 Dimensions d'entrée: {n_features} features")

        self.agent = DreamerV3AgentV2(
            input_dim=n_features, seq_len=48, embedding_dim=128,
            stoch_size=32, stoch_classes=32, deter_size=512,
            hidden_dim=512, action_dim=N_ACTIONS,
            horizon=30, gamma=0.997, lambda_=0.95,
        )

        self.data_dict = data_dict
        self.replay = ReplayBuffer(capacity=500000)
        self.best_val_pnl = -999
        self.log_path = os.path.join(save_dir, 'training_v4.log')
        self.metrics_path = os.path.join(save_dir, 'metrics.json')
        self.metrics = []

    def run(self):
        log_file = open(self.log_path, 'w')
        print("\n=== PHASE 1: Random collection ===")
        self._collect_random(3500)

        print(f"\n=== PHASE 2: Ultra training V4 ({self.n_episodes} episodes) ===")

        for ep in range(self.n_episodes):
            t0 = time.time()

            # Create env with curriculum phase based on episode
            env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=ep)
            obs = env.reset()

            ep_reward = 0
            ep_steps = 0
            while True:
                mask = env.get_action_mask()
                action, log_prob, value = self.agent.get_action(obs, action_mask=mask, temperature=1.0)
                next_obs, reward, done, info = env.step(action)

                # Store in replay
                self.replay.add(obs, action, reward, done, next_obs, mask)
                ep_reward += reward
                ep_steps += 1
                obs = next_obs
                if done or ep_steps > 2000:
                    break

            # Training step every episode
            wm_loss, jepa_loss, ac_loss, entropy = 0, 0, 0, 0
            if len(self.replay) > 1000:
                # World model + JEPA training
                wm_loss, jepa_loss = 0, 0
                for _ in range(2):
                    batch = self.replay.sample(512)
                    wml, jl = self.agent.train_wm(batch)
                    wm_loss += wml
                    jepa_loss += jl

                # Actor-Critic training
                ac_loss = 0
                ent = 0
                for _ in range(1):
                    batch = self.replay.sample(256)
                    acl, e = self.agent.train_ac(batch)
                    ac_loss += acl
                    ent = e

            # Temperature management
            self.agent.temperature = max(0.7, self.agent.temperature * 0.9995)
            if ep % 500 == 0 and ep > 0:
                self.agent.temperature = min(2.0, self.agent.temperature + 0.5)
                print(f"   🔄 Cyclic temperature restart → {self.agent.temperature:.2f}")

            # Validation
            val_pnl = 0
            val_trades = 0
            if ep % 25 == 0:
                val_pnl, val_trades, val_wr, val_dd, val_bs = self._validate()

            # Logging
            t = time.time() - t0
            phase = env._get_curriculum_phase()[0]
            log = (f"Ep {ep:>4d} | {t:.2f}s | replay={len(self.replay)} "
                   f"T={self.agent.temperature:.1f} | phase={phase} "
                   f"wm={wm_loss:.4f} jepa={jepa_loss:.2f} ac={ac_loss:.4f} ent={ent:.3f} "
                   f"| val={val_pnl:+.2f}% trades={val_trades} wr={val_wr:.0f}% dd={val_dd:.2f}% B/S={val_bs}")

            print(log)
            log_file.write(log + '\n')
            log_file.flush()

            # Save metrics
            self.metrics.append({
                'episode': ep, 'reward': ep_reward, 'steps': ep_steps,
                'val_pnl': val_pnl, 'val_trades': val_trades,
                'wm_loss': wm_loss, 'jepa_loss': jepa_loss,
                'ac_loss': ac_loss, 'entropy': entropy,
                'phase': phase, 'temperature': self.agent.temperature,
            })
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)

            # Save checkpoints
            if ep % 200 == 0:
                path = os.path.join(self.save_dir, f'ckpt_ep{ep}.pt')
                self.agent.save(path)
                print(f"   💾 Saved: {path}")

            if val_pnl > self.best_val_pnl:
                self.best_val_pnl = val_pnl
                path = os.path.join(self.save_dir, 'best_model.pt')
                self.agent.save(path)
                print(f"   🏆 NEW BEST: val_pnl={val_pnl:+.2f}%")

        # Final save
        path = os.path.join(self.save_dir, 'final_model.pt')
        self.agent.save(path)
        print(f"\n✅ Done! Best val PnL: {self.best_val_pnl:+.2f}%")
        log_file.close()

    def _collect_random(self, n_steps):
        """Phase 1: random exploration."""
        env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=0)
        collected = 0
        while collected < n_steps:
            obs = env.reset()
            for _ in range(200):
                mask = env.get_action_mask()
                action = np.random.choice(np.where(mask)[0])
                next_obs, reward, done, info = env.step(action)
                self.replay.add(obs, action, reward, done, next_obs, mask)
                collected += 1
                obs = next_obs
                if done:
                    break
        print(f"   Collected {collected} transitions (replay: {len(self.replay)})")

    def _validate(self):
        """Run validation episode on XAUUSD at current step."""
        symbol = 'XAUUSD'
        if symbol not in self.data_dict:
            return 0, 0, 0, 0, '0/0'

        env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=9999)
        env.current_symbol = symbol
        env.features, env.feature_names, env.df = self.data_dict[symbol]
        env.spec = SYMBOLS[symbol]
        env.current_step = env.lookback + len(env.df) - 3000  # use last part of data
        env.reset()

        obs = env._get_obs()
        for step in range(500):
            if env.current_step >= len(env.df) - 1:
                break
            mask = env.get_action_mask()
            action, _, _ = self.agent.get_action(obs, action_mask=mask, temperature=0.7)
            obs, reward, done, info = env.step(action)
            if done:
                break

        pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
        wr = env.winning_trades / max(1, env.total_trades) * 100
        dd = (env.peak_equity - env.balance) / FTMO_CONFIG['account_size'] * 100 if hasattr(env, 'peak_equity') else 0

        # use daily dd if peak_equity not available
        if 'peak_equity' not in dir(env) or dd == 0:
            dd = max(0, (env.daily_start_balance - env.balance) / FTMO_CONFIG['account_size'] * 100)

        return pnl, env.total_trades, wr, dd, f"{env.buy_trades}/{env.sell_trades}"


if __name__ == "__main__":
    trainer = V4Trainer(n_episodes=3000)
    trainer.run()