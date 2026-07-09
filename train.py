"""
ftmo_agent/train.py — Training V4 optimisé 2x RTX 3090.
- Curriculum learning (3 phases)
- Pure PnL reward
- Variable spread, slippage, commission
- Mixed precision training (AMP)
- Parallel collection
- Batch tailles optimisées pour 2x 25GB VRAM
"""
import sys, os, json, time, threading
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'octopus'))

from config import SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS
from features_v2 import compute_multi_tf_features, compute_correlations
from environment import MultiSymbolEnvV4
from dreamer_trainer_v2 import DreamerV3AgentV2, ReplayBuffer


# ─── Data Loading ─────────────────────────────────────────────

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


# ─── Environment worker for parallel collection ───────────────

class EnvWorker:
    """Threaded environment worker for parallel data collection."""
    def __init__(self, data_dict, replay_buffer, episode=0):
        self.env = MultiSymbolEnvV4(data_dict, lookback=48, curriculum_episode=episode)
        self.replay = replay_buffer
        self.lock = threading.Lock()

    def collect_episode(self, agent, max_steps=1000, temperature=1.0):
        obs = self.env.reset()
        ep_reward = 0
        for _ in range(max_steps):
            mask = self.env.get_action_mask()
            action, _, _ = agent.get_action(obs, action_mask=mask, temperature=temperature)
            next_obs, reward, done, info = self.env.step(action)
            with self.lock:
                self.replay.add(obs, np.eye(N_ACTIONS)[action], reward, next_obs, done, mask)
            ep_reward += reward
            obs = next_obs
            if done:
                break
        return ep_reward, info


# ─── Main Trainer ─────────────────────────────────────────────

class V4Trainer:
    """Trainer V4 with curriculum, AMP, parallel collect, optimized for 2x3090."""

    def __init__(self, n_episodes=3000, save_dir='checkpoints_v4'):
        self.n_episodes = n_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        print("📦 Chargement des données V4...")
        data_dict = load_all_symbols()
        print(f"   {len(data_dict)} symboles chargés")

        print("🔗 Calcul des corrélations inter-symboles...")
        self.correlations = compute_correlations(data_dict)
        print(f"   {len(self.correlations)} symboles corrélés")

        temp_env = MultiSymbolEnvV4(data_dict, lookback=48, curriculum_episode=0)
        n_features = temp_env.n_features
        print(f"🧠 Dimensions: {n_features} features")

        self.agent = DreamerV3AgentV2(
            input_dim=n_features, seq_len=48, embedding_dim=128,
            stoch_size=32, stoch_classes=32, deter_size=512,
            hidden_dim=512, action_dim=N_ACTIONS,
            horizon=15, gamma=0.997, lambda_=0.95, entropy_coeff=0.01
        )
        self.agent.temperature = 2.0  # V4 fix: start hot for exploration

        self.data_dict = data_dict
        self.replay = ReplayBuffer(capacity=4000000)  # V5: 4M transitions (~1.2 GB RAM)
        self.best_val_pnl = -999
        self.log_path = os.path.join(save_dir, 'training_v4.log')
        self.metrics_path = os.path.join(save_dir, 'metrics.json')
        self.metrics = []

        # AMP scalers (removed — integrated in dreamer_trainer if needed)

    def _train_step(self):
        """One combined WM+JEPA+AC training step. Returns metrics dict."""
        metrics = {}
        if len(self.replay) < 2000:
            return metrics

        # WM + JEPA: 8 steps per call, batch 1024 (V4.2: +weighted MSE)
        wm_losses, jepa_losses = [], []
        result = {}
        for _ in range(8):
            batch = self.replay.sample(1024)
            result = self.agent.train_world_model(batch)
            wm_losses.append(result['wm_loss'])
            jepa_losses.append(result['jepa_loss'])

        # AC: 16 steps per call, batch 512 (V5: plus d'échantillons + horizon réduit)
        ac_losses, entropies = [], []
        for _ in range(16):
            batch = self.replay.sample(512)
            result = self.agent.train_actor_critic(batch)
            ac_losses.append(result['ac_loss'])
            entropies.append(result.get('entropy', 0))

        metrics.update({
            'wm_loss': np.mean(wm_losses),
            'jepa_loss': np.mean(jepa_losses),
            'ac_loss': np.mean(ac_losses),
            'entropy': np.mean(entropies),
        })
        # Also track reward loss from last WM batch
        if 'reward_loss' in result:
            metrics['rwd_loss'] = result['reward_loss']
            # V4.2: log true value even if tiny
            if result['reward_loss'] < 0.001:
                metrics['rwd_loss_raw'] = result['reward_loss']
        if 'kl_loss' in result:
            metrics['kl_loss'] = result['kl_loss']
        return metrics

    def _collect_episode(self, agent, env):
        """Collect a single episode with epsilon-greedy exploration."""
        obs = env.reset()
        ep_reward = 0
        ep_steps = 0
        while True:
            mask = env.get_action_mask()
            # Epsilon-greedy: force random action 20% of the time
            if np.random.random() < 0.2:
                action = np.random.choice(np.where(mask)[0])
            else:
                action, _, _ = agent.get_action(obs, action_mask=mask, temperature=agent.temperature)
            next_obs, reward, done, info = env.step(action)
            self.replay.add(obs, np.eye(N_ACTIONS)[action], reward, next_obs, done, mask)
            ep_reward += reward
            ep_steps += 1
            obs = next_obs
            if done or ep_steps > 2000:
                break
        return ep_reward, ep_steps, info

    def _validate(self):
        """Run validation on XAUUSD tail data."""
        symbol = 'XAUUSD'
        if symbol not in self.data_dict:
            return 0, 0, 0, 0, '0/0'

        env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=9999)
        env.current_symbol = symbol
        env.features, env.feature_names, env.df = self.data_dict[symbol]
        env.spec = SYMBOLS[symbol]
        env.current_step = env.lookback + len(env.df) - 3000
        env.reset()
        obs = env._get_obs()

        for _ in range(500):
            if env.current_step >= len(env.df) - 1:
                break
            mask = env.get_action_mask()
            action, _, _ = self.agent.get_action(obs, action_mask=mask, temperature=0.7)
            obs, _, done, info = env.step(action)
            if done:
                break

        pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
        wr = env.winning_trades / max(1, env.total_trades) * 100
        daily_dd = (env.peak_equity - env.balance) / FTMO_CONFIG['account_size'] * 100 if hasattr(env, 'peak_equity') else 0
        if daily_dd == 0:
            daily_dd = max(0, (env.daily_start_balance - env.balance) / FTMO_CONFIG['account_size'] * 100)
        return pnl, env.total_trades, wr, daily_dd, f"{env.buy_trades}/{env.sell_trades}"

    # ── Main loop ──

    def run(self):
        log_file = open(self.log_path, 'w')
        print("\n=== PHASE 1: Random collection ===")
        self._collect_random(5000)

        print(f"\n=== PHASE 2: Training V4 ({self.n_episodes} episodes) ===")
        print(f"   2x RTX 3090 | AMP mixed precision | batch WM=1024 AC=512")
        print(f"   Replay: {len(self.replay)} transitions")
        print()

        for ep in range(self.n_episodes):
            t0 = time.time()

            # Environment
            env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=ep)

            # Collect
            ep_reward, ep_steps, info = self._collect_episode(self.agent, env)

            # Train
            train_metrics = self._train_step()

            # Temperature management
            self.agent.temperature = max(0.9, self.agent.temperature * 0.9998)
            if ep % 100 == 0 and ep > 0:
                self.agent.temperature = min(2.0, self.agent.temperature + 0.8)
                print(f"   🔄 Cyclic temperature restart → {self.agent.temperature:.2f}")

            # Validation
            val_pnl, val_trades, val_wr, val_dd, val_bs = 0, 0, 0, 0, '0/0'
            if ep % 25 == 0:
                val_pnl, val_trades, val_wr, val_dd, val_bs = self._validate()

            # Log
            t = time.time() - t0
            phase = env._get_curriculum_phase()[0]
            wml = train_metrics.get('wm_loss', 0)
            jl = train_metrics.get('jepa_loss', 0)
            acl = train_metrics.get('ac_loss', 0)
            ent = train_metrics.get('entropy', 0)

            log = (f"Ep {ep:>4d} | {t:.1f}s | buf={len(self.replay)} "
                   f"T={self.agent.temperature:.1f} | phase={phase} "
                   f"wm={wml:.4f} jepa={jl:.2f} ac={acl:.4f} ent={ent:.3f} "
                   f"post={train_metrics.get('posterior_loss',0):.4f} "
                   f"prior={train_metrics.get('prior_loss',0):.4f} "
                   f"kl={train_metrics.get('kl_loss',0):.4f} "
                   f"| val={val_pnl:+.2f}% tr={val_trades} wr={val_wr:.0f}% dd={val_dd:.2f}% {val_bs}")
            print(log)
            log_file.write(log + '\n')
            log_file.flush()

            # Metrics
            self.metrics.append({
                'episode': ep, 'reward': round(ep_reward, 2), 'steps': ep_steps,
                'val_pnl': round(val_pnl, 2), 'val_trades': val_trades,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in train_metrics.items()},
                'phase': phase, 'temperature': round(self.agent.temperature, 2),
            })
            if ep % 50 == 0:
                with open(self.metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)

            # Checkpoints
            if ep % 200 == 0:
                path = os.path.join(self.save_dir, f'ckpt_ep{ep}.pt')
                self.agent.save(path)
                print(f"   💾 Saved: {path}")

            if val_pnl > self.best_val_pnl:
                self.best_val_pnl = val_pnl
                path = os.path.join(self.save_dir, 'best_model.pt')
                self.agent.save(path)
                print(f"   🏆 NEW BEST: val_pnl={val_pnl:+.2f}%")

        path = os.path.join(self.save_dir, 'final_model.pt')
        self.agent.save(path)
        print(f"\n✅ Done! Best val PnL: {self.best_val_pnl:+.2f}%")
        log_file.close()

    def _collect_random(self, n_steps):
        """Collect random exploration data (multi-threaded)."""
        n_envs = 4
        workers = [EnvWorker(self.data_dict, self.replay, episode=0) for _ in range(n_envs)]
        collected = [0] * n_envs

        def collect_thread(idx):
            worker = workers[idx]
            env = worker.env
            while collected[idx] < n_steps // n_envs:
                obs = env.reset()
                for _ in range(200):
                    mask = env.get_action_mask()
                    action = np.random.choice(np.where(mask)[0])
                    next_obs, reward, done, _ = env.step(action)
                    with worker.lock:
                        worker.replay.add(obs, np.eye(N_ACTIONS)[action], reward, next_obs, done, mask)
                    collected[idx] += 1
                    obs = next_obs
                    if done:
                        break

        threads = [threading.Thread(target=collect_thread, args=(i,)) for i in range(n_envs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = sum(collected)
        print(f"   Collected {total} transitions ({n_envs} threads, replay: {len(self.replay)})")


if __name__ == "__main__":
    trainer = V4Trainer(n_episodes=500)
    trainer.run()