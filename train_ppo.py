"""
train_ppo.py — Entraînement PPO avec environnements parallèles.
Utilise l'architecture LSTM+Actor+Critic avec GAE et clipping PPO.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
import torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS, ACTION_NAMES,
                     HOLD, BUY, SELL, CLOSE, CURRICULUM_CONFIG)
from features_v2 import compute_multi_tf_features, compute_correlations
from environment import MultiSymbolEnvV4
from ppo_agent import PPOAgent


def load_all_symbols(data_dir='data'):
    data_dict = {}
    for symbol in ACTIVE_SYMBOLS:
        path = os.path.join(data_dir, f'{symbol}_m15.csv')
        if not os.path.exists(path):
            print(f"  ⚠️ {symbol}: pas de fichier")
            continue
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48, symbol=symbol)
        data_dict[symbol] = (features, feat_names, df_processed)
        print(f"  ✅ {symbol}: {len(df)} bars → {features.shape[1]} features")
    return data_dict


class PPOTrainer:
    def __init__(self, n_iterations=500, n_envs=8, rollout_steps=256,
                 n_epochs=4, batch_size=256, save_dir='checkpoints_ppo'):
        self.n_iterations = n_iterations
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print("📦 Chargement des données...")
        self.data_dict = load_all_symbols()
        print(f"   {len(self.data_dict)} symboles chargés")
        
        print("🔗 Corrélations inter-symboles...")
        self.correlations = compute_correlations(self.data_dict)
        
        temp_env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=0)
        self.n_features = temp_env.n_features
        
        self.agent = PPOAgent(
            input_dim=self.n_features, hidden_dim=256, action_dim=N_ACTIONS,
            lr=3e-4, gamma=0.997, lambda_=0.95, clip_eps=0.2,
            entropy_coeff=0.05, value_coeff=0.5, device='cuda:0'
        )
        
        self.best_val_pnl = -999
        self.log_path = os.path.join(save_dir, 'training_ppo.log')
        self.metrics_path = os.path.join(save_dir, 'metrics_ppo.json')
        self.metrics = []
    
    def _create_envs(self, episode=0):
        return [MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=episode)
                for _ in range(self.n_envs)]
    
    def _collect_rollout(self, envs):
        """Collecte rollout depuis n_envs parallèles."""
        n = len(envs)
        obs_buf = np.zeros((n, self.rollout_steps, 48, self.n_features), dtype=np.float32)
        action_buf = np.zeros((n, self.rollout_steps), dtype=np.int32)
        logprob_buf = np.zeros((n, self.rollout_steps), dtype=np.float32)
        reward_buf = np.zeros((n, self.rollout_steps), dtype=np.float32)
        done_buf = np.zeros((n, self.rollout_steps), dtype=np.float32)
        value_buf = np.zeros((n, self.rollout_steps), dtype=np.float32)
        mask_buf = np.zeros((n, self.rollout_steps, N_ACTIONS), dtype=bool)
        
        obs_list = [env.reset() for env in envs]
        
        for t in range(self.rollout_steps):
            obs_batch = np.stack(obs_list)
            masks_batch = np.stack([env.get_action_mask() for env in envs])
            actions, log_probs, values, probs = self.agent.get_action_batch(obs_batch, masks_batch)
            
            obs_buf[:, t] = obs_batch
            action_buf[:, t] = actions
            logprob_buf[:, t] = log_probs
            value_buf[:, t] = values
            mask_buf[:, t] = masks_batch
            
            for i, env in enumerate(envs):
                if done_buf[i, max(0, t-1)] > 0:
                    obs_list[i] = env.reset()
                    continue
                
                next_obs, reward, done, info = env.step(int(actions[i]))
                obs_list[i] = next_obs
                reward_buf[i, t] = reward
                done_buf[i, t] = float(done)
        
        # Last values
        obs_batch = np.stack(obs_list)
        last_values = self.agent.get_value(obs_batch)
        
        return obs_buf, action_buf, logprob_buf, reward_buf, done_buf, value_buf, mask_buf, last_values
    
    def _compute_gae_all(self, reward_buf, value_buf, done_buf, last_values):
        n_envs = reward_buf.shape[0]
        all_advantages = []
        all_returns = []
        for i in range(n_envs):
            adv, ret = self.agent.compute_gae(
                reward_buf[i], value_buf[i], done_buf[i], last_values[i]
            )
            all_advantages.append(adv)
            all_returns.append(ret)
        return (np.stack(all_advantages), np.stack(all_returns))
    
    def _train_epochs(self, obs_buf, action_buf, logprob_buf, returns, advantages, mask_buf):
        """Entraîne PPO sur le rollout pour n_epochs."""
        n_envs, T = obs_buf.shape[:2]
        total_steps = n_envs * T
        
        # Flatten
        obs_flat = obs_buf.reshape(-1, 48, self.n_features)
        action_flat = action_buf.reshape(-1)
        logprob_flat = logprob_buf.reshape(-1)
        returns_flat = returns.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        mask_flat = mask_buf.reshape(-1, N_ACTIONS) if mask_buf is not None else None
        
        metrics_list = []
        for _ in range(self.n_epochs):
            # Shuffle
            indices = np.random.permutation(total_steps)
            for start in range(0, total_steps, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                m = self.agent.update(
                    obs_flat[batch_idx],
                    action_flat[batch_idx],
                    logprob_flat[batch_idx],
                    returns_flat[batch_idx],
                    advantages_flat[batch_idx],
                    mask_flat[batch_idx] if mask_flat is not None else None
                )
                metrics_list.append(m)
        
        # Average metrics
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        return avg
    
    def _validate(self):
        """Validation sur XAUUSD."""
        symbol = 'XAUUSD'
        if symbol not in self.data_dict:
            return 0, 0, 0, 0
        
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
            action, _, _, _ = self.agent.get_action_batch(
                obs[np.newaxis], mask[np.newaxis], deterministic=True)
            obs, _, done, _ = env.step(int(action[0]))
            if done:
                break
        
        pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
        wr = env.winning_trades / max(1, env.total_trades) * 100
        dd = max(0, (env.daily_start_balance - env.balance) / FTMO_CONFIG['account_size'] * 100)
        return pnl, env.total_trades, wr, dd
    
    def run(self):
        log_file = open(self.log_path, 'w')
        print(f"\n=== PPO Training ({self.n_iterations} iters, {self.n_envs} envs) ===")
        print(f"   {self.rollout_steps} steps/iter, {self.n_epochs} epochs, batch={self.batch_size}")
        print(f"   Total steps: {self.n_iterations * self.n_envs * self.rollout_steps:,}")
        print()
        
        for it in range(self.n_iterations):
            t0 = time.time()
            
            # Create envs with curriculum
            envs = self._create_envs(episode=it)
            
            # Collect rollout
            obs_buf, act_buf, lp_buf, rwd_buf, done_buf, val_buf, mask_buf, last_vals = \
                self._collect_rollout(envs)
            
            # GAE
            advantages, returns = self._compute_gae_all(rwd_buf, val_buf, done_buf, last_vals)
            
            # Train
            metrics = self._train_epochs(obs_buf, act_buf, lp_buf, returns, advantages, mask_buf)
            
            # Validation
            val_pnl, val_trades, val_wr, val_dd = 0, 0, 0, 0
            if it % 10 == 0:
                val_pnl, val_trades, val_wr, val_dd = self._validate()
            
            # Log
            t = time.time() - t0
            log = (f"It {it:>4d} | {t:.1f}s | "
                   f"act={metrics['actor_loss']:.4f} crit={metrics['critic_loss']:.4f} "
                   f"ent={metrics['entropy']:.3f} "
                   f"rwd_μ={rwd_buf.mean():+.4f} "
                   f"| val={val_pnl:+.2f}% tr={val_trades} wr={val_wr:.0f}% dd={val_dd:.2f}%")
            print(log)
            log_file.write(log + '\n')
            log_file.flush()
            
            # Metrics
            self.metrics.append({
                'iteration': it,
                'reward_mean': round(float(rwd_buf.mean()), 4),
                **{k: round(v, 4) for k, v in metrics.items()},
                'val_pnl': round(val_pnl, 2),
                'val_trades': val_trades,
                'val_wr': round(val_wr, 1),
            })
            if it % 50 == 0:
                with open(self.metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            
            # Checkpoints
            if it % 100 == 0:
                self.agent.save(os.path.join(self.save_dir, f'ckpt_it{it}.pt'))
            
            if val_pnl > self.best_val_pnl:
                self.best_val_pnl = val_pnl
                self.agent.save(os.path.join(self.save_dir, 'best_model.pt'))
                print(f"   🏆 NEW BEST: {val_pnl:+.2f}%")
        
        self.agent.save(os.path.join(self.save_dir, 'final_model.pt'))
        print(f"\n✅ Done! Best val PnL: {self.best_val_pnl:+.2f}%")
        log_file.close()


if __name__ == "__main__":
    trainer = PPOTrainer(n_iterations=500, n_envs=8, rollout_steps=256, n_epochs=4, batch_size=256)
    trainer.run()
