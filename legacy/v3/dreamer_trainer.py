"""
ftmo_agent/dreamer_trainer.py — DreamerV3 Trainer V3
Combines JEPA encoder + RSSM World Model + Actor-Critic.
Uses BOTH GPUs: World Model on GPU0, Actor-Critic on GPU1.

Architecture:
1. JEPA encoder: self-supervised market representation (GPU0)
2. RSSM World Model: learns market dynamics in latent space (GPU0)
3. Actor-Critic: trained on imagined trajectories (GPU1)
4. MCTS: planning at inference time (optional, GPU0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os, sys, json, random, math
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, RISK_CONFIG,
                     ANTI_BIAS_CONFIG, N_ACTIONS, ACTION_NAMES,
                     HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL,
                     PYRAMID, PARTIAL_CLOSE)
from features_v2 import compute_multi_tf_features, get_feature_columns, get_symbol_embedding

# Import Octopus networks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'octopus'))
from octopus.engine_src_networks_jepa import TSJEPA, JEPAEncoder, VICRegLoss
from octopus.engine_src_networks_world_model import RSSMWorldModel, RSSMTransition, symlog, symexp
from octopus.engine_src_networks_actor_critic import ActorCritic, ActorNetwork, CriticNetwork

class DreamerV3Agent:
    """
    Full DreamerV3 agent for multi-symbol trading.
    
    Dual-GPU layout:
    - GPU 0: JEPA encoder + RSSM World Model (heavy compute)
    - GPU 1: Actor-Critic (fast, low memory)
    """
    def __init__(self, input_dim=132, seq_len=48, embedding_dim=128,
                 stoch_size=32, stoch_classes=32, deter_size=512,
                 hidden_dim=512, action_dim=N_ACTIONS,
                 wm_lr=3e-4, ac_lr=3e-4, jepa_lr=1e-4,
                 gamma=0.997, lambda_=0.95, horizon=15,
                 entropy_coeff=0.003, symlog_rewards=True):
        
        self.device_wm = torch.device('cuda:0')   # World Model on GPU 0
        self.device_ac = torch.device('cuda:1')   # Actor-Critic on GPU 1
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.symlog_rewards = symlog_rewards
        
        state_dim = stoch_size * stoch_classes + deter_size
        
        # === GPU 0: JEPA Encoder + World Model ===
        self.jepa = TSJEPA(
            input_dim=input_dim, seq_len=seq_len,
            embedding_dim=embedding_dim, momentum=0.99
        ).to(self.device_wm)
        
        self.world_model = RSSMWorldModel(
            stoch_size=stoch_size, stoch_classes=stoch_classes,
            deter_size=deter_size, hidden_dim=hidden_dim,
            action_dim=action_dim, embedding_dim=embedding_dim
        ).to(self.device_wm)
        
        # Optimizers for WM
        self.jepa_opt = torch.optim.AdamW(self.jepa.parameters(), lr=jepa_lr, weight_decay=0.01)
        self.wm_opt = torch.optim.AdamW(self.world_model.parameters(), lr=wm_lr, weight_decay=0.01)
        
        # === GPU 1: Actor-Critic ===
        self.actor_critic = ActorCritic(
            stoch_size=stoch_size, stoch_classes=stoch_classes,
            deter_size=deter_size, action_dim=action_dim,
            hidden_dim=hidden_dim, gamma=gamma, lambda_=lambda_,
            entropy_coeff=entropy_coeff
        ).to(self.device_ac)
        
        self.ac_opt = torch.optim.AdamW(self.actor_critic.parameters(), lr=ac_lr, weight_decay=0.01)
        
        # Schedulers
        self.wm_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.wm_opt, T_0=100, T_mult=2)
        self.ac_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.ac_opt, T_0=100, T_mult=2)
        
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.deter_size = deter_size
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        n_params = (sum(p.numel() for p in self.jepa.parameters()) +
                    sum(p.numel() for p in self.world_model.parameters()) +
                    sum(p.numel() for p in self.actor_critic.parameters()))
        print(f"DreamerV3 Agent: {n_params:,} params")
        print(f"  GPU0 (WM+JEPA): {sum(p.numel() for p in self.jepa.parameters()) + sum(p.numel() for p in self.world_model.parameters()):,}")
        print(f"  GPU1 (AC):      {sum(p.numel() for p in self.actor_critic.parameters()):,}")
    
    def encode_observation(self, obs):
        """Encode observation sequence with JEPA. obs: (B, seq, dim) → (B, emb)"""
        with torch.no_grad():
            return self.jepa.encoder(obs.to(self.device_wm))
    
    def train_world_model(self, batch):
        """
        Train World Model on real experience.
        batch: dict with 'obs', 'actions', 'rewards', 'next_obs', 'done'
        """
        obs = torch.FloatTensor(batch['obs']).to(self.device_wm)
        actions = torch.FloatTensor(batch['actions']).to(self.device_wm)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device_wm)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device_wm)
        dones = torch.FloatTensor(batch['dones']).to(self.device_wm)
        
        B = obs.shape[0]
        
        # 1. JEPA self-supervised loss
        z_online = self.jepa.encoder(obs)
        z_target = self.jepa.encode_target(next_obs)
        z_pred = self.jepa.predictor(z_online)
        jepa_loss, jepa_metrics = self.jepa.vicreg_loss(z_pred, z_target.detach())
        
        self.jepa_opt.zero_grad()
        jepa_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.jepa.parameters(), 1.0)
        self.jepa_opt.step()
        self.jepa.update_target_encoder()
        
        # 2. World Model training
        # Encode observations (no grad — JEPA already trained)
        with torch.no_grad():
            embeddings = self.jepa.encoder(obs)
            next_embeddings = self.jepa.encoder(next_obs)
        
        # Initial state
        stoch = torch.zeros(B, self.stoch_size, self.stoch_classes, device=self.device_wm)
        deter = torch.zeros(B, self.deter_size, device=self.device_wm)
        
        wm_losses = []
        reward_losses = []
        
        # Roll through sequence
        for t in range(obs.shape[1] if obs.dim() == 3 else 1):
            emb_t = embeddings if embeddings.dim() == 2 else embeddings[:, t]
            next_emb_t = next_embeddings if next_embeddings.dim() == 2 else next_embeddings[:, t]
            action_t = actions.view(-1, N_ACTIONS) if actions.dim() <= 2 else actions[:, t]
            
            # RSSM transition
            post, prior, deter_next, _ = self.world_model.transition(
                prev_stoch=stoch, prev_action=action_t,
                prev_embedding=emb_t, next_embedding=next_emb_t
            )
            
            # Reward prediction
            pred_reward = self.world_model.predict_reward(post, deter_next)
            
            # Symlog transform rewards for stability
            if self.symlog_rewards:
                target_reward = symlog(rewards.view(-1, 1))
            else:
                target_reward = rewards.view(-1, 1)
            
            reward_loss = F.mse_loss(pred_reward.squeeze(-1), target_reward.squeeze(-1))
            wm_losses.append(reward_loss)
            reward_losses.append(reward_loss.item())
            
            # KL divergence between prior and posterior (KL balancing)
            post_logits = self.world_model.transition.posterior_net(
                torch.cat([deter_next, next_emb_t], dim=-1))
            prior_logits = self.world_model.transition.prior_net(deter_next)
            post_dist = torch.distributions.OneHotCategorical(
                logits=post_logits.reshape(B, self.stoch_size, self.stoch_classes))
            prior_dist = torch.distributions.OneHotCategorical(
                logits=prior_logits.reshape(B, self.stoch_size, self.stoch_classes))
            kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
            wm_losses.append(0.1 * kl_loss)
            
            # Update state
            stoch = post.detach()
            deter = deter_next.detach()
        
        wm_loss = sum(wm_losses) / len(wm_losses)
        
        self.wm_opt.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.wm_opt.step()
        self.wm_sched.step()
        
        return {
            'jepa_loss': jepa_loss.item(),
            **jepa_metrics,
            'wm_loss': wm_loss.item(),
            'reward_loss': np.mean(reward_losses),
            'kl_loss': kl_loss.item(),
        }
    
    def train_actor_critic(self, batch):
        """
        Train Actor-Critic on imagined trajectories.
        1. Encode real observation with JEPA
        2. Roll out World Model in latent space (imagine)
        3. Train AC on the imagined trajectory
        """
        obs = torch.FloatTensor(batch['obs']).to(self.device_wm)
        
        # 1. Get initial latent state from JEPA + RSSM
        with torch.no_grad():
            embedding = self.jepa.encoder(obs)
            B = embedding.shape[0]
            stoch = torch.zeros(B, self.stoch_size, self.stoch_classes, device=self.device_wm)
            deter = torch.zeros(B, self.deter_size, device=self.device_wm)
            # One step of RSSM to get initial posterior
            post, _, deter, _ = self.world_model.transition(
                prev_stoch=stoch, prev_action=torch.zeros(B, N_ACTIONS, device=self.device_wm),
                prev_embedding=embedding, next_embedding=embedding
            )
        
        # 2. Imagine trajectory on GPU 0
        imagined = self.world_model.imagine(
            initial_stoch=post, initial_deter=deter,
            policy=lambda s: self._get_action_from_ac(s),
            horizon=self.horizon
        )
        
        # 3. Move to GPU 1 for AC training
        stoch_traj = imagined['stoch'].to(self.device_ac)
        deter_traj = imagined['deter'].to(self.device_ac)
        action_traj = imagined['action'].to(self.device_ac)
        reward_traj = imagined['reward'].to(self.device_ac)
        cont_traj = imagined['continue'].to(self.device_ac)
        
        # 4. Train AC on imagined data
        ac_loss, ac_metrics = self.actor_critic.compute_loss(
            stoch=stoch_traj, deter=deter_traj,
            actions=action_traj, rewards=reward_traj,
            continues=cont_traj
        )
        
        self.ac_opt.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.ac_opt.step()
        self.ac_sched.step()
        
        return {
            'ac_loss': ac_loss.item(),
            **ac_metrics,
        }
    
    def _get_action_from_ac(self, state):
        """Get action from AC (on GPU1). State is on GPU0 — transfer."""
        state_gpu1 = state.to(self.device_ac)
        action = self.actor_critic.get_action(
            stoch=state_gpu1[:, :self.stoch_size * self.stoch_classes].reshape(
                -1, self.stoch_size, self.stoch_classes),
            deter=state_gpu1[:, self.stoch_size * self.stoch_classes:],
            deterministic=False
        )
        return action.to(self.device_wm)  # back to GPU0 for World Model
    
    def get_action(self, obs, deterministic=False):
        """Inference: get trading action from observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_wm)
        
        with torch.no_grad():
            # Encode with JEPA
            embedding = self.jepa.encoder(obs_tensor)
            
            # Get latent state
            B = 1
            stoch = torch.zeros(B, self.stoch_size, self.stoch_classes, device=self.device_wm)
            deter = torch.zeros(B, self.deter_size, device=self.device_wm)
            post, _, deter, _ = self.world_model.transition(
                prev_stoch=stoch, prev_action=torch.zeros(B, N_ACTIONS, device=self.device_wm),
                prev_embedding=embedding, next_embedding=embedding
            )
            
            # Get action from AC (transfer to GPU1)
            state = torch.cat([post.reshape(1, -1), deter], dim=-1).to(self.device_ac)
            action_oh = self.actor_critic.get_action(
                stoch=post.to(self.device_ac),
                deter=deter.to(self.device_ac),
                deterministic=deterministic
            )
            action = action_oh.argmax(dim=-1).item()
            
            # Also get value estimate
            value = self.actor_critic.critic(state).item()
        
        return action, value
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'jepa': self.jepa.state_dict(),
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'wm_opt': self.wm_opt.state_dict(),
            'ac_opt': self.ac_opt.state_dict(),
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device_wm)
        self.jepa.load_state_dict(ckpt['jepa'])
        self.world_model.load_state_dict(ckpt['world_model'])
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        print(f"Loaded: {path}")


class ReplayBuffer:
    """Replay buffer for DreamerV3 training."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append({
            'obs': obs, 'action': action, 'reward': reward,
            'next_obs': next_obs, 'done': done,
        })
    
    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs = np.array([x['obs'] for x in batch])
        actions = np.array([x['action'] for x in batch])
        rewards = np.array([x['reward'] for x in batch])
        next_obs = np.array([x['next_obs'] for x in batch])
        dones = np.array([x['done'] for x in batch])
        return {'obs': obs, 'actions': actions, 'rewards': rewards,
                'next_obs': next_obs, 'dones': dones}
    
    def __len__(self):
        return len(self.buffer)
