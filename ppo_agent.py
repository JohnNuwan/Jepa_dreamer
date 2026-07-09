"""
ppo_agent.py — Agent PPO avec encodeur LSTM pour trading FTMO.
Architecture moderne : LSTM sur séquence temporelle → Actor + Critic.
GAE, clipping PPO, 8 environnements parallèles.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, List

class ObsEncoder(nn.Module):
    """Encodeur LSTM : (B, 48, 296) → (B, 256)"""
    def __init__(self, input_dim=296, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, hidden=None):
        # x: (B, T, D) where T=48, D=296
        out, (h, c) = self.lstm(x, hidden)
        # Take last timestep output
        last = out[:, -1, :]  # (B, hidden_dim)
        return F.gelu(self.proj(last)), (h, c)


class Actor(nn.Module):
    """Tête actor : hidden → 8 actions"""
    def __init__(self, hidden_dim=256, action_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)  # logits
    
    def get_action(self, x, action_mask=None, deterministic=False):
        logits = self.forward(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze(-1)
        return action, log_prob, probs


class Critic(nn.Module):
    """Tête critic : hidden → 1 valeur"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    """Agent PPO complet : LSTM + Actor + Critic + GAE"""
    
    def __init__(self, input_dim=296, hidden_dim=256, action_dim=8,
                 lr=3e-4, gamma=0.997, lambda_=0.95, clip_eps=0.2,
                 entropy_coeff=0.01, value_coeff=0.5, device='cuda:0'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.action_dim = action_dim
        
        self.encoder = ObsEncoder(input_dim, hidden_dim).to(self.device)
        self.actor = Actor(hidden_dim, action_dim).to(self.device)
        self.critic = Critic(hidden_dim).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()),
            lr=lr, weight_decay=0.01
        )
        
        n_params = sum(p.numel() for p in self.encoder.parameters()) + \
                   sum(p.numel() for p in self.actor.parameters()) + \
                   sum(p.numel() for p in self.critic.parameters())
        print(f"PPO Agent: {n_params:,} params | LSTM 2×{hidden_dim} | GPU: {device}")
    
    def get_action_batch(self, obs_batch, masks_batch=None, deterministic=False):
        """obs_batch: (B, T, D), masks_batch: (B, action_dim) bool"""
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        latent, _ = self.encoder(obs_t)
        if masks_batch is not None:
            masks_t = torch.BoolTensor(masks_batch).to(self.device)
        else:
            masks_t = None
        actions, log_probs, probs = self.actor.get_action(latent, masks_t, deterministic)
        values = self.critic(latent)
        return (actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(),
                values.detach().cpu().numpy(), probs.detach().cpu().numpy())
    
    def get_value(self, obs_batch):
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        latent, _ = self.encoder(obs_t)
        return self.critic(latent).detach().cpu().numpy()
    
    def compute_gae(self, rewards, values, dones, last_value=0.0):
        """Calcule les GAE λ-returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        return advantages, returns
    
    def update(self, obs_batch, actions_batch, old_log_probs, returns, advantages, action_masks=None):
        """Mise à jour PPO avec clipping."""
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        actions_t = torch.LongTensor(actions_batch).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        latent, _ = self.encoder(obs_t)
        
        # Actor loss (clipped PPO)
        logits = self.actor(latent)
        if action_masks is not None:
            masks_t = torch.BoolTensor(action_masks).to(self.device)
            logits = logits.masked_fill(~masks_t, float('-inf'))
        
        probs = F.softmax(logits, dim=-1)
        new_log_probs = torch.log(probs.gather(1, actions_t.unsqueeze(1)) + 1e-8).squeeze(-1)
        
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        
        # Critic loss (clipped)
        values = self.critic(latent)
        value_loss = F.mse_loss(values, returns_t)
        
        # Total loss
        total_loss = actor_loss - self.entropy_coeff * entropy + self.value_coeff * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 1.0)
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
        }
    
    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.encoder.to(self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.to(self.device)
        self.critic.load_state_dict(ckpt['critic'])
        self.critic.to(self.device)
