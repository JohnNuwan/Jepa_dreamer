"""
ftmo_agent/agent.py — PPO agent with LSTM for temporal patterns.
Designed for XAUUSD M15 trading with FTMO rules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

class TradingActorCritic(nn.Module):
    """
    Actor-Critic with LSTM for sequence modeling.
    - LSTM processes the lookback window
    - Actor: policy head (action probabilities)
    - Critic: value head (state value estimate)
    """
    def __init__(self, n_features, n_actions=4, hidden_dim=256, 
                 lstm_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Attention pooling (weighted average of LSTM outputs)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 (helps LSTM learn long-term)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
    
    def forward(self, x):
        """x: (batch, seq_len, n_features)"""
        B, T, F = x.shape
        
        # Project features
        x = self.feature_proj(x)  # (B, T, H)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, H)
        
        # Also use last hidden state
        last = lstm_out[:, -1, :]  # (B, H)
        combined = context + last  # residual
        
        # Actor and Critic
        logits = self.actor(combined)
        value = self.critic(combined)
        
        return logits, value
    
    def get_action(self, obs, deterministic=False):
        """obs: (1, seq_len, n_features)"""
        with torch.no_grad():
            logits, value = self.forward(obs)
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action.item(), log_prob.item(), value.item(), entropy.item()


class PPOTrainer:
    """PPO trainer for the trading agent."""
    
    def __init__(self, n_features, n_actions=4, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, n_epochs=10,
                 batch_size=64, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = TradingActorCritic(n_features, n_actions).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, 
                                            weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.rollout = []
    
    def collect(self, obs, action, reward, log_prob, value, done, info):
        """Collect a transition."""
        self.rollout.append({
            'obs': obs, 'action': action, 'reward': reward,
            'log_prob': log_prob, 'value': value, 'done': done,
        })
    
    def compute_gae(self, rewards, values, dones, last_value=0):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = list(values) + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return advantages, returns
    
    def update(self, last_value=0):
        """Update model with PPO."""
        if len(self.rollout) < self.batch_size:
            return {}
        
        # Extract data
        obs_list = [r['obs'] for r in self.rollout]
        actions = [r['action'] for r in self.rollout]
        rewards = [r['reward'] for r in self.rollout]
        old_log_probs = [r['log_prob'] for r in self.rollout]
        values = [r['value'] for r in self.rollout]
        dones = [r['done'] for r in self.rollout]
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) /                             (advantages_tensor.std() + 1e-8)
        
        # PPO updates
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'loss': []}
        n = len(self.rollout)
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                if end > n:
                    end = n
                idx = indices[start:end]
                
                batch_obs = obs_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_lp = old_log_probs_tensor[idx]
                batch_returns = returns_tensor[idx]
                batch_adv = advantages_tensor[idx]
                
                # Forward pass
                logits, value = self.model(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = (new_log_probs - batch_old_lp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_loss = F.mse_loss(value.squeeze(-1), batch_returns)
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.item())
                metrics['loss'].append(loss.item())
        
        self.scheduler.step()
        self.rollout = []
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        print(f"Model loaded from {path}")
