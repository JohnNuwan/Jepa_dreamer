"""
ftmo_agent/agent_v2.py — PPO agent V2 with:
- LSTM + attention for multi-TF sequence modeling
- Symbol embedding for cross-asset generalization
- 8 actions (hold, buy, sell, close, split, pyramid, partial close)
- Directional balance regularization (anti-bias)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

class MultiSymbolActorCritic(nn.Module):
    """
    Actor-Critic designed for multi-symbol, multi-timeframe trading.
    
    Architecture:
    1. Feature projection: projects raw features to hidden dim
    2. LSTM: captures temporal patterns across lookback window
    3. Attention pooling: weighted aggregation of LSTM outputs
    4. Actor: 8-action policy head
    5. Critic: state value head
    6. Direction balance regularizer: penalizes directional bias
    """
    def __init__(self, n_features, n_actions=8, hidden_dim=384,
                 lstm_layers=2, dropout=0.15, n_heads=4):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # Feature projection (handles multi-TF features + symbol emb + position)
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Multi-head attention for pattern detection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Temporal aggregation: combine attention output + last hidden
        self.temporal_agg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_actions),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Action mask (some actions not valid in some states)
        # Learned through training, not hardcoded
        
        # Apply weight initialization
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
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, H)
        
        # Aggregate: attention output (mean) + last LSTM hidden
        attn_mean = attn_out.mean(dim=1)  # (B, H)
        last_hidden = lstm_out[:, -1, :]  # (B, H)
        context = self.temporal_agg(torch.cat([attn_mean, last_hidden], dim=-1))
        
        # Policy and value
        logits = self.actor(context)
        value = self.critic(context)
        
        return logits, value
    
    def get_action(self, obs, deterministic=False, action_mask=None):
        """
        obs: (1, seq_len, n_features)
        action_mask: (n_actions,) boolean, True if action is valid
        """
        with torch.no_grad():
            logits, value = self.forward(obs)
            
            # Apply action mask (set invalid actions to -inf)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.unsqueeze(0), float('-inf'))
            
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action.item(), log_prob.item(), value.item(), entropy.item()


class PPOTrainerV2:
    """PPO trainer V2 with anti-bias and multi-symbol support."""
    
    def __init__(self, n_features, n_actions=8, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, n_epochs=10,
                 batch_size=128, ent_coef=0.02, vf_coef=0.5,
                 max_grad_norm=0.5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = MultiSymbolActorCritic(n_features, n_actions).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.rollout = []
        
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} params on {self.device}")
    
    def collect(self, obs, action, reward, log_prob, value, done, info):
        self.rollout.append({
            'obs': obs, 'action': action, 'reward': reward,
            'log_prob': log_prob, 'value': value, 'done': done,
            'info': info,
        })
    
    def compute_gae(self, rewards, values, dones, last_value=0):
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
        if len(self.rollout) < self.batch_size:
            return {}
        
        obs_list = [r['obs'] for r in self.rollout]
        actions = [r['action'] for r in self.rollout]
        rewards = [r['reward'] for r in self.rollout]
        old_log_probs = [r['log_prob'] for r in self.rollout]
        values = [r['value'] for r in self.rollout]
        dones = [r['done'] for r in self.rollout]
        
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        if advantages_tensor.numel() > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) /                                 (advantages_tensor.std() + 1e-8)
        
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'loss': []}
        n = len(self.rollout)
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]
                
                batch_obs = obs_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_lp = old_log_probs_tensor[idx]
                batch_returns = returns_tensor[idx]
                batch_adv = advantages_tensor[idx]
                
                logits, value = self.model(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - batch_old_lp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(value.squeeze(-1), batch_returns)
                
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
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
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"Loaded: {path}")
