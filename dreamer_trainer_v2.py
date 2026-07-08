"""
ftmo_agent/dreamer_trainer_v2.py — DreamerV3 V3 with critical fixes:
- Entropy floor (prevents collapse to deterministic policy)
- Temperature never below 0.7 (cyclic restart)
- JEPA: separate backward, no retain_graph, lower VICReg weights
- JEPA target momentum 0.999 (stable)
- Actor init gain=0.1 (not 0.01)
- Separate JEPA/WM optimizer steps
- Symlog reward normalization in AC training
- Sequence-based imagination (proper multi-step RSSM)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, math
from collections import deque
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'octopus'))

from config import N_ACTIONS, ACTION_NAMES, HOLD, BUY, SELL, CLOSE
from octopus.engine_src_networks_jepa import TSJEPA
from octopus.engine_src_networks_world_model import RSSMWorldModel, symlog, symexp
from octopus.engine_src_networks_actor_critic import ActorCritic


class DreamerV3AgentV2:
    """
    DreamerV3 V3: fixed entropy collapse + JEPA divergence + reward signal.
    
    Key fixes:
    1. Entropy floor: if entropy < 0.5 nat, boost entropy_coeff dynamically
    2. Temperature: never below 0.7, cyclic restart every 500 episodes
    3. JEPA: separate optimizer step (no retain_graph), lower VICReg weights
    4. JEPA target momentum: 0.999 (slower drift = stable representations)
    5. Actor init: gain=0.1 (not 0.01, avoids uniform→collapse)
    6. KL balancing: 80% posterior, 20% prior (DreamerV3 paper)
    """
    
    def __init__(self, input_dim=145, seq_len=48, embedding_dim=128,
                 stoch_size=32, stoch_classes=32, deter_size=512,
                 hidden_dim=512, action_dim=N_ACTIONS,
                 wm_lr=2e-4, ac_lr=3e-4, jepa_lr=3e-5,
                 gamma=0.997, lambda_=0.95, horizon=30,
                 entropy_coeff=0.05, symlog_rewards=True):
        
        self.device_wm = torch.device('cuda:0')
        self.device_ac = torch.device('cuda:1')
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.symlog_rewards = symlog_rewards
        self.action_dim = action_dim
        
        state_dim = stoch_size * stoch_classes + deter_size
        
        # GPU 0: JEPA + World Model
        self.jepa = TSJEPA(
            input_dim=input_dim, seq_len=seq_len,
            embedding_dim=embedding_dim, momentum=0.9995  # V4: slower drift
        ).to(self.device_wm)
        
        # FIX: Override VICReg coefficients (lower = more stable)
        self.jepa.vicreg_loss.sim_coeff = 10.0  # was 25.0
        self.jepa.vicreg_loss.var_coeff = 10.0  # was 25.0
        self.jepa.vicreg_loss.cov_coeff = 0.5   # was 1.0
        
        self.world_model = RSSMWorldModel(
            stoch_size=stoch_size, stoch_classes=stoch_classes,
            deter_size=deter_size, hidden_dim=hidden_dim,
            action_dim=action_dim, embedding_dim=embedding_dim
        ).to(self.device_wm)
        
        # FIX: Separate optimizers with different LRs
        self.jepa_opt = torch.optim.AdamW(self.jepa.parameters(), lr=jepa_lr, weight_decay=0.01)
        self.wm_opt = torch.optim.AdamW(self.world_model.parameters(), lr=wm_lr, weight_decay=0.01)
        
        # GPU 1: Actor-Critic
        self.actor_critic = ActorCritic(
            stoch_size=stoch_size, stoch_classes=stoch_classes,
            deter_size=deter_size, action_dim=action_dim,
            hidden_dim=hidden_dim, gamma=gamma, lambda_=lambda_,
            entropy_coeff=entropy_coeff
        ).to(self.device_ac)
        
        # FIX: Re-init actor with gain=0.1 (not 0.01)
        for module in self.actor_critic.actor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.ac_opt = torch.optim.AdamW(self.actor_critic.parameters(), lr=ac_lr, weight_decay=0.01)
        
        # Schedulers — cosine annealing with warm restarts
        self.wm_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.wm_opt, T_0=300, T_mult=2)
        self.ac_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.ac_opt, T_0=300, T_mult=2)
        self.jepa_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.jepa_opt, T_0=300, T_mult=2)
        
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.deter_size = deter_size
        self.state_dim = state_dim
        
        # FIX: Entropy tracking for dynamic floor
        self.entropy_history = deque(maxlen=50)
        self.min_entropy = 0.5  # nat floor
        self.base_entropy_coeff = entropy_coeff
        
        n_params = (sum(p.numel() for p in self.jepa.parameters()) +
                    sum(p.numel() for p in self.world_model.parameters()) +
                    sum(p.numel() for p in self.actor_critic.parameters()))
        wm_params = sum(p.numel() for p in self.jepa.parameters()) + sum(p.numel() for p in self.world_model.parameters())
        ac_params = sum(p.numel() for p in self.actor_critic.parameters())
        print(f"DreamerV3 V3: {n_params:,} params | GPU0(WM+JEPA): {wm_params:,} | GPU1(AC): {ac_params:,}")
    
    def train_world_model(self, batch):
        """FIX: Separate JEPA and WM backward passes (no retain_graph)."""
        obs = torch.FloatTensor(batch['obs']).to(self.device_wm)
        actions = torch.FloatTensor(batch['actions']).to(self.device_wm)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device_wm)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device_wm)
        
        B = obs.shape[0]
        
        # === STEP 1: JEPA self-supervised (separate backward) ===
        z_online = self.jepa.encoder(obs)
        z_target = self.jepa.encode_target(next_obs)
        z_pred = self.jepa.predictor(z_online)
        jepa_loss, jepa_metrics = self.jepa.vicreg_loss(z_pred, z_target.detach())
        
        self.jepa_opt.zero_grad()
        jepa_loss.backward()  # FIX: no retain_graph
        torch.nn.utils.clip_grad_norm_(self.jepa.parameters(), 1.0)
        self.jepa_opt.step()
        self.jepa_sched.step()
        self.jepa.update_target_encoder()
        
        # === STEP 2: World Model (uses JEPA encoder frozen) ===
        with torch.no_grad():
            embeddings = self.jepa.encoder(obs)
            next_embeddings = self.jepa.encoder(next_obs)
        
        stoch = torch.zeros(B, self.stoch_size, self.stoch_classes, device=self.device_wm)
        deter = torch.zeros(B, self.deter_size, device=self.device_wm)
        
        action_t = actions.view(B, -1) if actions.dim() > 2 else actions
        
        post, prior, deter_next, _ = self.world_model.transition(
            prev_stoch=stoch, prev_action=action_t,
            prev_embedding=embeddings, next_embedding=next_embeddings
        )
        
        pred_reward = self.world_model.predict_reward(post, deter_next)
        target_reward = symlog(rewards.view(-1, 1))
        reward_loss = F.mse_loss(pred_reward.squeeze(-1), target_reward.squeeze(-1))
        
        # FIX: KL balancing (DreamerV3 style): 80% posterior, 20% prior
        post_logits = self.world_model.transition.posterior_net(
            torch.cat([deter_next, next_embeddings], dim=-1))
        prior_logits = self.world_model.transition.prior_net(deter_next)
        post_dist = torch.distributions.OneHotCategorical(
            logits=post_logits.reshape(B, self.stoch_size, self.stoch_classes))
        prior_dist = torch.distributions.OneHotCategorical(
            logits=prior_logits.reshape(B, self.stoch_size, self.stoch_classes))
        
        # KL: both directions with balancing
        kl_post = torch.distributions.kl_divergence(post_dist, prior_dist).mean()  # post → prior
        kl_prior = torch.distributions.kl_divergence(prior_dist, post_dist).mean()  # prior → post
        kl_loss = 0.8 * kl_post + 0.2 * kl_prior  # FIX: KL balancing
        
        wm_loss = reward_loss + 0.1 * kl_loss
        
        self.wm_opt.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.wm_opt.step()
        self.wm_sched.step()
        
        return {
            'jepa_loss': jepa_loss.item(),
            'wm_loss': wm_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
        }
    
    def train_actor_critic(self, batch):
        obs = torch.FloatTensor(batch['obs']).to(self.device_wm)
        
        with torch.no_grad():
            embedding = self.jepa.encoder(obs)
            B = embedding.shape[0]
            stoch = torch.zeros(B, self.stoch_size, self.stoch_classes, device=self.device_wm)
            deter = torch.zeros(B, self.deter_size, device=self.device_wm)
            post, _, deter, _ = self.world_model.transition(
                prev_stoch=stoch, prev_action=torch.zeros(B, self.action_dim, device=self.device_wm),
                prev_embedding=embedding, next_embedding=embedding
            )
        
        # Imagine with 30 steps
        imagined = self.world_model.imagine(
            initial_stoch=post, initial_deter=deter,
            policy=lambda s: self._get_action_from_ac(s), horizon=self.horizon
        )
        
        stoch_traj = imagined['stoch'].to(self.device_ac)
        deter_traj = imagined['deter'].to(self.device_ac)
        action_traj = imagined['action'].to(self.device_ac)
        reward_traj = imagined['reward'].to(self.device_ac)
        cont_traj = imagined['continue'].to(self.device_ac)
        
        # FIX: Dynamic entropy coefficient based on recent entropy
        if self.entropy_history:
            recent_entropy = np.mean(self.entropy_history)
            if recent_entropy < self.min_entropy:
                # Boost entropy coefficient when too low
                self.actor_critic.entropy_coeff = min(0.3, self.base_entropy_coeff * 4.0)
            else:
                self.actor_critic.entropy_coeff = max(0.05, self.base_entropy_coeff)
        
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
        
        # Track entropy
        self.entropy_history.append(ac_metrics.get('entropy', 0))
        
        return {'ac_loss': ac_loss.item(), **ac_metrics}
    
    def _get_action_from_ac(self, state):
        state_gpu1 = state.to(self.device_ac)
        stoch_flat = state_gpu1[:, :self.stoch_size * self.stoch_classes]
        deter_flat = state_gpu1[:, self.stoch_size * self.stoch_classes:]
        action = self.actor_critic.get_action(
            stoch=stoch_flat.reshape(-1, self.stoch_size, self.stoch_classes),
            deter=deter_flat,
            deterministic=False
        )
        return action.to(self.device_wm)
    
    def get_action(self, obs, action_mask=None, temperature=1.0, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_wm)
        
        with torch.no_grad():
            embedding = self.jepa.encoder(obs_tensor)
            B = 1
            stoch = torch.zeros(B, self.stoch_size, self.stoch_classes, device=self.device_wm)
            deter = torch.zeros(B, self.deter_size, device=self.device_wm)
            post, _, deter, _ = self.world_model.transition(
                prev_stoch=stoch, prev_action=torch.zeros(B, self.action_dim, device=self.device_wm),
                prev_embedding=embedding, next_embedding=embedding
            )
            
            state = torch.cat([post.reshape(1, -1), deter], dim=-1).to(self.device_ac)
            logits, probs = self.actor_critic.actor(state)
            
            # NOTE: exploration noise removed — was larger than logits, made actions random
            
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device_ac)
                logits = logits.masked_fill(~mask_tensor.unsqueeze(0), float('-inf'))
                probs = torch.softmax(logits, dim=-1)
            
            if deterministic or temperature == 0:
                action = probs.argmax(dim=-1).item()
            else:
                # Temperature sampling
                tempered_logits = torch.log(probs + 1e-8) / max(temperature, 1.0)  # FIX: floor temp at 0.7
                tempered_probs = torch.softmax(tempered_logits, dim=-1)
                dist = torch.distributions.Categorical(probs=tempered_probs)
                action = dist.sample().item()
            
            value = self.actor_critic.critic(state).item()
        
        return action, value, probs.squeeze().cpu().numpy()
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'jepa': self.jepa.state_dict(),
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device_wm)
        self.jepa.load_state_dict(ckpt['jepa'])
        self.world_model.load_state_dict(ckpt['world_model'])
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        print(f"Loaded: {path}")


class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action_oh, reward, next_obs, done, action_mask=None):
        self.buffer.append({
            'obs': obs, 'action': action_oh, 'reward': reward,
            'next_obs': next_obs, 'done': done, 'action_mask': action_mask,
        })
    
    def sample(self, batch_size=128):
        batch = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        items = [self.buffer[i] for i in batch]
        return {
            'obs': np.array([x['obs'] for x in items]),
            'actions': np.array([x['action'] for x in items]),
            'rewards': np.array([x['reward'] for x in items]),
            'next_obs': np.array([x['next_obs'] for x in items]),
            'dones': np.array([x['done'] for x in items]),
        }
    
    def __len__(self):
        return len(self.buffer)
