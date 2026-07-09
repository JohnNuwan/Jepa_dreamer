"""
es_agent.py — Evolution Strategies pour trading FTMO.
Population d'agents LSTM évalués en parallèle, sélection naturelle sur le PnL.
Pas de critic, pas de value function, pas de reward shaping — juste le PnL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from config import FTMO_CONFIG


class ESPolicy(nn.Module):
    """Politique LSTM : (B, 48, 296) → 8 action logits."""
    def __init__(self, input_dim=296, hidden_dim=128, action_dim=8, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x, hidden=None):
        out, h = self.lstm(x, hidden)
        logits = self.head(out[:, -1, :])
        return logits, h
    
    def get_probs(self, x, action_mask=None):
        logits, _ = self.forward(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        return F.softmax(logits, dim=-1)
    
    def sample(self, x, action_mask=None, deterministic=False):
        probs = self.get_probs(x, action_mask)
        if deterministic:
            return probs.argmax(dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class ESAgent:
    """Evolution Strategies: population-based optimization of PnL."""
    
    def __init__(self, input_dim=296, hidden_dim=128, action_dim=8,
                 pop_size=16, sigma=0.02, lr=0.01, elite_frac=0.25, device='cuda:0'):
        self.device = torch.device(device)
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.elite_frac = elite_frac
        self.action_dim = action_dim
        self.generation = 0
        
        # Master policy
        self.master = ESPolicy(input_dim, hidden_dim, action_dim).to(self.device)
        n_params = self.master.count_params()
        print(f"ES Agent: {n_params:,} params | pop={pop_size} | σ={sigma} | GPU: {device}")
        
        # Population: list of (perturbed copy, noise_vector)
        self.population = []
        self._create_population()
    
    def _create_population(self):
        """Crée la population en perturbant le master."""
        self.population = []
        master_vec = self._get_params_flat(self.master)
        for _ in range(self.pop_size):
            noise = torch.randn_like(master_vec) * self.sigma
            perturbed = ESPolicy(
                self.master.lstm.input_size,
                self.master.lstm.hidden_size,
                self.action_dim,
                self.master.lstm.num_layers
            ).to(self.device)
            self._set_params_flat(perturbed, master_vec + noise)
            self.population.append((perturbed, noise))
    
    def _get_params_flat(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    
    def _set_params_flat(self, model, flat):
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].view_as(p))
            offset += n
    
    def evaluate_single(self, policy, env, steps=1000) -> float:
        """Évalue une politique, retourne le REWARD TOTAL (pas le PnL pur).
        Le reward dense inclut time-decay, opening bonus, holding bonus."""
        obs = env.reset()
        total_reward = 0.0
        lstm_hidden = None
        
        for _ in range(steps):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                logits, lstm_hidden = policy(obs_t, lstm_hidden)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
                logits_masked = logits.masked_fill(~mask_t, float('-inf'))
                probs = F.softmax(logits_masked, dim=-1)
                if np.random.random() < 0.3:
                    valid = np.where(mask)[0]
                    action = np.random.choice(valid) if len(valid) > 0 else 0
                else:
                    action = probs.argmax(dim=-1).item()
            
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        
        return float(total_reward)
    
    def evaluate_population(self, envs: List, steps=500) -> List[float]:
        """Évalue toute la population sur des environnements parallèles."""
        assert len(envs) == self.pop_size, f"Need {self.pop_size} envs, got {len(envs)}"
        fitness = []
        for (policy, _), env in zip(self.population, envs):
            fit = self.evaluate_single(policy, env, steps)
            fitness.append(fit)
        return fitness
    
    def evolve(self, fitness: List[float]):
        """Met à jour le master via sélection naturelle."""
        # Rank by fitness (higher is better)
        ranked = sorted(enumerate(zip(fitness, self.population)), 
                       key=lambda x: x[1][0], reverse=True)
        
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elite = ranked[:n_elite]
        
        # Weighted average of elite noises
        elite_fitness = [f for f, _ in [x[1] for x in elite]]
        total_fit = sum(max(0, f) for f in elite_fitness) + 1e-8
        weights = [max(0, f) / total_fit for f in elite_fitness]
        
        # Compute gradient estimate
        master_vec = self._get_params_flat(self.master)
        grad = torch.zeros_like(master_vec)
        for (idx, (fit, (_, noise))), w in zip(elite, weights):
            grad += w * noise
        
        # Update master
        master_vec += self.lr * grad
        self._set_params_flat(self.master, master_vec)
        
        # Create new population
        self._create_population()
        self.generation += 1
        
        return {
            'best_fitness': elite_fitness[0] if elite_fitness else 0,
            'mean_fitness': np.mean(fitness),
            'elite_mean': np.mean(elite_fitness) if elite_fitness else 0,
        }
    
    def get_best_policy(self):
        """Retourne la meilleure politique (master)."""
        return self.master
    
    def save(self, path):
        torch.save({'master': self.master.state_dict(), 'gen': self.generation}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.master.load_state_dict(ckpt['master'])
        self.master.to(self.device)
        self.generation = ckpt.get('gen', 0)
