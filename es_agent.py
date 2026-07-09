"""
es_agent.py — Evolution Strategies pour trading FTMO.
Population d'agents LSTM évalués en parallèle, sélection naturelle sur le PnL.
V2: antithetic sampling, pénalité zéro-trade, évaluation déterministe.
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
    """Evolution Strategies V2: antithetic sampling + pénalité zéro-trade."""
    
    def __init__(self, input_dim=296, hidden_dim=128, action_dim=8,
                 pop_size=16, sigma=0.015, lr=0.01, elite_frac=0.25, device='cuda:0'):
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
        print(f"ES Agent V2: {n_params:,} params | pop={pop_size} | σ={sigma} | lr={lr} | GPU: {device}")
        print(f"   Antithetic sampling: ON | Zero-trade penalty: ON")
        
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
    
    def evaluate_single(self, policy, env, steps=1000) -> tuple:
        """Évalue une politique DÉTERMINISTE (argmax), retourne (reward_total, num_trades)."""
        obs = env.reset()
        total_reward = 0.0
        num_trades = 0
        lstm_hidden = None
        
        for _ in range(steps):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                logits, lstm_hidden = policy(obs_t, lstm_hidden)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
                logits_masked = logits.masked_fill(~mask_t, float('-inf'))
                # DÉTERMINISTE: toujours argmax, pas d'exploration aléatoire
                action = logits_masked.argmax(dim=-1).item()
            
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Compter les trades (BUY, SELL, SPLIT_BUY, SPLIT_SELL)
            if action in (1, 2, 4, 5):
                num_trades += 1
            
            if done:
                break
        
        return float(total_reward), num_trades
    
    def evaluate_population(self, envs: List, steps=500) -> tuple:
        """Évalue toute la population avec antithetic sampling.
        
        Pour chaque vecteur de bruit ε, on évalue :
        - master + ε  (fitness positive)
        - master - ε  (fitness négative, avec une copie anti-perturbée)
        
        fitness_effective[i] = (fitness(master+ε) - fitness(master-ε)) × signe_arbitraire
        Ça donne un gradient 2× plus précis (antithetic variates).
        
        PENALITÉ: si une politique fait 0 trades, fitness -= 100.0
        """
        master_vec = self._get_params_flat(self.master)
        
        fitness_plus = []
        fitness_minus = []
        
        for i, ((policy_plus, noise), env) in enumerate(zip(self.population, envs)):
            # Évaluer master + noise (déjà créé dans population)
            fit_p, trades_p = self.evaluate_single(policy_plus, env, steps)
            
            # Créer master - noise pour évaluation antithetic
            anti_policy = ESPolicy(
                self.master.lstm.input_size,
                self.master.lstm.hidden_size,
                self.action_dim,
                self.master.lstm.num_layers
            ).to(self.device)
            self._set_params_flat(anti_policy, master_vec - noise)
            
            # Créer un env séparé pour l'anti-évaluation
            from environment import MultiSymbolEnvV4
            anti_env = MultiSymbolEnvV4(
                env.data_dict, lookback=env.lookback, 
                curriculum_episode=env.curriculum_episode
            )
            
            fit_m, trades_m = self.evaluate_single(anti_policy, anti_env, steps)
            
            # Pénalité zéro-trade des deux côtés
            ZERO_TRADE_PENALTY = 100.0
            if trades_p == 0:
                fit_p -= ZERO_TRADE_PENALTY
            if trades_m == 0:
                fit_m -= ZERO_TRADE_PENALTY
            
            fitness_plus.append(fit_p)
            fitness_minus.append(fit_m)
        
        # Fitness effective = différence (le gradient pointe vers +ε si fit_plus > fit_minus)
        effective_fitness = [p - m for p, m in zip(fitness_plus, fitness_minus)]
        
        # DEBUG: afficher stats
        trades_plus = sum(1 for f in fitness_plus if f > -50)  # approx: pas pénalisé
        trades_minus = sum(1 for f in fitness_minus if f > -50)
        
        return effective_fitness, fitness_plus, fitness_minus
    
    def evolve(self, fitness: List[float]):
        """Met à jour le master via sélection naturelle."""
        # Rank by fitness (higher is better)
        ranked = sorted(enumerate(zip(fitness, self.population)), 
                       key=lambda x: x[1][0], reverse=True)
        
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elite = ranked[:n_elite]
        
        # Weighted average of elite noises
        elite_fitness = [f for f, _ in [x[1] for x in elite]]
        # Utiliser softmax des rangs pour les poids (plus robuste que fitness brute)
        ranks = np.arange(len(elite_fitness), 0, -1)  # [n_elite, n_elite-1, ..., 1]
        total_rank = sum(ranks)
        weights = ranks / total_rank
        
        # Compute gradient estimate
        master_vec = self._get_params_flat(self.master)
        grad = torch.zeros_like(master_vec)
        for (idx, (fit, (_, noise))), w in zip(elite, weights):
            grad += w * noise * np.sign(fit)  # signe: direction du gradient
        
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
