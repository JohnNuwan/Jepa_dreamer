"""
es_agent.py — Evolution Strategies pour trading FTMO.
V4: fitness = PnL réalisé pur (balance finale), pas le reward dense.
Dual GPU, antithetic sampling, pénalité zéro-trade.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import FTMO_CONFIG, HOLD, BUY, SELL, SPLIT_BUY, SPLIT_SELL
from environment import MultiSymbolEnvV4


class ESPolicy(nn.Module):
    """Politique LSTM : (B, 48, 296) → 8 action logits + bias fixe anti-HOLD.
    Le bias est un buffer non-apprenable (exclu des paramètres ES)."""
    def __init__(self, input_dim=296, hidden_dim=128, action_dim=8, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim, bias=False)  # bias externe via buffer
        )
        # Bias fixe anti-HOLD — buffer (non appris par l'ES)
        # HOLD=0, BUY=1, SELL=2, CLOSE=3, SPLIT_BUY=4, SPLIT_SELL=5, PYRAMID=6, PARTIAL_CLOSE=7
        action_bias = torch.zeros(action_dim)
        action_bias[0] = -2.0    # HOLD
        action_bias[1] = +1.0    # BUY
        action_bias[2] = +1.0    # SELL
        action_bias[4] = +0.5    # SPLIT_BUY
        action_bias[5] = +0.5    # SPLIT_SELL
        # CLOSE(3), PYRAMID(6), PARTIAL_CLOSE(7) = 0
        self.register_buffer('action_bias', action_bias)
    
    def forward(self, x, hidden=None):
        out, h = self.lstm(x, hidden)
        logits = self.head(out[:, -1, :])
        return logits + self.action_bias, h
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class ESAgent:
    """Evolution Strategies V4: fitness = realized PnL, dual GPU."""
    
    def __init__(self, input_dim=296, hidden_dim=128, action_dim=8,
                 pop_size=16, sigma=0.015, lr=0.1, elite_frac=0.25, 
                 devices=('cuda:0', 'cuda:1'), temp_start=1.5, temp_end=0.3, temp_decay_gens=150):
        self.devices = [torch.device(d) for d in devices]
        self.primary_device = self.devices[0]
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.elite_frac = elite_frac
        self.action_dim = action_dim
        self.generation = 0
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay_gens = temp_decay_gens
        
        self.master = ESPolicy(input_dim, hidden_dim, action_dim).to(self.primary_device)
        n_params = self.master.count_params()
        print(f"ES Agent V4: {n_params:,} params | pop={pop_size} | σ={sigma} | lr={lr}")
        print(f"   GPUs: {len(self.devices)}× | Antithetic: ON | Stochastique temp={temp_start}→{temp_end} sur {temp_decay_gens} gens")
        
        self.population = []
        self._create_population()
    
    def _create_population(self):
        self.population = []
        master_vec = self._get_params_flat(self.master)
        
        for i in range(self.pop_size):
            noise = torch.randn_like(master_vec) * self.sigma
            device = self.devices[i % len(self.devices)]
            
            perturbed = ESPolicy(
                self.master.lstm.input_size,
                self.master.lstm.hidden_size,
                self.action_dim,
                self.master.lstm.num_layers
            ).to(device)
            
            master_on_device = master_vec.to(device)
            noise_on_device = noise.to(device)
            self._set_params_flat(perturbed, master_on_device + noise_on_device)
            
            self.population.append((perturbed, noise, device))
    
    def _get_params_flat(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    
    def _set_params_flat(self, model, flat):
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].view_as(p))
            offset += n
    
    def _evaluate_one(self, policy, env, steps, device_idx, temperature=1.0):
        """Évalue une politique STOCHASTIQUE (sample softmax).
        
        fitness = PnL réalisé en % + bonus trading.
        Le sampling stochastique permet d'explorer même quand les logits sont plats.
        """
        INITIAL_BALANCE = FTMO_CONFIG['account_size']
        ZERO_TRADE_PENALTY = 50.0
        
        device = self.devices[device_idx]
        obs = env.reset()
        num_trades = 0
        lstm_hidden = None
        
        for _ in range(steps):
            if env.current_step >= len(env.df) - 1:
                break
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits, lstm_hidden = policy(obs_t, lstm_hidden)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
                logits_masked = logits.masked_fill(~mask_t, float('-inf'))
                
                # Sampling stochastique avec température
                if temperature > 0:
                    probs = F.softmax(logits_masked / temperature, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = logits_masked.argmax(dim=-1).item()
            
            obs, _, done, _ = env.step(action)
            
            if action in (1, 2, 4, 5):
                num_trades += 1
            
            if done:
                break
        
        # Fitness = PnL réalisé en %
        realized_pnl_pct = (env.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0
        fitness = realized_pnl_pct
        
        # Bonus modéré pour l'activité
        if num_trades > 0:
            fitness += 2.0
        
        if num_trades == 0:
            fitness -= ZERO_TRADE_PENALTY
        
        return fitness, num_trades
    
    def get_temperature(self):
        """Température décroissante: temp_start → temp_end sur temp_decay_gens."""
        if self.generation >= self.temp_decay_gens:
            return self.temp_end
        progress = self.generation / self.temp_decay_gens
        return self.temp_start + (self.temp_end - self.temp_start) * progress
    
    def evaluate_population(self, envs: List, steps=500) -> tuple:
        """Évalue toute la population avec antithetic sampling sur GPUs parallèles."""
        master_vec = self._get_params_flat(self.master)
        
        # Préparer toutes les tâches
        all_tasks = []  # (policy, env, device_idx, anti, idx)
        
        for i, ((policy_plus, noise, device), env_plus) in enumerate(zip(self.population, envs)):
            device_idx = i % len(self.devices)
            all_tasks.append((policy_plus, env_plus, device_idx, False, i))
        
        # Antithetic: master - noise
        for i, ((_, noise, _), _) in enumerate(zip(self.population, envs)):
            device_idx = i % len(self.devices)
            anti_device = self.devices[device_idx]
            
            anti_policy = ESPolicy(
                self.master.lstm.input_size,
                self.master.lstm.hidden_size,
                self.action_dim,
                self.master.lstm.num_layers
            ).to(anti_device)
            
            master_on_anti = master_vec.to(anti_device)
            noise_on_anti = noise.to(anti_device)
            self._set_params_flat(anti_policy, master_on_anti - noise_on_anti)
            
            anti_env = MultiSymbolEnvV4(
                envs[0].data_dict, lookback=envs[0].lookback,
                curriculum_episode=envs[0].curriculum_episode
            )
            
            all_tasks.append((anti_policy, anti_env, device_idx, True, i))
        
        results_plus: list = [None] * self.pop_size
        results_minus: list = [None] * self.pop_size
        
        max_workers = len(self.devices) * 3
        temperature = self.get_temperature()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for policy, env, device_idx, anti, idx in all_tasks:
                future = executor.submit(self._evaluate_one, policy, env, steps, device_idx, temperature)
                futures[future] = (anti, idx)
            
            for future in as_completed(futures):
                anti, idx = futures[future]
                fitness, num_trades = future.result()
                if anti:
                    results_minus[idx] = fitness
                else:
                    results_plus[idx] = fitness
        
        fitness_plus = [f for f in results_plus]
        fitness_minus = [f for f in results_minus]
        
        effective_fitness = [p - m for p, m in zip(fitness_plus, fitness_minus)]
        
        trades_plus = sum(1 for f in fitness_plus if f > -30)
        trades_minus = sum(1 for f in fitness_minus if f > -30)
        
        return effective_fitness, fitness_plus, fitness_minus
    
    def evolve(self, fitness: List[float]):
        """Met à jour le master via sélection naturelle."""
        ranked = sorted(enumerate(zip(fitness, self.population)), 
                       key=lambda x: x[1][0], reverse=True)
        
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elite = ranked[:n_elite]
        
        ranks = np.arange(len(elite), 0, -1)
        total_rank = sum(ranks)
        weights = ranks / total_rank
        
        master_vec = self._get_params_flat(self.master)
        grad = torch.zeros_like(master_vec)
        
        for (idx, (fit, (_, noise, device))), w in zip(elite, weights):
            noise_primary = noise.to(self.primary_device)
            # Normaliser la fitness pour éviter les explosions de gradient
            fit_clipped = np.clip(fit, -100, 100)
            grad += w * noise_primary * np.sign(fit_clipped)
        
        # Normaliser le gradient par le nombre d'élites
        grad = grad / max(1, n_elite)
        
        master_vec += self.lr * grad
        self._set_params_flat(self.master, master_vec)
        
        self._create_population()
        self.generation += 1
        
        elite_fitness = [f for f, _ in [x[1] for x in elite]]
        return {
            'best_fitness': elite_fitness[0] if elite_fitness else 0,
            'mean_fitness': np.mean(fitness),
            'elite_mean': np.mean(elite_fitness) if elite_fitness else 0,
        }
    
    def get_best_policy(self):
        return self.master
    
    def save(self, path):
        torch.save({'master': self.master.state_dict(), 'gen': self.generation}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.master.load_state_dict(ckpt['master'])
        self.master.to(self.primary_device)
        self.generation = ckpt.get('gen', 0)
