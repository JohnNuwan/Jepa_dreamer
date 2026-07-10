"""
es_agent.py — Evolution Strategies pour trading FTMO.
V5: antithetic corrigé (+ε/-ε sur MÊME marché), bias renforcé, pop_size doublé.
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
        # V5: bias renforcé pour résister à la dérive aléatoire des poids
        action_bias = torch.zeros(action_dim)
        action_bias[0] = -4.0    # HOLD  ← fortement pénalisé (diff de 7.0 vs BUY)
        action_bias[1] = +3.0    # BUY   ← fortement favorisé
        action_bias[2] = +3.0    # SELL  ← fortement favorisé
        action_bias[4] = +1.5    # SPLIT_BUY
        action_bias[5] = +1.5    # SPLIT_SELL
        action_bias[3] = +0.5    # CLOSE (léger encouragement à fermer)
        # PYRAMID(6), PARTIAL_CLOSE(7) = 0
        self.register_buffer('action_bias', action_bias)

        # V5: Masque pour geler les canaux HOLD/BUY/SELL du head[-1]
        # Ces 3 canaux sont déterminés UNIQUEMENT par le bias fixe.
        # Les poids du head[-1] correspondants sont exclus de l'évolution ES
        # pour empêcher la dérive qui contrerait le bias.
        self.register_buffer('frozen_action_mask', torch.ones(action_dim))
        self.frozen_action_mask[0] = 0.0  # HOLD  → gelé (bias seul)
        self.frozen_action_mask[1] = 0.0  # BUY   → gelé (bias seul)
        self.frozen_action_mask[2] = 0.0  # SELL  → gelé (bias seul)
    
    def forward(self, x, hidden=None):
        out, h = self.lstm(x, hidden)
        logits = self.head(out[:, -1, :])
        # V5: Appliquer le masque de gel sur les canaux d'action
        # Les canaux HOLD/BUY/SELL (0,1,2) sont masqués → contribution du réseau = 0
        # Seul le bias fixe détermine ces canaux
        return logits * self.frozen_action_mask + self.action_bias, h
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class ESAgent:
    """Evolution Strategies V5: antithetic corrigé, bias gelé, gradient propre."""
    
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
        print(f"ES Agent V5: {n_params:,} params | pop={pop_size} | σ={sigma} | lr={lr}")
        print(f"   GPUs: {len(self.devices)}× | Antithetic: ON (même marché) | Stochastique temp={temp_start}→{temp_end} sur {temp_decay_gens} gens")
        print(f"   Bias HOLD/SELL/BUY gelé via frozen_action_mask")
        
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
    
    def _evaluate_one(self, policy, env, steps, device_idx, temperature=1.0,
                      init_state=None):
        """Évalue une politique STOCHASTIQUE (sample softmax).
        
        fitness = PnL réalisé en % + bonus trading.
        Le sampling stochastique permet d'explorer même quand les logits sont plats.
        
        Si init_state est fourni, l'environnement est initialisé avec cet état
        (même symbole, même step) au lieu de reset() aléatoire.
        Cela permet à l'antithetic sampling de comparer +ε et -ε sur le MÊME marché.
        """
        INITIAL_BALANCE = FTMO_CONFIG['account_size']
        ZERO_TRADE_PENALTY = 50.0
        
        device = self.devices[device_idx]
        
        if init_state is not None:
            # BUGFIX: antithetic — utiliser le MÊME état initial que env_plus
            env.current_symbol = init_state['symbol']
            env.features = init_state['features']
            env.feature_names = init_state['feature_names']
            env.df = init_state['df']
            env.spec = init_state['spec']
            env.current_step = init_state['step']
            # Init manuelle (même que reset() mais sans aléatoire)
            env.balance = INITIAL_BALANCE
            env.peak_balance = env.balance
            env.daily_start_balance = env.balance
            env.prev_equity = env.balance
            env.positions = []
            env.trades_today = 0
            env.consecutive_losses = 0
            env.cooldown_until = 0
            env.last_trade_day = -1
            env.total_trades = 0
            env.winning_trades = 0
            env.buy_trades = 0
            env.sell_trades = 0
            env.episode_pnl = 0
            env.realized_pnl = 0
            env.bars_since_last_trade = 0
            env.max_dd_exceeded = False
            env.peak_equity = env.balance
            env.episode_reward = 0.0
            obs = env._get_obs()
        else:
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
        """Évalue toute la population avec antithetic sampling sur GPUs parallèles.
        
        V5 FIX: les paires antithetiques (+ε, -ε) sont évaluées sur le MÊME
        marché (même symbole, même step de départ). Avant, -ε utilisait un
        reset() aléatoire → marché différent → antithetic inutile.
        """
        master_vec = self._get_params_flat(self.master)
        
        # Étape 0 : capturer l'état initial de chaque env AVANT évaluation
        # pour que l'antithetic voie exactement le même marché.
        env_states = []
        for env_plus in envs:
            # reset() a déjà été appelé dans __init__ → on capture l'état
            env_states.append({
                'symbol': env_plus.current_symbol,
                'step': env_plus.current_step,
                'features': env_plus.features,
                'feature_names': env_plus.feature_names,
                'df': env_plus.df,
                'spec': env_plus.spec,
            })
        
        # Préparer toutes les tâches
        # Format: (policy, env, device_idx, anti, idx, init_state)
        all_tasks = []
        
        for i, ((policy_plus, noise, device), env_plus) in enumerate(zip(self.population, envs)):
            device_idx = i % len(self.devices)
            all_tasks.append((policy_plus, env_plus, device_idx, False, i, None))
        
        # Antithetic: master - noise, MÊME état initial que env_plus
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
            
            # BUGFIX: créer un env frais mais lui injecter le MÊME état initial
            anti_env = MultiSymbolEnvV4(
                envs[0].data_dict, lookback=envs[0].lookback,
                curriculum_episode=envs[0].curriculum_episode
            )
            
            all_tasks.append((anti_policy, anti_env, device_idx, True, i, env_states[i]))
        
        results_plus: list = [None] * self.pop_size
        results_minus: list = [None] * self.pop_size
        
        max_workers = len(self.devices) * 3
        temperature = self.get_temperature()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for policy, env, device_idx, anti, idx, init_state in all_tasks:
                future = executor.submit(self._evaluate_one, policy, env, steps,
                                        device_idx, temperature, init_state)
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

        # Collecter les fitness effectives des élites pour normalisation
        elite_fitness = []
        for (idx, (fit, (_, noise, device))), w in zip(elite, weights):
            elite_fitness.append(fit)
        elite_fitness = np.array(elite_fitness)

        # Normalisation par la magnitude max pour éviter que sign() amplifie le bruit
        max_abs_fit = max(np.max(np.abs(elite_fitness)), 1.0)

        for (idx, (fit, (_, noise, device))), w in zip(elite, weights):
            noise_primary = noise.to(self.primary_device)
            # V5: Utiliser la magnitude normalisée du fitness au lieu de sign()
            # sign() donne un poids égal à toutes les perturbations, amplifiant le bruit
            # La normalisation pondère les bons signaux plus fortement
            fit_clipped = np.clip(fit, -100, 100)
            grad += w * noise_primary * (fit_clipped / max_abs_fit)
        
        # Normaliser le gradient par le nombre d'élites
        grad = grad / max(1, n_elite)
        
        master_vec += self.lr * grad
        self._set_params_flat(self.master, master_vec)

        # V5: Remettre à zéro les poids du head[-1] pour les canaux gelés (HOLD, BUY, SELL)
        # Ces poids dérivent inutilement car leur contribution est masquée dans forward().
        # On les maintient à zéro pour éviter le gaspillage de capacité.
        with torch.no_grad():
            self.master.head[-1].weight.data[self.master.frozen_action_mask == 0] = 0.0

        self._create_population()
        self.generation += 1
        
        elite_fitness = [f for f, _ in [x[1] for x in elite]]
        grad_norm = grad.norm().item()  # V5: norme du gradient pour diagnostic
        return {
            'best_fitness': elite_fitness[0] if elite_fitness else 0,
            'mean_fitness': np.mean(fitness),
            'elite_mean': np.mean(elite_fitness) if elite_fitness else 0,
            'grad_norm': round(grad_norm, 6),
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
