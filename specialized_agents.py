"""
specialized_agents.py — Architecture multi-agent V6 : un agent ES par symbole.
Chaque agent est spécialisé sur UN symbole avec ses propres features multi-TF.
Le MasterAgent agrège les sorties et alloue le capital.

V6: agents spécialisés + maître d'allocation.
"""
import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS, ACTION_NAMES,
                     HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, CURRICULUM_CONFIG)
from environment import MultiSymbolEnvV4
from es_agent import ESPolicy, ESAgent


class SingleSymbolEnv:
    """Wrapper qui force MultiSymbolEnvV4 sur un seul symbole."""
    def __init__(self, data_dict, symbol, lookback=48, curriculum_episode=0):
        self.symbol = symbol
        self.env = MultiSymbolEnvV4(data_dict, lookback=lookback, 
                                     curriculum_episode=curriculum_episode)
        # Forcer le symbole après construction (reset() dans __init__ l'a mis aléatoire)
        if symbol in data_dict:
            features, feat_names, df = data_dict[symbol]
            self.env.current_symbol = symbol
            self.env.features = features
            self.env.feature_names = feat_names
            self.env.df = df
            self.env.spec = SYMBOLS[symbol]
    
    def __getattr__(self, name):
        """Délègue tout le reste à self.env."""
        return getattr(self.env, name)


class SpecializedAgent:
    """Un agent ES entraîné sur un seul symbole."""
    def __init__(self, symbol, data_dict, input_dim, hidden_dim=128, 
                 pop_size=8, sigma=0.02, lr=0.1, devices=('cuda:0', 'cuda:1')):
        self.symbol = symbol
        self.data_dict = data_dict
        self.devices = devices
        
        self.es = ESAgent(input_dim=input_dim, hidden_dim=hidden_dim,
                          action_dim=N_ACTIONS, pop_size=pop_size, sigma=sigma,
                          lr=lr, elite_frac=0.25, devices=devices)
        
        self.best_val_pnl = -float('inf')
        self.best_gen = 0
    
    def _create_envs(self, gen=0):
        """Crée pop_size environnements single-symbol."""
        return [SingleSymbolEnv(self.data_dict, self.symbol, 
                                curriculum_episode=gen) 
                for _ in range(self.es.pop_size)]
    
    def train_generation(self, gen, eval_steps=500):
        """Une génération d'entraînement ES. Retourne les métriques."""
        envs = self._create_envs(gen)
        
        try:
            eff_fitness, fp, fm = self.es.evaluate_population(envs, steps=eval_steps)
        except Exception as e:
            print(f"   [{self.symbol}] ❌ CRASH gen {gen}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        metrics = self.es.evolve(eff_fitness)
        metrics['trades_plus'] = sum(1 for f in fp if f > -50)
        metrics['trades_minus'] = sum(1 for f in fm if f > -50)
        
        return metrics
    
    def validate(self, val_steps=500, val_bars=3000):
        """Validation déterministe sur les dernières barres du symbole."""
        env = SingleSymbolEnv(self.data_dict, self.symbol, curriculum_episode=9999)
        
        # Init manuelle (pattern #6: pas de reset après set_params)
        features, feat_names, df = self.data_dict[self.symbol]
        env.current_symbol = self.symbol
        env.features = features
        env.feature_names = feat_names
        env.df = df
        env.spec = SYMBOLS[self.symbol]
        env.current_step = env.lookback + len(df) - val_bars
        
        env.balance = FTMO_CONFIG['account_size']
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
        
        policy = self.es.get_best_policy()
        lstm_hidden = None
        first_logits = None
        
        for _ in range(val_steps):
            if env.current_step >= len(df) - 1:
                break
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.es.primary_device)
                logits, lstm_hidden = policy(obs_t, lstm_hidden)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.es.primary_device)
                logits_masked = logits.masked_fill(~mask_t, float('-inf'))
                action = logits_masked.argmax(dim=-1).item()
                if first_logits is None:
                    first_logits = logits_masked.cpu().numpy().flatten()
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
        wr = env.winning_trades / max(1, env.total_trades) * 100
        
        # Diagnostic logits
        if first_logits is not None:
            buy_gap = first_logits[BUY] - first_logits[HOLD] if not np.isinf(first_logits[BUY]) else 0
            sell_gap = first_logits[SELL] - first_logits[HOLD] if not np.isinf(first_logits[SELL]) else 0
            buy_vs_sell = first_logits[BUY] - first_logits[SELL] if not np.isinf(first_logits[BUY]) and not np.isinf(first_logits[SELL]) else 0
        else:
            buy_gap = sell_gap = buy_vs_sell = 0
        
        return {
            'pnl': pnl, 'trades': env.total_trades, 'wr': wr,
            'buy_gap': buy_gap, 'sell_gap': sell_gap, 'buy_vs_sell': buy_vs_sell,
            'buy_trades': env.buy_trades, 'sell_trades': env.sell_trades,
        }
    
    def save(self, path):
        self.es.save(path)
    
    def load(self, path):
        self.es.load(path)


class MasterAgent(nn.Module):
    """Agent maître : reçoit les features agrégées des agents spécialisés
    et produit des poids d'allocation de capital par symbole + une décision
    globale HOLD (ne pas trader)."""
    def __init__(self, n_symbols, features_per_symbol=8, hidden_dim=64):
        super().__init__()
        self.n_symbols = n_symbols
        input_dim = n_symbols * features_per_symbol
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_symbols + 1)  # +1 = HOLD global
        )
    
    def forward(self, agent_features):
        """
        agent_features: (B, n_symbols * features_per_symbol)
        Retourne: allocation weights (B, n_symbols+1) — softmax sur tous
        """
        logits = self.net(agent_features)
        return F.softmax(logits, dim=-1)


class MultiAgentOrchestrator:
    """Coordonne l'entraînement de N agents spécialisés + maître."""
    def __init__(self, symbols=None, n_generations=100, pop_size=8, 
                 eval_steps=500, hidden_dim=128, sigma=0.02, lr=0.1,
                 save_dir='checkpoints_specialized', data_dir='data'):
        self.symbols = symbols or ACTIVE_SYMBOLS
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.eval_steps = eval_steps
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Charger toutes les données
        print("📦 Chargement des données...")
        from features_v2 import compute_multi_tf_features, compute_correlations
        import pandas as pd
        
        self.data_dict = {}
        for symbol in self.symbols:
            path = os.path.join(data_dir, f'{symbol}_m15.csv')
            if not os.path.exists(path):
                print(f"   ⚠️  {symbol}: fichier introuvable, ignoré")
                continue
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48, symbol=symbol)
            self.data_dict[symbol] = (features, feat_names, df_processed)
            print(f"  ✅ {symbol}: {len(df)} bars → {features.shape[1]} features")
        
        self.symbols = [s for s in self.symbols if s in self.data_dict]
        print(f"   {len(self.symbols)} symboles chargés")
        
        # Déterminer input_dim via l'environnement (ajoute symbol embedding, corrélations, etc.)
        temp_env = SingleSymbolEnv(self.data_dict, self.symbols[0], curriculum_episode=0)
        self.n_features = temp_env.n_features
        print(f"   Input dim: {self.n_features} (incl. embeddings, corrélations)")
        
        # Créer un agent spécialisé par symbole
        print(f"\n🤖 Création de {len(self.symbols)} agents spécialisés...")
        self.agents: Dict[str, SpecializedAgent] = {}
        devices = ('cuda:0', 'cuda:1')
        
        for i, symbol in enumerate(self.symbols):
            device_pair = (devices[i % 2], devices[(i+1) % 2]) if len(devices) > 1 else devices
            agent = SpecializedAgent(
                symbol=symbol, data_dict=self.data_dict,
                input_dim=self.n_features, hidden_dim=hidden_dim,
                pop_size=pop_size, sigma=sigma, lr=lr, devices=device_pair
            )
            self.agents[symbol] = agent
            print(f"  ✅ {symbol}: {agent.es.master.count_params():,} params")
        
        # Agent maître (créé après pré-entraînement des esclaves)
        self.master = None
        self.features_per_symbol = 8  # [buy_gap, sell_gap, buy_vs_sell, wr, trades, ...]
        
        self.log_path = os.path.join(save_dir, 'training.log')
        self.metrics_path = os.path.join(save_dir, 'metrics.json')
    
    def train_phase1(self, n_gens=50):
        """Phase 1: entraînement indépendant de chaque agent spécialisé."""
        print(f"\n{'='*60}")
        print(f"PHASE 1: Entraînement indépendant ({n_gens} générations par agent)")
        print(f"{'='*60}")
        
        log_file = open(self.log_path, 'w')
        all_metrics = []
        
        for gen in range(n_gens):
            t0 = time.time()
            gen_metrics = {'generation': gen, 'agents': {}}
            
            # Entraîner chaque agent sur une génération
            for symbol, agent in self.agents.items():
                t_a = time.time()
                metrics = agent.train_generation(gen, eval_steps=self.eval_steps)
                if metrics is None:
                    continue
                gen_metrics['agents'][symbol] = {
                    'best_fitness': round(metrics['best_fitness'], 2),
                    'mean_fitness': round(metrics['mean_fitness'], 2),
                    'elite_mean': round(metrics['elite_mean'], 2),
                    'trades_plus': metrics['trades_plus'],
                    'trades_minus': metrics['trades_minus'],
                }
            
            # Validation toutes les 10 générations
            if gen % 10 == 0:
                print(f"\n--- Gen {gen} Validation ---")
                val_results = {}
                for symbol, agent in self.agents.items():
                    val = agent.validate(val_steps=500)
                    val_results[symbol] = val
                    status = "🟢" if val['pnl'] > 0 else "🔴"
                    direction = "BUY" if val['buy_vs_sell'] > 1.0 else ("SELL" if val['buy_vs_sell'] < -1.0 else "≈")
                    print(f"  {status} {symbol:12s} | PnL={val['pnl']:+.2f}% | "
                          f"tr={val['trades']:>2d} wr={val['wr']:.0f}% | "
                          f"BUYvsSELL={val['buy_vs_sell']:+.2f} {direction} | "
                          f"buy={val['buy_trades']} sell={val['sell_trades']}")
                
                # Sauvegarder les meilleurs agents
                for symbol, agent in self.agents.items():
                    if val_results[symbol]['pnl'] > agent.best_val_pnl:
                        agent.best_val_pnl = val_results[symbol]['pnl']
                        agent.best_gen = gen
                        agent.save(os.path.join(self.save_dir, f'{symbol}_best.pt'))
                
                gen_metrics['validation'] = {s: round(v['pnl'], 2) for s, v in val_results.items()}
            
            t = time.time() - t0
            # Log compact: une ligne par génération
            best_fits = [m['best_fitness'] for m in gen_metrics['agents'].values()]
            mean_fits = [m['mean_fitness'] for m in gen_metrics['agents'].values()]
            symbols_str = ' | '.join(
                    f"{s}={gen_metrics['agents'][s]['best_fitness']:+.2f}" 
                    for s in self.symbols[:4])
            log = (f"Gen {gen:>4d} | {t:.1f}s | "
                   f"best_avg={np.mean(best_fits):+.2f} "
                   f"mean_avg={np.mean(mean_fits):+.2f} | "
                   f"{symbols_str}")
            print(log)
            log_file.write(log + '\n')
            log_file.flush()
            
            all_metrics.append(gen_metrics)
            
            # Sauvegarde périodique des métriques
            if gen % 25 == 0:
                with open(self.metrics_path, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
            
            torch.cuda.empty_cache()
        
        # Sauvegarde finale
        for symbol, agent in self.agents.items():
            agent.save(os.path.join(self.save_dir, f'{symbol}_final.pt'))
        with open(self.metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        log_file.close()
        print(f"\n✅ Phase 1 terminée")
    
    def train_phase2(self, n_gens=50):
        """Phase 2: entraînement du maître + fine-tuning joint."""
        # TODO: à implémenter après validation de la phase 1
        pass
    
    def run(self):
        """Pipeline complet: phase 1 → phase 2."""
        self.train_phase1(n_gens=self.n_generations)
        # Phase 2 sera activée quand la phase 1 donne des résultats


if __name__ == "__main__":
    orch = MultiAgentOrchestrator(
        symbols=ACTIVE_SYMBOLS,
        n_generations=150,    # 150 générations par agent
        pop_size=8,           # pop plus petite car gradient plus propre
        eval_steps=500,       # un peu moins de steps (plus rapide)
        hidden_dim=128,
        sigma=0.02,
        lr=0.1,
        save_dir='checkpoints_specialized'
    )
    orch.run()