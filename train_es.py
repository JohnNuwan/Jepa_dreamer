"""
train_es.py — Evolution Strategies V2 : antithetic sampling, pénalité zéro-trade.
Population évaluée avec antithetic variates pour un gradient 2× plus précis.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, N_ACTIONS, ACTION_NAMES,
                     HOLD, CURRICULUM_CONFIG)
from features_v2 import compute_multi_tf_features, compute_correlations
from environment import MultiSymbolEnvV4
from es_agent import ESAgent


def load_all_symbols(data_dir='data'):
    data_dict = {}
    for symbol in ACTIVE_SYMBOLS:
        path = os.path.join(data_dir, f'{symbol}_m15.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        features, feat_names, df_processed = compute_multi_tf_features(df, lookback=48, symbol=symbol)
        data_dict[symbol] = (features, feat_names, df_processed)
        print(f"  ✅ {symbol}: {len(df)} bars → {features.shape[1]} features")
    return data_dict


class ESTrainer:
    def __init__(self, n_generations=500, pop_size=16, eval_steps=2000,
                 hidden_dim=128, sigma=0.015, lr=0.01, save_dir='checkpoints_es'):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.eval_steps = eval_steps
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print("📦 Chargement des données...")
        self.data_dict = load_all_symbols()
        print(f"   {len(self.data_dict)} symboles")
        
        self.correlations = compute_correlations(self.data_dict)
        
        temp_env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=0)
        self.n_features = temp_env.n_features
        
        self.agent = ESAgent(input_dim=self.n_features, hidden_dim=hidden_dim,
                             action_dim=N_ACTIONS, pop_size=pop_size, sigma=sigma,
                             lr=lr, elite_frac=0.25, devices=('cuda:0', 'cuda:1'))
        
        self.best_val_pnl = -float('inf')
        self.best_val_gen = 0
        self.log_path = os.path.join(save_dir, 'training_es.log')
        self.metrics_path = os.path.join(save_dir, 'metrics_es.json')
        self.metrics = []
    
    def _create_envs(self, gen=0):
        return [MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=gen)
                for _ in range(self.pop_size)]
    
    def run(self):
        log_file = open(self.log_path, 'w')
        print(f"\n=== ES Training V4 ({self.n_generations} gens, pop={self.pop_size}) ===")
        print(f"   {self.eval_steps} steps/eval, σ={self.agent.sigma}, lr={self.agent.lr}")
        print(f"   Stochastique temp={self.agent.temp_start}→{self.agent.temp_end} | Dual GPU")
        print()
        
        for gen in range(self.n_generations):
            t0 = time.time()
            
            # Créer environnements
            envs = self._create_envs(gen)
            
            # Évaluer la population (retourne effective_fitness, fitness_plus, fitness_minus)
            effective_fitness, fitness_plus, fitness_minus = \
                self.agent.evaluate_population(envs, steps=self.eval_steps)
            
            # Stats debug
            trades_plus = sum(1 for f in fitness_plus if f > -50)
            trades_minus = sum(1 for f in fitness_minus if f > -50)
            
            # Évoluer sur la fitness effective
            evo_metrics = self.agent.evolve(effective_fitness)
            
            # Validation avec la meilleure politique (toutes les 5 gens pour plus de feedback)
            val_pnl, val_trades, val_wr = 0, 0, 0
            if gen % 5 == 0:
                val_pnl, val_trades, val_wr = self._validate()
            
            # Log
            t = time.time() - t0
            log = (f"Gen {gen:>4d} | {t:.1f}s | "
                   f"best={evo_metrics['best_fitness']:+.2f} "
                   f"mean={evo_metrics['mean_fitness']:+.2f} "
                   f"elite={evo_metrics['elite_mean']:+.2f} "
                   f"t±={trades_plus}/{trades_minus} "
                   f"| val={val_pnl:+.2f}% tr={val_trades} wr={val_wr:.0f}%")
            print(log)
            log_file.write(log + '\n')
            log_file.flush()
            
            self.metrics.append({
                'generation': gen,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in evo_metrics.items()},
                'val_pnl': round(val_pnl, 2),
                'val_trades': val_trades,
                'trades_plus': trades_plus,
                'trades_minus': trades_minus,
            })
            
            if gen % 50 == 0:
                with open(self.metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            
            if val_pnl > self.best_val_pnl:
                self.best_val_pnl = val_pnl
                self.best_val_gen = gen
                self.agent.save(os.path.join(self.save_dir, 'best.pt'))
                print(f"   🏆 NEW BEST: {val_pnl:+.2f}% (gen {gen}, {val_trades} trades)")
        
        self.agent.save(os.path.join(self.save_dir, 'final.pt'))
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n✅ Done! Best val PnL: {self.best_val_pnl:+.2f}% (gen {self.best_val_gen})")
        log_file.close()
    
    def _validate(self):
        symbol = 'XAUUSD'
        
        env = MultiSymbolEnvV4(self.data_dict, lookback=48, curriculum_episode=9999)
        env.current_symbol = symbol
        env.features, env.feature_names, env.df = self.data_dict[symbol]
        env.spec = SYMBOLS[symbol]
        env.current_step = env.lookback + len(env.df) - 3000
        env.reset()
        obs = env._get_obs()
        
        policy = self.agent.get_best_policy()
        lstm_hidden = None
        
        for _ in range(500):
            if env.current_step >= len(env.df) - 1:
                break
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.primary_device)
                logits, lstm_hidden = policy(obs_t, lstm_hidden)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.agent.primary_device)
                logits_masked = logits.masked_fill(~mask_t, float('-inf'))
                action = logits_masked.argmax(dim=-1).item()
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
        wr = env.winning_trades / max(1, env.total_trades) * 100
        return pnl, env.total_trades, wr


if __name__ == "__main__":
    trainer = ESTrainer(n_generations=200, pop_size=16, eval_steps=1000,
                        hidden_dim=128, sigma=0.02, lr=0.1)
    trainer.run()
