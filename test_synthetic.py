"""
test_synthetic.py — Test de capacité RL : marché synthétique qui monte toujours.
Si l'agent n'apprend pas BUY, le problème vient du RL, pas des données.
"""
import sys, os, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_agent import ESAgent
from ppo_agent import PPOAgent
from config import N_ACTIONS, FTMO_CONFIG, HOLD, BUY, SELL, CLOSE


class TrendingEnv:
    """Marché synthétique qui monte de +0.1% par step avec un peu de bruit.
    L'action optimale est BUY."""
    
    def __init__(self, lookback=48):
        self.lookback = lookback
        self.price = 100.0
        self.balance = FTMO_CONFIG['account_size']
        self.position = None  # (direction, entry_price)
        self._last_open = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.done = False
        self.step_count = 0
    
    def reset(self):
        self.price = 100.0
        self.balance = FTMO_CONFIG['account_size']
        self.position = None
        self._last_open = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.done = False
        self.step_count = 0
        # Generate a fake observation: just repeating the price history
        obs = np.zeros((self.lookback, 296), dtype=np.float32)
        obs[:, 0] = np.linspace(self.price - self.lookback * 0.05, self.price, self.lookback)
        return obs
    
    def get_action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[HOLD] = True
        mask[BUY] = self.position is None
        mask[SELL] = self.position is None
        mask[CLOSE] = self.position is not None
        return mask
    
    def step(self, action):
        self.step_count += 1
        reward = 0.0
        
        old_unrealized = 0.0
        if self.position is not None:
            direction, entry = self.position
            old_unrealized = direction * (self.price - entry) / entry * 100
        
        # Price goes up with small noise
        self.price *= 1.001 + np.random.randn() * 0.0005
        self.price = max(1, self.price)
        
        # DENSE reward: delta of unrealized PnL each step
        new_unrealized = 0.0
        if self.position is not None:
            direction, entry = self.position
            new_unrealized = direction * (self.price - entry) / entry * 100
        reward = new_unrealized - old_unrealized  # delta PnL
        
        if action == BUY and self.position is None:
            self.position = (1, self.price)
            self._last_open = self.step_count
        elif action == SELL and self.position is None:
            self.position = (-1, self.price)
            self._last_open = self.step_count
        elif action == CLOSE and self.position is not None:
            direction, entry = self.position
            pnl = direction * (self.price - entry) * 100
            self.balance += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            reward += pnl / FTMO_CONFIG['account_size'] * 100
            self.position = None
            self._last_open = 0
        
        # Auto-close after 50 steps to force realization
        if self.position is not None and self.step_count - self._last_open > 50:
            self._simulate_close()
        
        if self.step_count > 500:
            self.done = True
        
        # Build fake observation with position info
        obs = np.zeros((self.lookback, 296), dtype=np.float32)
        obs[:, 0] = np.linspace(self.price - self.lookback * 0.05, self.price, self.lookback)
        # Signal crucial : PnL non-réalisé dans la colonne 1
        if self.position is not None:
            direction, entry = self.position
            unrealized = direction * (self.price - entry) / entry
            obs[:, 1] = unrealized
        # Position status dans la colonne 2
        if self.position is not None:
            obs[:, 2] = self.position[0]  # direction
        return obs, reward, self.done, {}
    
    def _simulate_close(self):
        """Force-close la position et réalise le PnL."""
        direction, entry = self.position
        pnl = direction * (self.price - entry) * 100
        self.balance += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.position = None
        self._last_open = 0


def test_es():
    print("\n=== Test ES sur marché synthétique (trending up) ===")
    agent = ESAgent(input_dim=296, hidden_dim=64, action_dim=N_ACTIONS,
                    pop_size=8, sigma=0.03, lr=0.02, device='cuda:0')
    
    for gen in range(30):
        envs = [TrendingEnv() for _ in range(agent.pop_size)]
        fitness = agent.evaluate_population(envs, steps=200)
        metrics = agent.evolve(fitness)
        
        if gen % 5 == 0:
            print(f"  Gen {gen:>3d}: best={metrics['best_fitness']:+.2f}% "
                  f"mean={metrics['mean_fitness']:+.2f}% elite={metrics['elite_mean']:+.2f}%")
    
    # Test best policy
    env = TrendingEnv()
    policy = agent.get_best_policy()
    obs = env.reset()
    lstm_hidden = None
    trades = []
    for _ in range(200):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            logits, lstm_hidden = policy(obs_t, lstm_hidden)
            action = logits.argmax(dim=-1).item()
        obs, _, done, _ = env.step(action)
        trades.append(action)
        if done: break
    
    n_buy = trades.count(BUY)
    n_sell = trades.count(SELL)
    n_hold = trades.count(HOLD)
    pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
    print(f"\n  Résultat final: PnL={pnl:+.2f}% | BUY={n_buy} SELL={n_sell} HOLD={n_hold}")
    print(f"  Trades: {env.total_trades} | Win rate: {env.winning_trades/max(1,env.total_trades)*100:.0f}%")
    
    if n_buy > n_sell + n_hold:
        print("  ✅ L'agent a appris BUY — le RL fonctionne !")
    else:
        print("  ❌ L'agent n'a pas appris BUY — problème RL")
    return n_buy > n_sell + n_hold


def test_ppo():
    print("\n=== Test PPO sur marché synthétique (trending up) ===")
    agent = PPOAgent(input_dim=296, hidden_dim=64, action_dim=N_ACTIONS,
                     lr=3e-4, gamma=0.99, lambda_=0.95, clip_eps=0.2,
                     entropy_coeff=0.05, value_coeff=0.5, device='cuda:0')
    
    env = TrendingEnv()
    obs = env.reset()
    for it in range(30):
        # Collect
        obs_buf = np.zeros((1, 100, 48, 296), dtype=np.float32)
        act_buf = np.zeros((1, 100), dtype=np.int32)
        lp_buf = np.zeros((1, 100), dtype=np.float32)
        rwd_buf = np.zeros((1, 100), dtype=np.float32)
        done_buf = np.zeros((1, 100), dtype=np.float32)
        val_buf = np.zeros((1, 100), dtype=np.float32)
        
        obs = env.reset()
        for t in range(100):
            mask = env.get_action_mask()
            actions, log_probs, values, probs = agent.get_action_batch(
                obs[np.newaxis], mask[np.newaxis])
            obs_buf[0, t] = obs
            act_buf[0, t] = actions[0]
            lp_buf[0, t] = log_probs[0]
            val_buf[0, t] = values[0]
            next_obs, reward, done, _ = env.step(int(actions[0]))
            rwd_buf[0, t] = reward
            done_buf[0, t] = float(done)
            if done:
                break
            obs = next_obs
        
        last_val = agent.get_value(obs[np.newaxis])[0]
        advantages, returns = agent.compute_gae(
            rwd_buf[0], val_buf[0], done_buf[0], last_val)
        agent.update(obs_buf[0], act_buf[0], lp_buf[0], returns, advantages)
    
    # Test
    env = TrendingEnv()
    obs = env.reset()
    trades = []
    for _ in range(200):
        mask = env.get_action_mask()
        a, _, _, _ = agent.get_action_batch(obs[np.newaxis], mask[np.newaxis], deterministic=True)
        obs, _, done, _ = env.step(int(a[0]))
        trades.append(int(a[0]))
        if done: break
    
    n_buy = trades.count(BUY)
    n_sell = trades.count(SELL)
    n_hold = trades.count(HOLD)
    pnl = (env.balance - FTMO_CONFIG['account_size']) / FTMO_CONFIG['account_size'] * 100
    print(f"  Résultat: PnL={pnl:+.2f}% | BUY={n_buy} SELL={n_sell} HOLD={n_hold}")
    
    if n_buy > n_sell + n_hold:
        print("  ✅ PPO a appris BUY — le RL fonctionne !")
        return True
    else:
        print("  ⚠️ PPO n'a pas appris BUY")
        return False


if __name__ == "__main__":
    es_ok = test_es()
    ppo_ok = test_ppo()
    print(f"\n=== BILAN ===")
    print(f"  ES  : {'✅ Fonctionne' if es_ok else '❌ Cassé'}")
    print(f"  PPO : {'✅ Fonctionne' if ppo_ok else '⚠️  Problème'}")
