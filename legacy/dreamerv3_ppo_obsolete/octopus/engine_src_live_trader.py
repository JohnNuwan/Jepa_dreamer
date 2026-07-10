"""Trader live pour le moteur Octopus — exécution en temps réel.

Connecte le modèle MuZero entraîné à MT5 pour le trading en direct
avec les règles FTMO actives.

References:
    MuZero Pro Trader V3.1 (JohnNuwan, 2024)
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MuZeroConfig
from src.networks.jepa import TSJEPA
from src.networks.world_model import RSSMWorldModel
from src.networks.actor_critic import ActorCritic
from src.mcts import MCTS
from src.environment import OctopusTradingEnv, FTMOEnforcer, SLBESystem


@dataclass
class LiveTraderConfig:
    """Configuration du trader live."""
    symbol: str = "XAUUSD"
    model_path: str = "weights/best_model.pth"
    cooldown_seconds: int = 1800  # 30 minutes
    inactivity_threshold: int = 60  # 60 pas = KICK
    live_buffer_dir: str = "results/live_buffer"


class LiveTrader:
    """Trader temps réel pour Octopus.
    
    Attributes:
        config: Configuration du trader live.
        muzero_config: Configuration du moteur MuZero.
        jepa: Encodeur JEPA chargé.
        world_model: World Model RSSM chargé.
        actor_critic: Actor-Critic chargé.
        mcts: MCTS pour planification.
        ftmo: Moteur de règles FTMO.
        slbe: Système Stop Loss Break Even.
    """
    
    def __init__(
        self,
        live_config: Optional[LiveTraderConfig] = None,
        muzero_config: Optional[MuZeroConfig] = None
    ) -> None:
        """Initialise le trader live.
        
        Args:
            live_config: Configuration live.
            muzero_config: Configuration MuZero.
        """
        self.live_config = live_config or LiveTraderConfig()
        self.muzero_config = muzero_config or MuZeroConfig()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialiser les modules
        self.jepa = TSJEPA(
            input_dim=self.muzero_config.observation_shape[0],
            seq_len=self.muzero_config.observation_shape[1],
            embedding_dim=self.muzero_config.jepa.embedding_dim
        ).to(self.device)
        
        self.world_model = RSSMWorldModel(
            stoch_size=self.muzero_config.world_model.stoch_size,
            stoch_classes=self.muzero_config.world_model.stoch_classes,
            deter_size=self.muzero_config.world_model.deter_size,
            action_dim=self.muzero_config.action_space_size,
            embedding_dim=self.muzero_config.jepa.embedding_dim
        ).to(self.device)
        
        self.actor_critic = ActorCritic(
            stoch_size=self.muzero_config.world_model.stoch_size,
            stoch_classes=self.muzero_config.world_model.stoch_classes,
            deter_size=self.muzero_config.world_model.deter_size,
            action_dim=self.muzero_config.action_space_size
        ).to(self.device)
        
        self.mcts = MCTS(
            action_dim=self.muzero_config.action_space_size,
            num_simulations=self.muzero_config.num_simulations,
            discount=self.muzero_config.discount
        )
        
        # Composants de trading
        self.ftmo = FTMOEnforcer(self.muzero_config.ftmo.initial_capital)
        self.slbe = SLBESystem()
        
        # État
        self.position = 0
        self.entry_price = 0.0
        self.current_lots = 0.0
        self.last_trade_time = datetime.min
        self.steps_since_trade = 0
        self.step_count = 0
        
        # Dossiers
        os.makedirs(self.live_config.live_buffer_dir, exist_ok=True)
    
    def load_model(self, path: Optional[str] = None) -> None:
        """Charge les poids du modèle entraîné.
        
        Args:
            path: Chemin du checkpoint (utilise la config si None).
        """
        model_path = path or self.live_config.model_path
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.jepa.load_state_dict(checkpoint['jepa'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        
        print(f"✅ Modèle chargé depuis {model_path} (step {checkpoint.get('step', '?')})")
    
    def step(self, market_data: np.ndarray) -> int:
        """Exécute un pas de décision sur des données de marché live.
        
        Args:
            market_data: Observation du marché (lookback, features).
            
        Returns:
            Action choisie (0-4).
        """
        self.step_count += 1
        
        # Encoder l'observation
        obs_t = torch.FloatTensor(market_data).unsqueeze(0).to(self.device)
        embedding = self.jepa(obs_t)
        
        # Inférence initiale (World Model)
        with torch.no_grad():
            # État initial
            if self.step_count == 1:
                self.stoch = torch.zeros(
                    1, self.muzero_config.world_model.stoch_size,
                    self.muzero_config.world_model.stoch_classes
                ).to(self.device)
                self.deter = torch.zeros(
                    1, self.muzero_config.world_model.deter_size
                ).to(self.device)
            
            prev_action = torch.zeros(
                1, self.muzero_config.action_space_size
            ).to(self.device)
            
            post, _, self.deter, _ = self.world_model.transition.forward(
                prev_stoch=self.stoch,
                prev_action=prev_action,
                prev_embedding=embedding
            )
            self.stoch = post
        
        # MCTS
        state_for_mcts = torch.cat([
            self.stoch.reshape(1, -1), self.deter
        ], dim=-1)
        
        visit_dist = self.mcts.search(
            state_for_mcts, self.world_model, self.device
        )
        
        best_action = np.argmax(visit_dist).item()
        
        # Anti-overtrading : cooldown
        time_since_trade = (datetime.now() - self.last_trade_time).total_seconds()
        if time_since_trade < self.live_config.cooldown_seconds:
            best_action = 0  # Hold
        
        # KICK : forcer une action si inactivité
        if self.steps_since_trade > self.live_config.inactivity_threshold and best_action == 0:
            alt_actions = [(a, v) for a, v in enumerate(visit_dist) if a != 0]
            if alt_actions:
                best_action = max(alt_actions, key=lambda x: x[1])[0]
        
        # Health bar
        life_pct = max(0, (self.live_config.inactivity_threshold - self.steps_since_trade) / 
                       self.live_config.inactivity_threshold)
        life_bars = int(life_pct * 10)
        print(f"\r🔍 {self.live_config.symbol} Action={best_action} "
              f"| Life: {'█'*life_bars}{'░'*(10-life_bars)} "
              f"| Steps since trade: {self.steps_since_trade}", end="")
        
        if best_action != 0:
            self.last_trade_time = datetime.now()
            self.steps_since_trade = 0
        else:
            self.steps_since_trade += 1
        
        return best_action
    
    def run_loop(self, data_source, interval_seconds: int = 60) -> None:
        """Boucle de trading live.
        
        Args:
            data_source: Source de données (fonction qui retourne une observation).
            interval_seconds: Intervalle entre les pas de décision (secondes).
        """
        print(f"🐙 Octopus Live Trader — {self.live_config.symbol}")
        print(f"   Interval: {interval_seconds}s | Cooldown: {self.live_config.cooldown_seconds}s")
        print(f"   FTMO Daily Loss: ${self.ftmo.max_daily_loss:.0f}")
        print(f"   FTMO Total Loss: ${self.ftmo.max_total_loss:.0f}")
        print()
        
        self.load_model()
        
        try:
            while True:
                market_data = data_source()
                action = self.step(market_data)
                
                # Exécuter l'action via gRPC vers le service Execution
                self._execute_action(action)
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Trading arrêté")
    
    def _execute_action(self, action: int) -> Dict[str, Any]:
        """Exécute une action de trading via le service gRPC.
        
        Args:
            action: Action à exécuter (0-4).
            
        Returns:
            Résultat de l'exécution.
        """
        # En production : appel gRPC vers le service Execution (Rust)
        action_names = {0: "Hold", 1: "Buy", 2: "Sell", 3: "Split", 4: "Close"}
        return {"action": action_names.get(action, "Unknown")}


if __name__ == '__main__':
    trader = LiveTrader()
    
    # Démo : source de données synthétique
    def demo_source() -> np.ndarray:
        return np.random.randn(
            trader.muzero_config.observation_shape[1],
            trader.muzero_config.observation_shape[0]
        ).astype(np.float32)
    
    trader.run_loop(demo_source, interval_seconds=5)