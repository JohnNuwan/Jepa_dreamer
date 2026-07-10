"""Module d'environnement de trading Gymnasium pour le moteur Octopus.

Définit l'environnement de trading XAUUSD avec :
- Espace d'actions discret (5 actions : Hold/Buy/Sell/Split/Close)
- Système SLBE (Stop Loss Break Even)
- Reward shaping style MuZero Pro Trader V3.1 "Hunger Mode"
- Règles FTMO intégrées

References:
    MuZero Pro Trader V3.1 "Hunger Mode" (JohnNuwan, 2024)
    CommissionTrinityEnvV3 (github.com/JohnNuwan/Muzero_Pro_Trader)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass, field

# Import conditionnel du module macro
try:
    from macro.src.macro_features import MacroFeatureEngine
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False


@dataclass
class Trade:
    """Représente un trade exécuté.
    
    Attributes:
        direction: Direction du trade (1 = LONG, -1 = SHORT).
        entry_price: Prix d'entrée.
        entry_step: Pas de temps d'entrée.
        lots: Taille de la position en lots standards.
        exit_price: Prix de sortie (None si ouvert).
        exit_step: Pas de temps de sortie (None si ouvert).
        pnl: Profit and loss réalisé (None si ouvert).
    """
    direction: int
    entry_price: float
    entry_step: int
    lots: float
    exit_price: Optional[float] = None
    exit_step: Optional[int] = None
    pnl: Optional[float] = None


class SLBESystem:
    """Système Stop Loss Break Even.
    
    Active un stop loss au prix d'entrée quand le trade atteint +0.5%
    de profit non réalisé. Verrouille les gains et rend le trade
    "gratuit" (risk-free).
    """
    
    def __init__(self) -> None:
        """Initialise le système SLBE."""
        self.active: bool = False
        self.slbe_price: float = 0.0
        self.secured_count: int = 0
    
    def reset(self) -> None:
        """Réinitialise le système SLBE."""
        self.active = False
        self.slbe_price = 0.0
    
    def update(
        self,
        unrealized_pnl_pct: float,
        avg_entry: float,
        current_price: float,
        direction: int
    ) -> Tuple[bool, float]:
        """Met à jour le système SLBE et retourne les bonus.
        
        Args:
            unrealized_pnl_pct: PnL non réalisé en pourcentage.
            avg_entry: Prix d'entrée moyen.
            current_price: Prix actuel.
            direction: Direction du trade (1 ou -1).
            
        Returns:
            Tuple (slbe_just_activated, bonus_reward).
        """
        bonus = 0.0
        just_activated = False
        
        if unrealized_pnl_pct >= 0.005 and not self.active:
            self.active = True
            self.slbe_price = avg_entry
            self.secured_count += 1
            bonus += 6.0  # SLBE activation bonus
            just_activated = True
        
        return just_activated, bonus


class FTMOEnforcer:
    """Moteur des règles FTMO Challenge.
    
    Surveille en temps réel les limites de perte quotidienne,
    perte totale, objectif de profit et jours minimum de trading.
    """
    
    def __init__(self, initial_capital: float = 10000.0) -> None:
        """Initialise le moteur FTMO.
        
        Args:
            initial_capital: Capital initial du challenge.
        """
        self.initial_capital = initial_capital
        self.phase = 1
        self.phase_start_balance = initial_capital
        self.balance = initial_capital
        self.highest_balance = initial_capital
        self.daily_start_balance = initial_capital
        self.daily_pnl = 0.0
        self.trading_days: set = set()
        self.failed = False
        self.fail_reason = ""
        self.passed = False
    
    @property
    def max_daily_loss(self) -> float:
        """Perte quotidienne maximale autorisée (5% du solde de phase)."""
        return self.phase_start_balance * 0.05
    
    @property
    def max_total_loss(self) -> float:
        """Perte totale maximale autorisée (10% du solde de phase)."""
        return self.phase_start_balance * 0.10
    
    @property
    def profit_target(self) -> float:
        """Objectif de profit pour valider la phase (10%)."""
        return self.initial_capital * 0.10
    
    def new_day(self, day) -> None:
        """Réinitialise les compteurs quotidiens.
        
        Args:
            day: Nouveau jour de trading.
        """
        self.daily_start_balance = self.balance
        self.daily_pnl = 0.0
    
    def check_daily_loss(self) -> bool:
        """Vérifie si la limite de perte quotidienne est dépassée.
        
        Returns:
            True si la limite est dépassée.
        """
        if self.daily_pnl <= -self.max_daily_loss:
            self.fail_reason = (
                f"Limite quotidienne (${self.max_daily_loss:.0f}) "
                f"atteinte — PnL: ${self.daily_pnl:.2f}"
            )
            self.failed = True
            return True
        return False
    
    def check_total_loss(self) -> bool:
        """Vérifie si la limite de perte totale est dépassée.
        
        Returns:
            True si la limite est dépassée.
        """
        total_pnl = self.balance - self.phase_start_balance
        if total_pnl <= -self.max_total_loss:
            self.fail_reason = (
                f"Limite totale (${self.max_total_loss:.0f}) "
                f"atteinte — PnL: ${total_pnl:.2f}"
            )
            self.failed = True
            return True
        return False
    
    def check_profit_target(self) -> str:
        """Vérifie si l'objectif de profit est atteint.
        
        Returns:
            'passed' si le challenge est réussi,
            'phase1_passed' si la phase 1 est réussie,
            '' sinon.
        """
        total_pnl = self.balance - self.initial_capital
        
        if total_pnl >= self.profit_target:
            if self.phase == 1:
                self.phase = 2
                self.phase_start_balance = self.balance
                return "phase1_passed"
            elif self.phase == 2:
                self.passed = True
                return "passed"
        return ""


class TradingResult:
    """Résultat d'un épisode de trading.
    
    Attributes:
        equity_curve: Courbe d'equity sur l'épisode.
        total_return: Rendement total en pourcentage.
        trades: Liste des trades exécutés.
        ftmo_status: Statut FTMO final.
        metrics: Métriques de performance.
    """
    
    def __init__(
        self,
        equity_curve: Optional[np.ndarray] = None,
        total_return: float = 0.0,
        trades: Optional[List[Trade]] = None,
        ftmo_status: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Initialise le résultat de trading.
        
        Args:
            equity_curve: Courbe d'equity.
            total_return: Rendement total.
            trades: Liste des trades.
            ftmo_status: Statut FTMO.
            metrics: Métriques.
        """
        self.equity_curve = equity_curve or np.array([1.0])
        self.total_return = total_return
        self.trades = trades or []
        self.ftmo_status = ftmo_status or {}
        self.metrics = metrics or {}


class OctopusTradingEnv:
    """Environnement de trading XAUUSD pour le moteur Octopus.
    
    Interface Gymnasium-like avec 5 actions :
    0 = Hold, 1 = Buy, 2 = Sell, 3 = Split (50%), 4 = Close All
    
    Attributes:
        df: DataFrame de données de marché (M15/M1).
        lookback: Nombre de pas de temps d'observation.
        action_dim: Nombre d'actions possibles.
        ftmo: Moteur de règles FTMO.
        slbe: Système Stop Loss Break Even.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 96,
        initial_capital: float = 10000.0,
        max_position: float = 0.5,
        fee: float = 0.0001,
        macro_feature_engine: Optional[Any] = None,
    ) -> None:
        """Initialise l'environnement de trading.
        
        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume.
            lookback: Nombre de pas d'observation.
            initial_capital: Capital initial.
            max_position: Taille de position maximale en lots.
            fee: Frais de transaction (0.01%).
            macro_feature_engine: Moteur de features macro (optionnel).
        """
        self.df = df
        self.lookback = lookback
        self.max_position = max_position
        self.fee = fee
        self.action_dim = 5
        self.macro_engine = macro_feature_engine
        self.macro_features_history: List[np.ndarray] = []
        
        # Composants
        self.ftmo = FTMOEnforcer(initial_capital)
        self.slbe = SLBESystem()
        
        self.reset()
    
    @property
    def state_dim(self) -> int:
        """Dimension de l'état d'observation."""
        return self.lookback * len(self._get_features(0))
    
    def _get_features(self, idx: int) -> np.ndarray:
        """Extrait les features à un pas de temps donné.
        
        5 features techniques + 15 features macro (si disponibles) = 20 features.
        
        Args:
            idx: Index dans le DataFrame.
            
        Returns:
            Vecteur de 20 features.
        """
        row = self.df.iloc[idx]
        
        # 5 features techniques
        tech_features = np.array([
            row.get('Close', 0) / row.get('Open', 1) - 1,  # return
            (row.get('High', 0) - row.get('Low', 0)) / row.get('Close', 1e-8),  # spread
            float(idx % 96) / 96.0,  # position dans la journée
            1.0 if 8 <= idx % 24 < 17 else 0.0,  # session London
            1.0 if 13 <= idx % 24 < 22 else 0.0,  # session NY
        ], dtype=np.float32)
        
        # 15 features macro (si moteur disponible)
        if self.macro_engine is not None:
            macro_features = self.macro_engine.get_features_vector()
            self.macro_features_history.append(macro_features.copy())
            return np.concatenate([tech_features, macro_features])
        
        return tech_features
    
    def reset(self) -> np.ndarray:
        """Réinitialise l'environnement.
        
        Returns:
            Observation initiale.
        """
        self.idx = self.lookback
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.current_lots = 0.0
        self.equity_curve = [1.0]
        self.trades: List[Trade] = []
        self.steps_since_trade = 0
        
        self.ftmo = FTMOEnforcer(self.ftmo.initial_capital)
        self.slbe.reset()
        self.macro_features_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Construit l'observation courante.
        
        Returns:
            Vecteur d'observation (features × lookback).
        """
        start = max(0, self.idx - self.lookback)
        obs = []
        for i in range(start, self.idx):
            obs.append(self._get_features(i))
        return np.array(obs).flatten().astype(np.float32)
    
    def _get_position_size(self, atr: float) -> float:
        """Calcule la taille de position optimale.
        
        Args:
            atr: Average True Range courant.
            
        Returns:
            Taille de position en lots (0.01 min).
        """
        risk_amount = self.ftmo.balance * 0.008  # 0.8% risk per trade
        stop_distance = max(atr * 1.3, 5.0)
        risk_per_lot = stop_distance * 100
        lots = risk_amount / max(risk_per_lot, 1.0)
        return max(0.01, min(lots, self.max_position))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Exécute une action dans l'environnement.
        
        Args:
            action: Action à exécuter (0-4).
            
        Returns:
            Tuple (observation, reward, done, info).
        """
        price = float(self.df.iloc[self.idx]['Close'])
        atr = float(self.df.iloc[self.idx].get('atr', price * 0.01))
        
        reward = 0.0
        done = False
        
        # Mise à jour du PnL non réalisé
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = (
                (price - self.entry_price) * self.position * self.current_lots * 100
            )
        
        unrealized_pnl_pct = unrealized_pnl / max(self.ftmo.balance, 1.0)
        equity = self.ftmo.balance + unrealized_pnl
        
        # Mise à jour du SLBE
        if self.position != 0:
            slbe_activated, slbe_bonus = self.slbe.update(
                unrealized_pnl_pct, self.entry_price, price, self.position
            )
            reward += slbe_bonus
            
            # Vérifier si le SLBE est touché
            if self.slbe.active:
                if (self.position == 1 and price <= self.slbe.slbe_price) or \
                   (self.position == -1 and price >= self.slbe.slbe_price):
                    reward += 1.0  # Bonus protection
                    action = 4  # Forcer Close All
        
        # Exécution de l'action
        if action == 1:  # BUY
            if self.position <= 0:
                self.position = 1
                self.entry_price = price
                self.current_lots = self._get_position_size(atr)
                self.steps_since_trade = 0
                self.ftmo.trading_days.add(pd.Timestamp(self.df.index[self.idx]).date())
        
        elif action == 2:  # SELL
            if self.position >= 0:
                self.position = -1
                self.entry_price = price
                self.current_lots = self._get_position_size(atr)
                self.steps_since_trade = 0
                self.ftmo.trading_days.add(pd.Timestamp(self.df.index[self.idx]).date())
        
        elif action == 3:  # SPLIT (close 50%)
            if self.position != 0 and unrealized_pnl > 0:
                split_lots = self.current_lots * 0.5
                split_pnl = (price - self.entry_price) * self.position * split_lots * 100
                self.ftmo.balance += split_pnl
                self.current_lots *= 0.5
                reward += 10.0  # Split bonus
                self.steps_since_trade = 0
        
        elif action == 4:  # CLOSE ALL
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position * self.current_lots * 100
                self.ftmo.balance += pnl
                
                if pnl / max(self.ftmo.balance - pnl, 1.0) > 0.02:
                    reward += 15.0  # Big winner bonus
                elif pnl > 0:
                    reward += 10.0  # Quality trade bonus
                
                self.trades.append(Trade(
                    direction=self.position,
                    entry_price=self.entry_price,
                    entry_step=self.idx,
                    lots=self.current_lots,
                    exit_price=price,
                    exit_step=self.idx,
                    pnl=pnl
                ))
                
                self.position = 0
                self.current_lots = 0.0
                self.slbe.reset()
                self.steps_since_trade = 0
        
        # Pénalités
        if unrealized_pnl < 0:
            reward += unrealized_pnl_pct * 100  # Pénalité proportionnelle
        
        if unrealized_pnl_pct < -0.05:
            reward -= 10.0  # Max drawdown penalty
        
        # Pénalité d'inactivité
        self.steps_since_trade += 1
        if self.steps_since_trade > 100:
            reward -= 1.0
        
        # Mise à jour FTMO
        self.ftmo.daily_pnl = equity - self.ftmo.daily_start_balance
        self.ftmo.balance = self.ftmo.balance  # Balance cash
        self.equity_curve.append(equity / self.ftmo.initial_capital)
        
        self.idx += 1
        done = self.idx >= len(self.df) - 1
        
        # Bonus final de croissance
        if done:
            total_growth = (equity - self.ftmo.initial_capital) / self.ftmo.initial_capital
            if total_growth >= 0.10:
                reward += 50.0  # Final growth bonus
        
        return self._get_observation(), reward, done, {
            "balance": self.ftmo.balance,
            "equity": equity,
            "position": self.position,
            "trades": len(self.trades),
        }