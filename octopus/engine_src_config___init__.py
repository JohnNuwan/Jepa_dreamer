"""Configuration du moteur Octopus.

Définit l'ensemble des hyperparamètres pour l'architecture JEPA + World Model
appliquée au trading XAUUSD sur timeframe M15/M1.

Attributes:
    OBSERVATION_SHAPE: Nombre de features d'entrée (OHLCV + indicateurs + contexte).
    ACTION_SPACE_SIZE: Nombre d'actions discrètes (Hold/Buy/Sell/Split/Close).
    HIDDEN_STATE_SIZE: Dimension de l'état latent du World Model.
    NUM_SIMULATIONS: Nombre de simulations MCTS par décision.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class JEPAConfig:
    """Configuration du module JEPA (Joint Embedding Predictive Architecture).
    
    Contrôle l'encodeur qui transforme les observations brutes en embeddings
    débruités via l'objectif VICReg (Variance-Invariance-Covariance Regularization).
    
    Reference:
        Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance
        Regularization for Self-Supervised Learning." ICLR 2022.
    """

    # Dimensions des réseaux
    embedding_dim: int = 64
    """Dimension de l'espace latent de sortie."""
    
    hidden_dim: int = 256
    """Dimension des couches cachées de l'encodeur."""
    
    predictor_hidden_dim: int = 128
    """Dimension du prédicteur d'embedding (JEPA predictor)."""
    
    # Paramètres VICReg
    vicreg_sim_coeff: float = 25.0
    """Coefficient de similarité (invariance) — MSE entre prédiction et cible."""
    
    vicreg_var_coeff: float = 25.0
    """Coefficient de variance — pousse std(embedding) > 1."""
    
    vicreg_cov_coeff: float = 1.0
    """Coefficient de covariance — décorrèle les dimensions de l'embedding."""
    
    # Momentum pour target encoder (copie retardée des poids)
    target_momentum: float = 0.99
    """Taux de mise à jour du target encoder par moyenne exponentielle."""


@dataclass
class WorldModelConfig:
    """Configuration du RSSM (Recurrent State-Space Model).
    
    Définit l'architecture du modèle du monde qui prédit les transitions
    d'états dans l'espace latent. Inspiré de DreamerV3.
    
    Reference:
        Hafner et al. "Mastering Diverse Domains through World Models."
        Nature, 2025.
    """

    # Dimensions des états latents
    stoch_size: int = 32
    """Nombre de classes catégorielles pour la variable stochastique."""
    
    stoch_classes: int = 32
    """Nombre de catégories par variable stochastique (32×32 = 1024 combinaisons)."""
    
    deter_size: int = 256
    """Dimension de l'état déterministe (sortie du GRU récurrent)."""
    
    hidden_dim: int = 256
    """Dimension des couches cachées du RSSM."""
    
    # Paramètres d'entraînement
    kl_balance: float = 0.8
    """Facteur d'équilibrage KL : si > 0.5, prior apprend plus vite que posterior."""
    
    kl_free_nats: float = 0.1
    """Seuil en dessous duquel la perte KL n'est pas appliquée (évite collapse)."""
    
    # Horizon d'imagination
    imagine_horizon: int = 15
    """Nombre de pas d'imagination pour l'entraînement de l'actor-critic."""


@dataclass
class ActorCriticConfig:
    """Configuration de l'actor-critic entraîné sur les rêves du World Model.
    
    L'actor propose des actions, le critic estime la valeur des états.
    Entraîné exclusivement sur des trajectoires imaginaires (latent space).
    """

    hidden_dim: int = 256
    """Dimension des couches cachées de l'actor et du critic."""
    
    gamma: float = 0.997
    """Facteur d'actualisation (discount factor)."""
    
    lambda_: float = 0.95
    """Paramètre λ pour le calcul du λ-return (TD(λ))."""
    
    entropy_coeff: float = 0.001
    """Coefficient du bonus d'entropie pour encourager l'exploration."""
    
    learning_rate: float = 3e-5
    """Taux d'apprentissage pour l'actor-critic."""


@dataclass
class FTMOConfig:
    """Configuration des règles FTMO Challenge.
    
    Définit les limites de risque et les objectifs de profit
    pour le challenge FTMO 10K. Appliqué comme couche de sécurité
    au-dessus des décisions de l'agent.
    """

    initial_capital: float = 10000.0
    """Capital initial du challenge FTMO ($)."""
    
    profit_target_pct: float = 0.10
    """Objectif de profit pour valider une phase (10%)."""
    
    max_daily_loss_pct: float = 0.05
    """Perte quotidienne maximale (5% du capital de phase)."""
    
    max_total_loss_pct: float = 0.10
    """Perte totale maximale sur la phase (10%)."""
    
    min_trading_days: int = 4
    """Nombre minimum de jours de trading requis."""
    
    risk_per_trade: float = 0.008
    """Risque maximum par trade (0.8% du capital)."""


@dataclass
class MuZeroConfig:
    """Configuration globale du moteur Octopus.
    
    Rassemble toutes les sous-configurations et les paramètres
    d'entraînement du système complet.
    """

    # Espaces
    observation_shape: Tuple[int, ...] = (20, 96)
    """(20 features, 96 lookback) — 5 techniques + 15 macro-économiques."""
    
    action_space_size: int = 5
    """Actions : 0=Hold, 1=Buy, 2=Sell, 3=Split, 4=Close."""
    
    # Symboles tradés
    symbols: List[str] = field(
        default_factory=lambda: ["XAUUSD"]
    )
    """Instruments financiers tradés par l'agent."""
    
    # Sous-configurations
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    """Configuration du module JEPA."""
    
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    """Configuration du RSSM World Model."""
    
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    """Configuration de l'actor-critic."""
    
    ftmo: FTMOConfig = field(default_factory=FTMOConfig)
    """Configuration des règles FTMO."""
    
    # Entraînement
    training_steps: int = 30000
    """Nombre total de pas d'entraînement."""
    
    batch_size: int = 128
    """Taille des mini-lots pour l'entraînement."""
    
    replay_buffer_size: int = 100000
    """Capacité du tampon de rejeu (replay buffer)."""
    
    checkpoint_interval: int = 100
    """Intervalle de sauvegarde des checkpoints (en pas)."""
    
    learning_rate: float = 5e-5
    """Taux d'apprentissage global."""
    
    # MCTS
    num_simulations: int = 150
    """Nombre de simulations MCTS par décision."""
    
    discount: float = 0.99
    """Facteur d'actualisation pour MCTS."""
    
    root_dirichlet_alpha: float = 0.3
    """Paramètre α du bruit Dirichlet pour l'exploration à la racine."""
    
    root_exploration_fraction: float = 0.50
    """Fraction de bruit Dirichlet mélangée à la politique à la racine."""
    
    # Anti-overtrading
    cooldown_seconds: int = 1800
    """Temps de pause minimum entre deux trades (30 minutes)."""
    
    inactivity_threshold: int = 60
    """Nombre de pas avant d'activer le mécanisme KICK."""