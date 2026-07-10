"""Module Actor-Critic pour le moteur Octopus.

L'actor-critic est entraîné exclusivement sur les trajectoires
imaginaires produites par le World Model (pas d'interaction réelle
avec le marché pendant l'entraînement).

References:
    Hafner et al. "Dream to Control: Learning Behaviors by
    Latent Imagination." ICLR 2020.
    
    Schulman et al. "Proximal Policy Optimization Algorithms."
    arXiv 1707.06347, 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Transformation symlog pour stabilité numérique.
    
    Args:
        x: Tenseur d'entrée.
    
    Returns:
        Tenseur transformé.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


class ActorNetwork(nn.Module):
    """Réseau de politique (Actor) pour le trading.
    
    Prend un état latent en entrée et produit une distribution
    sur les actions (Hold/Buy/Sell/Split/Close).
    
    Attributes:
        action_dim: Nombre d'actions discrètes.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        hidden_dim: int = 256
    ) -> None:
        """Initialise le réseau de politique.
        
        Args:
            state_dim: Dimension de l'état latent (stoch + deter).
            action_dim: Nombre d'actions discrètes.
            hidden_dim: Dimension des couches cachées.
        """
        super().__init__()
        
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialisation : valeurs proches de zéro
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialise les poids du réseau.
        
        Args:
            module: Module PyTorch à initialiser.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule la distribution sur les actions.
        
        Args:
            state: État latent (batch, state_dim).
            
        Returns:
            Tuple (logits, probabilités) de forme (batch, action_dim).
        """
        logits = self.network(state)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
    
    def sample_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Échantillonne une action selon la politique.
        
        Args:
            state: État latent (batch, state_dim).
            deterministic: Si True, action déterministe (argmax).
            
        Returns:
            Tuple (action one_hot, log_prob).
        """
        logits, probs = self.forward(state)
        
        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
        
        action_onehot = F.one_hot(actions, num_classes=self.action_dim)
        log_prob = torch.log(probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        
        return action_onehot.float(), log_prob.squeeze(-1)


class CriticNetwork(nn.Module):
    """Réseau de valeur (Critic) pour le trading.
    
    Estime la valeur d'un état latent (retour actualisé attendu).
    Utilise symlog pour gérer les échelles variables.
    
    Attributes:
        state_dim: Dimension de l'état latent.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256
    ) -> None:
        """Initialise le réseau de valeur.
        
        Args:
            state_dim: Dimension de l'état latent.
            hidden_dim: Dimension des couches cachées.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialise les poids du réseau.
        
        Args:
            module: Module PyTorch à initialiser.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estime la valeur d'un état latent.
        
        Args:
            state: État latent (batch, state_dim).
            
        Returns:
            Valeur estimée (batch, 1) en espace symlog.
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """Module Actor-Critic complet entraîné sur des rêves.
    
    L'acteur propose des actions, le critique estime leur valeur.
    Entraîné exclusivement sur les trajectoires imaginaires
    du World Model (DreamerV3-style).
    
    References:
        Hafner et al. "Dream to Control." ICLR 2020.
        Schulman et al. "PPO." arXiv 1707.06347.
    """
    
    def __init__(
        self,
        stoch_size: int = 32,
        stoch_classes: int = 32,
        deter_size: int = 256,
        action_dim: int = 5,
        hidden_dim: int = 256,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        entropy_coeff: float = 0.05
    ) -> None:
        """Initialise le module Actor-Critic.
        
        Args:
            stoch_size: Nombre de catégories stochastiques.
            stoch_classes: Nombre de classes par catégorie.
            deter_size: Dimension de l'état déterministe.
            action_dim: Dimension de l'action.
            hidden_dim: Dimension cachée.
            gamma: Facteur d'actualisation (discount).
            lambda_: Paramètre λ pour TD(λ).
            entropy_coeff: Coefficient du bonus d'entropie.
        """
        super().__init__()
        
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff
        
        # Dimension de l'état complet (stoch + deter)
        state_dim = stoch_size * stoch_classes + deter_size
        
        # Réseaux
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim
        )
    
    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor
    ) -> torch.Tensor:
        """Calcule les λ-returns pour l'entraînement du critic.
        
        Args:
            rewards: Récompenses (batch, steps, 1).
            values: Valeurs estimées (batch, steps, 1).
            continues: Probabilités de continuation (batch, steps, 1).
            
        Returns:
            λ-returns (batch, steps, 1).
        """
        batch_size, steps = rewards.shape[:2]
        
        # Préparer les tenseurs
        rewards = rewards.squeeze(-1)  # (batch, steps)
        values = values.squeeze(-1)
        continues = continues.squeeze(-1)
        
        # Ajouter un pas final (valeur de bootstrap = 0)
        values = torch.cat([
            values,
            torch.zeros(batch_size, 1, device=values.device)
        ], dim=-1)
        
        # Calculer les λ-returns en partant de la fin
        returns = torch.zeros_like(rewards)
        g = 0.0
        
        for t in reversed(range(steps)):
            discount = self.gamma * continues[:, t]
            g = rewards[:, t] + discount * (
                (1 - self.lambda_) * values[:, t + 1] +
                self.lambda_ * g
            )
            returns[:, t] = g
        
        return returns.unsqueeze(-1)
    
    def compute_loss(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calcule la perte actor-critic sur une trajectoire imaginaire.
        
        Args:
            stoch: États stochastiques (batch, steps, stoch, classes).
            deter: États déterministes (batch, steps, deter).
            actions: Actions one-hot (batch, steps, action_dim).
            rewards: Récompenses (batch, steps, 1).
            continues: Continuations (batch, steps, 1).
            
        Returns:
            Tuple (perte totale, dictionnaire des métriques).
        """
        batch_size, steps = actions.shape[:2]
        
        # Construire les états (stoch + deter → vecteur plat)
        stoch_flat = stoch.reshape(batch_size, steps, -1)
        states = torch.cat([stoch_flat, deter], dim=-1)
        
        # Évaluer les actions avec l'actor (log_probs)
        _, new_probs = self.actor(states.reshape(-1, states.shape[-1]))
        new_probs = new_probs.reshape(batch_size, steps, -1)
        
        actions_idx = actions.argmax(dim=-1)
        new_log_probs = torch.log(
            new_probs.gather(-1, actions_idx.unsqueeze(-1)) + 1e-8
        ).squeeze(-1)
        
        # Entropie de la politique
        entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(-1)
        
        # Valeurs estimées par le critic
        values = self.critic(states.reshape(-1, states.shape[-1]))
        values = values.reshape(batch_size, steps, -1)
        
        # λ-returns (cibles du critic)
        lambda_returns = self._compute_lambda_returns(
            rewards, values, continues
        )
        
        # Avantage
        advantages = lambda_returns - values
        advantages = advantages.detach()
        
        # Perte actor : maximize log_prob × advantage + entropy
        actor_loss = -(new_log_probs * advantages.squeeze(-1)).mean()
        entropy_loss = self.entropy_coeff * entropy.mean()
        total_actor_loss = actor_loss - entropy_loss
        
        # Perte critic : MSE entre valeur et λ-return
        critic_loss = F.mse_loss(values, lambda_returns)
        
        # Perte totale
        total_loss = total_actor_loss + critic_loss
        
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "advantage": advantages.mean().item(),
            "lambda_return": lambda_returns.mean().item(),
            "value": values.mean().item(),
        }
        
        return total_loss, metrics
    
    def get_action(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Obtient une action depuis un état latent.
        
        Args:
            stoch: État stochastique (batch, stoch, classes).
            deter: État déterministe (batch, deter).
            deterministic: Si True, action déterministe.
            
        Returns:
            Action one-hot (batch, action_dim).
        """
        state = torch.cat([
            stoch.reshape(stoch.shape[0], -1), deter
        ], dim=-1)
        
        action, _ = self.actor.sample_action(state, deterministic)
        return action