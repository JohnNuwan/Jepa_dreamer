"""World Model RSSM (Recurrent State-Space Model) pour le moteur Octopus.

Implémente le modèle du monde inspiré de DreamerV3 avec :
- GRU récurrent pour l'état déterministe
- Variables latentes catégorielles (32×32 classes)
- KL balancing et symlog pour la stabilité

References:
    Hafner et al. "Mastering Diverse Domains through World Models."
    Nature, 2025.
    
    Hafner et al. "Dream to Control: Learning Behaviors by
    Latent Imagination." ICLR 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Transformation symlog : symlog(x) = sign(x) * ln(|x| + 1).
    
    Écrase les grandes valeurs tout en préservant le signe.
    Rend l'apprentissage robuste aux échelles variables.
    
    Args:
        x: Tenseur d'entrée.
    
    Returns:
        Tenseur transformé par symlog.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse de symlog : symexp(x) = sign(x) * (exp(|x|) - 1).
    
    Args:
        x: Tenseur en espace symlog.
        
    Returns:
        Tenseur reconstruit en espace original.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RSSMTransition(nn.Module):
    """Module de transition du RSSM.
    
    Prédit l'état latent suivant (stochastique + déterministe)
    à partir de l'état courant et de l'action.
    
    Attributes:
        stoch_size: Dimension de l'état stochastique (catégoriel).
        stoch_classes: Nombre de classes par variable.
        deter_size: Dimension de l'état déterministe (GRU).
    """
    
    def __init__(
        self,
        stoch_size: int = 32,
        stoch_classes: int = 32,
        deter_size: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 5,
        embedding_dim: int = 64
    ) -> None:
        """Initialise le module de transition RSSM.
        
        Args:
            stoch_size: Nombre de catégories pour la variable stochastique.
            stoch_classes: Nombre de classes par catégorie.
            deter_size: Dimension de l'état déterministe GRU.
            hidden_dim: Dimension des couches cachées.
            action_dim: Dimension de l'espace d'action.
            embedding_dim: Dimension de l'embedding d'observation.
        """
        super().__init__()
        
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.deter_size = deter_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Cellule GRU pour la récurrence déterministe
        self.gru = nn.GRUCell(
            input_size=deter_size,
            hidden_size=deter_size
        )
        
        # Projection de l'entrée (embedding + action + hidden)
        self.input_projection = nn.Linear(
            stoch_size * stoch_classes + action_dim + embedding_dim,
            hidden_dim
        )
        
        # Prédicteur a priori (sans observation future)
        self.prior_net = nn.Linear(deter_size, stoch_size * stoch_classes)
        
        # Encodeur a posteriori (avec observation future)
        self.posterior_net = nn.Linear(
            deter_size + embedding_dim,
            stoch_size * stoch_classes
        )
        
        # Prédicteur de l'embedding (pour reconstruction)
        self.embed_predictor = nn.Sequential(
            nn.Linear(stoch_size * stoch_classes + deter_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def _stoch_to_hidden(self, stoch: torch.Tensor) -> torch.Tensor:
        """Convertit la variable stochastique en vecteur continu.
        
        Args:
            stoch: Distribution stochastique (batch, stoch_size, stoch_classes).
            
        Returns:
            Vecteur continu (batch, stoch_size * stoch_classes).
        """
        batch_size = stoch.shape[0]
        return stoch.reshape(batch_size, -1)
    
    def _sample_stoch(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Échantillonne la variable stochastique avec straight-through.
        
        Args:
            logits: Logits de la distribution (batch, stoch_size * stoch_classes).
            
        Returns:
            Tuple (one_hot échantillonné, log_probs).
        """
        # Reshape pour la distribution catégorielle
        logits = logits.reshape(-1, self.stoch_size, self.stoch_classes)
        
        # Distribution catégorielle
        dist = torch.distributions.OneHotCategorical(logits=logits)
        
        # Échantillonnage avec straight-through gradients
        # Note: rsample() n'est pas supporté par OneHotCategorical
        # On utilise sample() avec le straight-through estimator manuel
        sample = dist.sample()  # one_hot
        # Straight-through : gradient = gradient de l'argmax
        # En utilisant le sample + (logits - logits.detach()) pour le gradient
        sample_soft = F.softmax(logits, dim=-1)
        sample = sample + (sample_soft - sample_soft.detach())
        
        return sample, dist
    
    def forward(
        self,
        prev_stoch: torch.Tensor,
        prev_action: torch.Tensor,
        prev_embedding: Optional[torch.Tensor] = None,
        next_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Effectue une transition : état actuel → état suivant.
        
        Args:
            prev_stoch: État stochastique précédent (batch, stoch, classes).
            prev_action: Action précédente (batch, action_dim), one-hot.
            prev_embedding: Embedding d'observation précédent (optionnel).
            next_embedding: Embedding d'observation suivante (optionnel).
            
        Returns:
            Tuple (posterior, prior, deter_next, log_dict):
                posterior: (batch, stoch_size, stoch_classes)
                prior: (batch, stoch_size, stoch_classes)
                deter_next: (batch, deter_size)
                d: Dictionnaire des loggers.
        """
        d = {}
        prev_stoch_flat = self._stoch_to_hidden(prev_stoch)
        
        # Embedding combiné : stochastique + embedding entrée
        if prev_embedding is not None:
            x = torch.cat([
                prev_stoch_flat,
                prev_action,
                prev_embedding
            ], dim=-1)
        else:
            # Utiliser un embedding nul de la bonne dimension
            batch_size = prev_stoch.shape[0]
            dummy_embedding = torch.zeros(
                batch_size, self.embedding_dim,
                device=prev_stoch.device
            )
            x = torch.cat([
                prev_stoch_flat,
                prev_action,
                dummy_embedding
            ], dim=-1)
        
        x = self.input_projection(x)
        
        # Mise à jour GRU (état déterministe)
        deter_next = self.gru(x)
        
        # Prédiction a priori (sans voir l'observation suivante)
        prior_logits = self.prior_net(deter_next)
        prior_sample, prior_dist = self._sample_stoch(prior_logits)
        
        # Prédiction a posteriori (avec l'observation suivante)
        if next_embedding is not None:
            post_input = torch.cat([deter_next, next_embedding], dim=-1)
            post_logits = self.posterior_net(post_input)
            post_sample, post_dist = self._sample_stoch(post_logits)
        else:
            post_sample = prior_sample
            post_dist = prior_dist
        
        return post_sample, prior_sample, deter_next, d


class RSSMWorldModel(nn.Module):
    """World Model RSSM complet — le cœur de l'architecture DreamerV3.
    
    Gère le séquencement des transitions et la prédiction
    des récompenses/continuations.
    
    Attributes:
        transition: Module de transition RSSM.
        reward_head: Prédicteur de récompense.
        continue_head: Prédicteur de fin d'épisode.
    """
    
    def __init__(
        self,
        stoch_size: int = 32,
        stoch_classes: int = 32,
        deter_size: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 5,
        embedding_dim: int = 64
    ) -> None:
        """Initialise le World Model RSSM.
        
        Args:
            stoch_size: Nombre de catégories stochastiques.
            stoch_classes: Nombre de classes par catégorie.
            deter_size: Dimension de l'état déterministe.
            hidden_dim: Dimension cachée.
            action_dim: Dimension de l'action.
            embedding_dim: Dimension de l'embedding d'observation.
        """
        super().__init__()
        
        self.transition = RSSMTransition(
            stoch_size=stoch_size,
            stoch_classes=stoch_classes,
            deter_size=deter_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim
        )
        
        # Prédicteur de récompense (avec symlog pour stabilité)
        self.reward_head = nn.Sequential(
            nn.Linear(stoch_size * stoch_classes + deter_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Prédicteur de continuation (fin d'épisode)
        self.continue_head = nn.Sequential(
            nn.Linear(stoch_size * stoch_classes + deter_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def predict_reward(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor
    ) -> torch.Tensor:
        """Prédit la récompense d'un état latent.
        
        Args:
            stoch: État stochastique (batch, stoch, classes).
            deter: État déterministe (batch, deter).
            
        Returns:
            Récompense prédite en espace symlog.
        """
        x = torch.cat([
            stoch.reshape(stoch.shape[0], -1), deter
        ], dim=-1)
        return self.reward_head(x)
    
    def predict_continue(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor
    ) -> torch.Tensor:
        """Prédit la probabilité de continuation de l'épisode.
        
        Args:
            stoch: État stochastique (batch, stoch, classes).
            deter: État déterministe (batch, deter).
            
        Returns:
            Probabilité de continuation (batch, 1).
        """
        x = torch.cat([
            stoch.reshape(stoch.shape[0], -1), deter
        ], dim=-1)
        return torch.sigmoid(self.continue_head(x))

    def initial_inference(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inférence initiale MuZero : état → hidden, policy_logits, value.

        Args:
            state: État latent (batch, stoch*classes + deter).

        Returns:
            Tuple (hidden_state, policy_logits, value).
        """
        batch = state.shape[0]
        # L'état latent sert directement de hidden state pour MCTS
        hidden = state
        # Policy logits uniformes (seront raffinés par l'actor-critic)
        policy_logits = torch.zeros(
            batch, 5, device=state.device
        )
        # Valeur initiale nulle
        value = torch.zeros(batch, 1, device=state.device)
        return hidden, policy_logits, value

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inférence récurrente MuZero : hidden + action → next, reward, policy, value.

        Args:
            hidden_state: État caché (batch, stoch*classes + deter).
            action: Action one-hot (batch, action_dim).

        Returns:
            Tuple (next_hidden, reward, policy_logits, value).
        """
        batch = hidden_state.shape[0]
        stoch_dim = self.transition.stoch_size * self.transition.stoch_classes

        # Reconstruire stoch et deter depuis l'état latent
        stoch_flat = hidden_state[:, :stoch_dim]
        deter = hidden_state[:, stoch_dim:]

        # Reconstruire la variable stochastique
        stoch = stoch_flat.reshape(
            batch, self.transition.stoch_size, self.transition.stoch_classes
        )

        # Effectuer la transition sans embedding
        post, prior, deter_next, _ = self.transition(
            prev_stoch=stoch,
            prev_action=action,
        )

        # Construire le prochain état latent
        next_hidden = torch.cat([
            post.reshape(batch, -1), deter_next
        ], dim=-1)

        # Prédire la récompense
        reward = self.predict_reward(post, deter_next)

        # Policy et valeur factices (seront raffinées par l'actor-critic)
        policy_logits = torch.zeros(batch, 5, device=hidden_state.device)
        value = torch.zeros(batch, 1, device=hidden_state.device)

        return next_hidden, reward, policy_logits, value

    def imagine(
        self,
        initial_stoch: torch.Tensor,
        initial_deter: torch.Tensor,
        policy: nn.Module,
        horizon: int = 15
    ) -> Dict[str, torch.Tensor]:
        """Déroule des trajectoires imaginaires dans l'espace latent.
        
        Utilise la politique pour choisir les actions et le World Model
        pour prédire les transitions. Aucune interaction avec le vrai marché.
        
        Args:
            initial_stoch: État stochastique initial.
            initial_deter: État déterministe initial.
            policy: Module de politique (actor) qui prend les décisions.
            horizon: Nombre de pas d'imagination.
            
        Returns:
            Dictionnaire contenant les trajectoires imaginaires :
                - 'stoch': Tenseur (batch, steps, stoch, classes)
                - 'deter': Tenseur (batch, steps, deter)
                - 'action': Tenseur (batch, steps, action_dim)
                - 'reward': Tenseur (batch, steps, 1)
            continue: Tenseur (batch, steps, 1)
        """
        batch_size = initial_stoch.shape[0]
        
        # Tensors de sortie
        stoch_traj = [initial_stoch]
        deter_traj = [initial_deter]
        action_traj = []
        reward_traj = []
        continue_traj = []
        
        stoch = initial_stoch
        deter = initial_deter
        
        for _ in range(horizon):
            # Choisir une action avec la politique
            state = torch.cat([
                stoch.reshape(batch_size, -1), deter
            ], dim=-1)
            action = policy(state)
            action_traj.append(action)
            
            # Prédire la transition dans le monde latent (sans embedding)
            stoch, _, deter, _ = self.transition(
                prev_stoch=stoch,
                prev_action=action
            )
            stoch_traj.append(stoch)
            deter_traj.append(deter)
            
            # Prédire récompense et continuation
            reward = self.predict_reward(stoch, deter)
            cont = self.predict_continue(stoch, deter)
            
            reward_traj.append(reward)
            continue_traj.append(cont)
        
        return {
            "stoch": torch.stack(stoch_traj[:-1], dim=1),
            "deter": torch.stack(deter_traj[:-1], dim=1),
            "action": torch.stack(action_traj, dim=1),
            "reward": torch.stack(reward_traj, dim=1),
            "continue": torch.stack(continue_traj, dim=1),
        }