"""MCTS (Monte Carlo Tree Search) pour le moteur Octopus.

Implémente la recherche arborescente utilisée par MuZero pour
planifier dans l'espace latent appris par le World Model.

References:
    Schrittwieser et al. "Mastering Atari, Go, chess and shogi
    by planning with a learned model." Nature, 2020.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """Nœud de l'arbre de recherche MCTS.
    
    Attributes:
        hidden_state: État latent associé à ce nœud.
        action: Action qui a mené à ce nœud.
        parent: Nœud parent.
        children: Dictionnaire des enfants {action: MCTSNode}.
        visit_count: Nombre de visites.
        total_value: Somme des valeurs backpropagées.
        prior: Probabilité a priori de l'action.
        reward: Récompense immédiate à la transition.
        expanded: Si le nœud a été expansé.
    """
    hidden_state: Optional[torch.Tensor] = None
    action: Optional[int] = None
    parent: Optional['MCTSNode'] = None
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    reward: float = 0.0
    expanded: bool = False
    
    @property
    def value(self) -> float:
        """Valeur moyenne du nœud."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(
        self,
        total_visits: int,
        pb_c_base: float = 19652.0,
        pb_c_init: float = 1.25
    ) -> float:
        """Score UCB (Upper Confidence Bound) pour la sélection.
        
        Args:
            total_visits: Nombre total de visites du parent.
            pb_c_base: Paramètre de base pour l'exploration.
            pb_c_init: Paramètre initial pour l'exploration.
            
        Returns:
            Score UCB.
        """
        pb_c = math.log(
            (total_visits + pb_c_base + 1) / pb_c_base
        ) + pb_c_init
        pb_c *= math.sqrt(total_visits) / (self.visit_count + 1)
        
        return self.value + pb_c * self.prior


class MCTS:
    """Monte Carlo Tree Search pour la planification dans l'espace latent.
    
    Utilise le World Model (Dynamics Network) pour simuler des futurs
    sans interagir avec le vrai marché.
    
    Attributes:
        action_dim: Nombre d'actions possibles.
        num_simulations: Nombre de simulations par décision.
        discount: Facteur d'actualisation.
        dirichlet_alpha: Paramètre α du bruit Dirichlet.
        exploration_fraction: Fraction de bruit à la racine.
    """
    
    def __init__(
        self,
        action_dim: int = 5,
        num_simulations: int = 150,
        discount: float = 0.99,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.50,
        pb_c_base: float = 19652.0,
        pb_c_init: float = 1.25
    ) -> None:
        """Initialise le MCTS.
        
        Args:
            action_dim: Nombre d'actions discrètes.
            num_simulations: Nombre de simulations par recherche.
            discount: Facteur d'actualisation.
            dirichlet_alpha: Paramètre du bruit Dirichlet.
            exploration_fraction: Fraction d'exploration à la racine.
            pb_c_base: Paramètre UCB de base.
            pb_c_init: Paramètre UCB initial.
        """
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
    
    def search(
        self,
        root_state: torch.Tensor,
        network: object,
        device: torch.device = None
    ) -> np.ndarray:
        """Effectue une recherche MCTS depuis l'état racine.
        
        Args:
            root_state: État latent initial (batch, state_dim).
            network: Réseau MuZero avec initial_inference et recurrent_inference.
            device: Device PyTorch.
            
        Returns:
            Distribution de visite des actions (action_dim,).
        """
        if device is None:
            device = torch.device('cpu')
        
        # Inférence initiale
        with torch.no_grad():
            hidden, policy_logits, value = network.initial_inference(
                root_state.to(device)
            )
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        
        # Créer la racine avec bruit Dirichlet
        root = MCTSNode(hidden_state=hidden)
        root.expanded = True
        
        # Ajouter le bruit Dirichlet pour l'exploration
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * self.action_dim
        )
        noisy_policy = (
            policy * (1 - self.exploration_fraction) +
            noise * self.exploration_fraction
        )
        
        # Expanser la racine
        for action in range(self.action_dim):
            child = MCTSNode(action=action, parent=root)
            child.prior = noisy_policy[action]
            root.children[action] = child
        
        # Simulations
        for _ in range(self.num_simulations):
            self._simulate(root, network, device)
        
        # Extraire la distribution de visite
        visit_counts = np.array([
            root.children[a].visit_count for a in range(self.action_dim)
        ], dtype=np.float32)
        
        # Normaliser
        visit_dist = visit_counts / visit_counts.sum()
        
        return visit_dist
    
    def _simulate(
        self,
        root: MCTSNode,
        network: object,
        device: torch.device
    ) -> None:
        """Effectue une simulation MCTS (select → expand → evaluate → backup).
        
        Args:
            root: Nœud racine.
            network: Réseau MuZero.
            device: Device PyTorch.
        """
        node = root
        
        # Phase 1: SELECT — descendre dans l'arbre
        path = [node]
        while node.expanded and node.children:
            total_visits = sum(
                c.visit_count for c in node.children.values()
            )
            action = max(
                node.children.keys(),
                key=lambda a: node.children[a].ucb_score(
                    total_visits, self.pb_c_base, self.pb_c_init
                )
            )
            node = node.children[action]
            path.append(node)
        
        # Phase 2: EXPAND — ajouter les enfants
        parent = path[-2] if len(path) >= 2 else root
        
        with torch.no_grad():
            action_onehot = torch.zeros(
                1, self.action_dim, device=device
            )
            action_onehot[0, node.action] = 1.0
            
            next_hidden, reward, policy_logits, value = \
                network.recurrent_inference(
                    parent.hidden_state, action_onehot
                )
            
            policy = torch.softmax(
                policy_logits, dim=-1
            ).cpu().numpy()[0]
            value = value.cpu().item()
            reward = reward.cpu().item()
        
        node.hidden_state = next_hidden
        node.reward = reward
        node.expanded = True
        
        # Créer les enfants
        for action in range(self.action_dim):
            child = MCTSNode(action=action, parent=node)
            child.prior = policy[action]
            node.children[action] = child
        
        # Phase 3 & 4: BACKUP — remonter les valeurs
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value
            value = n.reward + self.discount * value