"""Tampon de rejeu (Replay Buffer) pour l'entraînement du World Model.

Stocke les trajectoires d'expérience (observations, actions, récompenses)
et permet l'échantillonnage de batches pour l'entraînement.

References:
    Schrittwieser et al. "Mastering Atari, Go, chess and shogi
    by planning with a learned model." Nature, 2020.
    
    Mnih et al. "Human-level control through deep reinforcement
    learning." Nature, 2015.
"""

import numpy as np
import torch
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import os


@dataclass
class GameHistory:
    """Enregistre l'historique complet d'un épisode de trading.
    
    Attributes:
        observations: Liste des observations (chaque step).
        actions: Liste des actions prises.
        rewards: Liste des récompenses obtenues.
        policies: Distribution de politique à chaque step (MCTS).
        values: Valeur estimée à chaque step.
        continues: Indicateurs de fin d'épisode (0=done, 1=continue).
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    policies: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    continues: List[int] = field(default_factory=list)
    total_reward: float = 0.0
    
    def add_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        policy: np.ndarray,
        value: float,
        done: bool = False
    ) -> None:
        """Ajoute un pas dans l'historique.
        
        Args:
            obs: Observation de l'environnement.
            action: Action choisie.
            reward: Récompense obtenue.
            policy: Distribution de politique MCTS.
            value: Valeur estimée de l'état.
            done: Si l'épisode est terminé.
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.continues.append(0 if done else 1)
        self.total_reward += reward
    
    def __len__(self) -> int:
        """Retourne le nombre de pas dans l'épisode."""
        return len(self.actions)
    
    def to_dict(self) -> dict:
        """Convertit l'historique en dictionnaire sérialisable.
        
        Returns:
            Dictionnaire avec les données de l'épisode.
        """
        return {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'policies': self.policies,
            'values': self.values,
            'continues': self.continues,
            'total_reward': self.total_reward,
        }
    
    def save(self, path: str) -> None:
        """Sauvegarde l'historique au format pickle.
        
        Args:
            path: Chemin du fichier de sauvegarde.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, path: str) -> 'GameHistory':
        """Charge un historique depuis un fichier pickle.
        
        Args:
            path: Chemin du fichier de sauvegarde.
            
        Returns:
            Instance de GameHistory chargée.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        game = cls()
        game.observations = data['observations']
        game.actions = data['actions']
        game.rewards = data['rewards']
        game.policies = data['policies']
        game.values = data['values']
        game.continues = data['continues']
        game.total_reward = data['total_reward']
        return game


class ReplayBuffer:
    """Tampon de rejeu pour l'entraînement du World Model.
    
    Stocke une collection de GameHistory et échantillonne des
    mini-lots pour l'entraînement du réseau MuZero.
    
    Attributes:
        capacity: Capacité maximale en nombre d'épisodes.
        games: Liste des épisodes stockés.
        min_samples: Nombre minimum d'échantillons avant échantillonnage.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        min_samples: int = 128
    ) -> None:
        """Initialise le tampon de rejeu.
        
        Args:
            capacity: Nombre maximum d'épisodes stockés.
            min_samples: Nombre minimum d'échantillons avant échantillonnage.
        """
        self.capacity = capacity
        self.min_samples = min_samples
        self.games: List[GameHistory] = []
    
    def add_game(self, game: GameHistory) -> None:
        """Ajoute un épisode complet au tampon.
        
        Args:
            game: Épisode à ajouter.
        """
        self.games.append(game)
        if len(self.games) > self.capacity:
            self.games.pop(0)  # FIFO
    
    def add_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        policy: np.ndarray,
        value: float,
        done: bool = False
    ) -> None:
        """Ajoute un pas au dernier épisode du tampon.
        
        Si le dernier épisode est terminé, crée un nouvel épisode.
        
        Args:
            obs: Observation.
            action: Action.
            reward: Récompense.
            policy: Distribution de politique.
            value: Valeur.
            done: Si l'épisode est terminé.
        """
        if not self.games or self.games[-1].continues[-1] == 0:
            self.games.append(GameHistory())
        
        self.games[-1].add_step(obs, action, reward, policy, value, done)
    
    def sample_batch(
        self,
        batch_size: int = 128,
        n_steps: int = 5
    ) -> Tuple[torch.Tensor, ...]:
        """Échantillonne un mini-lot aléatoire pour l'entraînement.
        
        Args:
            batch_size: Taille du lot.
            n_steps: Nombre de pas de déroulement (unroll steps).
            
        Returns:
            Tuple de tenseurs (obs, actions, rewards, policies, values, continues).
        """
        # Sélectionner des épisodes aléatoires
        games = np.random.choice(
            self.games,
            size=batch_size,
            replace=len(self.games) < batch_size
        )
        
        # Pour chaque épisode, sélectionner un point de départ aléatoire
        batches = []
        for game in games:
            if len(game) <= n_steps:
                continue
            
            start = np.random.randint(0, len(game) - n_steps)
            
            batches.append({
                'obs': game.observations[start:start + n_steps],
                'action': game.actions[start:start + n_steps],
                'reward': game.rewards[start:start + n_steps],
                'policy': game.policies[start:start + n_steps],
                'value': game.values[start:start + n_steps],
                'continue': game.continues[start:start + n_steps],
            })
        
        # Empiler en tenseurs
        obs_batch = torch.FloatTensor(
            np.stack([b['obs'] for b in batches])
        )
        action_batch = torch.LongTensor(
            np.stack([b['action'] for b in batches])
        )
        reward_batch = torch.FloatTensor(
            np.stack([b['reward'] for b in batches])
        )
        policy_batch = torch.FloatTensor(
            np.stack([b['policy'] for b in batches])
        )
        value_batch = torch.FloatTensor(
            np.stack([b['value'] for b in batches])
        )
        continue_batch = torch.FloatTensor(
            np.stack([b['continue'] for b in batches])
        )
        
        return (
            obs_batch,
            action_batch,
            reward_batch,
            policy_batch,
            value_batch,
            continue_batch
        )
    
    def __len__(self) -> int:
        """Retourne le nombre total de pas dans le buffer."""
        return sum(len(game) for game in self.games)
    
    @property
    def num_games(self) -> int:
        """Retourne le nombre d'épisodes dans le buffer."""
        return len(self.games)
    
    def save(self, path: str) -> None:
        """Sauvegarde le tampon complet.
        
        Args:
            path: Chemin du fichier de sauvegarde.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.games, f)
    
    def load(self, path: str) -> None:
        """Charge le tampon depuis un fichier.
        
        Args:
            path: Chemin du fichier de sauvegarde.
        """
        with open(path, 'rb') as f:
            self.games = pickle.load(f)