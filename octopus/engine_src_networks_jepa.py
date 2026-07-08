"""Encodeur JEPA (Joint Embedding Predictive Architecture) pour séries temporelles.

Apprend des représentations latentes débruitées des données de marché via
l'objectif VICReg (Variance-Invariance-Covariance Regularization).
Basé sur TS-JEPA pour l'adaptation aux séries temporelles financières.

References:
    Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning." ICLR 2022.
    
    TS-JEPA. "Joint Embeddings Go Temporal." arXiv 2509.25449, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class JEPAEncoder(nn.Module):
    """Encodeur JEPA pour séries temporelles financières.
    
    Transforme une séquence de features de marché (OHLCV + indicateurs)
    en un embedding latent qui filtre le bruit et capture la structure
    fondamentale du marché.
    
    Attributes:
        input_dim: Nombre de features d'entrée par pas de temps.
        seq_len: Longueur de la séquence d'entrée.
        embedding_dim: Dimension de l'embedding latent de sortie.
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        seq_len: int = 96,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        """Initialise l'encodeur JEPA avec attention temporelle.
        
        Args:
            input_dim: Nombre de features par pas de temps.
            seq_len: Longueur de la séquence d'entrée (lookback).
            embedding_dim: Dimension de l'espace latent de sortie.
            hidden_dim: Dimension des couches cachées.
            num_heads: Nombre de têtes d'attention.
            dropout: Taux de dropout pour la régularisation.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        # Projection d'entrée avec encodage positionnel
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Encodeur temporel : Transformer encoder avec attention causale
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )
        
        # Pooling avec attention sur la séquence
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Projection finale vers l'espace latent
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode une séquence de features en embedding latent.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_len, input_dim).
            
        Returns:
            Embedding latent de forme (batch_size, embedding_dim).
        """
        # Projection et encoding positionnel
        x = self.input_projection(x)  # (B, S, hidden)
        x = self.positional_encoding(x)
        
        # Encodage temporel avec attention
        x = self.temporal_encoder(x)  # (B, S, hidden)
        
        # Pooling par attention : une requête apprise résume la séquence
        query = self.pool_query.expand(x.size(0), -1, -1)  # (B, 1, hidden)
        pooled, _ = self.attention_pool(query, x, x)
        
        # Projection finale
        embedding = self.output_projection(pooled.squeeze(1))
        
        return embedding


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal (Vaswani et al., 2017).
    
    Ajoute une information de position relative aux tokens
    de la séquence temporelle.
    
    Attributes:
        dropout: Taux de dropout.
        pe: Tenseur d'encodage positionnel.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000) -> None:
        """Initialise l'encodage positionnel.
        
        Args:
            d_model: Dimension du modèle.
            dropout: Taux de dropout.
            max_len: Longueur maximale de séquence supportée.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ajoute l'encodage positionnel à l'entrée.
        
        Args:
            x: Tenseur d'entrée (batch, seq_len, d_model).
            
        Returns:
            Tenseur avec encodage positionnel ajouté.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class JEPAPredictor(nn.Module):
    """Prédicteur JEPA : prédit un embedding cible depuis un embedding source.
    
    Dans l'architecture JEPA, le prédicteur apprend à transformer
    l'embedding d'une vue du marché en l'embedding d'une autre vue
    (ex: passé → futur, brut → augmenté).
    
    Attributes:
        hidden_dim: Dimension cachée du prédicteur.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128
    ) -> None:
        """Initialise le prédicteur JEPA.
        
        Args:
            embedding_dim: Dimension des embeddings d'entrée/sortie.
            hidden_dim: Dimension des couches cachées.
        """
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Prédit l'embedming cible depuis l'embedding source.
        
        Args:
            z: Embedding source (batch, embedding_dim).
            
        Returns:
            Embedding prédit (batch, embedding_dim).
        """
        return self.predictor(z)


class VICRegLoss(nn.Module):
    """Perte VICReg (Variance-Invariance-Covariance Regularization).
    
    Trois termes :
    1. Variance : maintient l'écart-type de chaque dimension > 1
       (empêche l'effondrement des représentations).
    2. Invariance : minimise l'erreur quadratique entre prédiction et cible.
    3. Covariance : décorrèle les dimensions de l'embedding
       (réduit la redondance d'information).
    
    Reference:
        Bardes, Ponce, LeCun. "VICReg." ICLR 2022.
    """
    
    def __init__(
        self,
        sim_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        eps: float = 1e-4
    ) -> None:
        """Initialise la perte VICReg.
        
        Args:
            sim_coeff: Coefficient du terme de similarité (invariance).
            var_coeff: Coefficient du terme de variance.
            cov_coeff: Coefficient du terme de covariance.
            eps: Petit epsilon pour la stabilité numérique.
        """
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps
    
    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Calcule la perte VICReg.
        
        Args:
            z_pred: Embeddings prédits (batch, embedding_dim).
            z_target: Embeddings cibles (batch, embedding_dim).
            
        Returns:
            Tuple (perte totale, dictionnaire des pertes individuelles).
        """
        batch_size = z_pred.size(0)
        
        # ─── Terme d'invariance (similarité) ──────────────
        sim_loss = F.mse_loss(z_pred, z_target)
        
        # ─── Terme de variance ────────────────────────────
        # Pousse l'écart-type de chaque dimension > 1
        std_pred = torch.sqrt(z_pred.var(dim=0) + self.eps)
        std_target = torch.sqrt(z_target.var(dim=0) + self.eps)
        
        var_loss = torch.mean(
            F.relu(1.0 - std_pred)
        ) + torch.mean(
            F.relu(1.0 - std_target)
        )
        
        # ─── Terme de covariance ──────────────────────────
        # Décorrèle les dimensions de l'embedding
        def off_diagonal_covariance(x: torch.Tensor) -> torch.Tensor:
            """Calcule la somme des éléments hors-diagonale de la covariance."""
            x = x - x.mean(dim=0, keepdim=True)
            cov = (x.T @ x) / (batch_size - 1)
            mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
            return (cov[mask] ** 2).sum() / cov.size(0)
        
        cov_loss = off_diagonal_covariance(z_pred) + \
                   off_diagonal_covariance(z_target)
        
        # ─── Perte totale ─────────────────────────────────
        total_loss = (
            self.sim_coeff * sim_loss +
            self.var_coeff * var_loss +
            self.cov_coeff * cov_loss
        )
        
        losses = {
            "vicreg_total": total_loss.item(),
            "vicreg_sim": sim_loss.item(),
            "vicreg_var": var_loss.item(),
            "vicreg_cov": cov_loss.item(),
        }
        
        return total_loss, losses


class TSJEPA(nn.Module):
    """Architecture TS-JEPA complète pour séries temporelles financières.
    
    Combine :
    - Un encodeur JEPA (online) avec target encoder (momentum)
    - Un prédicteur JEPA
    - La perte VICReg
    - Une cible mobile (momentum encoder) pour la stabilité
    
    Reference:
        TS-JEPA. "Joint Embeddings Go Temporal." arXiv 2509.25449, 2025.
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        seq_len: int = 96,
        embedding_dim: int = 64,
        momentum: float = 0.99
    ) -> None:
        """Initialise l'architecture TS-JEPA.
        
        Args:
            input_dim: Nombre de features d'entrée par pas de temps.
            seq_len: Longueur de la séquence d'entrée.
            embedding_dim: Dimension de l'espace latent.
            momentum: Taux de momentum pour la mise à jour du target encoder.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        
        # Encodeur online (appris par gradient)
        self.encoder = JEPAEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            embedding_dim=embedding_dim
        )
        
        # Encodeur cible (mis à jour par moyenne exponentielle)
        self.target_encoder = JEPAEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            embedding_dim=embedding_dim
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Désactiver les gradients du target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Prédicteur JEPA
        self.predictor = JEPAPredictor(
            embedding_dim=embedding_dim
        )
        
        # Perte VICReg
        self.vicreg_loss = VICRegLoss()
    
    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Met à jour l'encodeur cible par moyenne exponentielle.
        
        θ_target = momentum * θ_target + (1 - momentum) * θ_encoder
        """
        for param_e, param_t in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data = self.momentum * param_t.data + \
                          (1.0 - self.momentum) * param_e.data
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode une observation avec l'encodeur online.
        
        Args:
            x: Observation d'entrée (batch, seq_len, input_dim).
            
        Returns:
            Embedding latent (batch, embedding_dim).
        """
        return self.encoder(x)
    
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Encode une observation avec l'encodeur cible.
        
        Args:
            x: Observation d'entrée (batch, seq_len, input_dim).
            
        Returns:
            Embedding latent cible (batch, embedding_dim).
        """
        with torch.no_grad():
            return self.target_encoder(x)
    
    def train_step(
        self,
        x_online: torch.Tensor,
        x_target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Effectue un pas d'entraînement JEPA.
        
        Args:
            x_online: Vue source (batch, seq_len, input_dim).
            x_target: Vue cible (batch, seq_len, input_dim).
            
        Returns:
            Tuple (perte, dictionnaire des métriques).
        """
        # Encoder les deux vues
        z_online = self.encoder(x_online)
        z_target = self.encode_target(x_target)
        
        # Prédire l'embedding cible depuis l'embedding source
        z_pred = self.predictor(z_online)
        
        # Arrêter le gradient sur la cible
        z_target = z_target.detach()
        
        # Calculer la perte VICReg
        loss, metrics = self.vicreg_loss(z_pred, z_target)
        
        # Mettre à jour l'encodeur cible
        self.update_target_encoder()
        
        return loss, metrics
    
    def get_embedding(
        self,
        x: torch.Tensor,
        use_target: bool = False
    ) -> torch.Tensor:
        """Obtient l'embedding d'une observation.
        
        Args:
            x: Observation (batch, seq_len, input_dim).
            use_target: Si True, utilise l'encodeur cible.
            
        Returns:
            Embedding latent (batch, embedding_dim).
        """
        if use_target:
            return self.encode_target(x)
        return self.encoder(x)