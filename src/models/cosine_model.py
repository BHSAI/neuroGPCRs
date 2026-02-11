"""Cosine similarity-based model for DTI prediction."""

import torch
import torch.nn as nn


class Cosine(nn.Module):
    """Cosine similarity layer."""

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return nn.CosineSimilarity()(x1, x2)


class DTIProfilerCosine(nn.Module):
    """
    Dual-projection cosine similarity network for DTI prediction.

    Projects protein and molecule embeddings to a common space and computes
    cosine similarity as the binding prediction.

    Args:
        drug_dim: Dimensionality of drug/molecule embeddings
        target_dim: Dimensionality of target/protein embeddings
        latent_dim: Dimensionality of the shared latent space
        latent_activation: Activation function for projections
    """

    def __init__(
        self,
        drug_dim: int = 768,
        target_dim: int = 1024,
        latent_dim: int = 1024,
        latent_activation: nn.Module = nn.ReLU
    ):
        super(DTIProfilerCosine, self).__init__()

        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim

        # Projection layers
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_dim, latent_dim),
            latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(target_dim, latent_dim),
            latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        # Cosine similarity
        self.activator = Cosine()

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug: Drug/molecule embeddings (batch_size, drug_dim)
            target: Target/protein embeddings (batch_size, target_dim)

        Returns:
            Binding predictions (batch_size, 1)
        """
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        distance = self.activator(drug_projection, target_projection)
        return distance.unsqueeze(1)

    def get_projections(self, drug: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Get projected representations (useful for visualization).

        Args:
            drug: Drug/molecule embeddings
            target: Target/protein embeddings

        Returns:
            Tuple of (drug_projection, target_projection)
        """
        with torch.no_grad():
            drug_projection = self.drug_projector(drug)
            target_projection = self.target_projector(target)
        return drug_projection, target_projection
