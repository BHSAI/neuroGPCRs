"""Feature extractor for XGBoost DTI model."""

import torch
import torch.nn as nn


class DTIFeatureExtractor(nn.Module):
    """
    Dual-projection feature extractor for XGBoost DTI prediction.

    Projects protein and molecule embeddings to a shared latent dimension
    and concatenates them to produce a combined feature vector.  The
    projector is randomly initialised (Xavier) and kept frozen; all
    learning happens inside XGBoost.

    Args:
        drug_dim: Dimensionality of drug/molecule embeddings
        target_dim: Dimensionality of target/protein embeddings
        latent_dim: Shared projection dimension
        latent_activation: Activation function for projections
    """

    def __init__(
        self,
        drug_dim: int = 768,
        target_dim: int = 1024,
        latent_dim: int = 1024,
        latent_activation: nn.Module = nn.ReLU
    ):
        super(DTIFeatureExtractor, self).__init__()

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

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Extract concatenated projected features.

        Args:
            drug: Drug/molecule embeddings (batch_size, drug_dim)
            target: Target/protein embeddings (batch_size, target_dim)

        Returns:
            Concatenated features (batch_size, 2 * latent_dim)
        """
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        return torch.cat([drug_projection, target_projection], dim=1)
