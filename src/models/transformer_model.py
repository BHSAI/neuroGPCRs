"""Transformer encoder-based model for DTI prediction."""

import torch
import torch.nn as nn


class DTIProfilerTransformer(nn.Module):
    """
    Transformer encoder network for DTI prediction.

    Projects protein and molecule embeddings, concatenates them, and processes
    through a transformer encoder followed by an MLP classifier.

    Args:
        drug_dim: Dimensionality of drug/molecule embeddings
        target_dim: Dimensionality of target/protein embeddings
        latent_dim: Dimensionality of the shared latent space
        n_heads: Number of attention heads in transformer
        n_layers: Number of transformer encoder layers
        dropout: Dropout rate
        latent_activation: Activation function for projections
    """

    def __init__(
        self,
        drug_dim: int = 768,
        target_dim: int = 1024,
        latent_dim: int = 1024,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        latent_activation: nn.Module = nn.ReLU
    ):
        super(DTIProfilerTransformer, self).__init__()

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

        # Projection for concatenated features
        self.projection = nn.Linear(2 * latent_dim, latent_dim)

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug: Drug/molecule embeddings (batch_size, drug_dim)
            target: Target/protein embeddings (batch_size, target_dim)

        Returns:
            Binding predictions (batch_size, 1)
        """
        # Project inputs
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        # Concatenate and project
        combined_features = torch.cat((target_projection, drug_projection), dim=1)
        projected_features = self.projection(combined_features)

        # Add sequence dimension for transformer
        transformer_input = projected_features.unsqueeze(1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_output_squeezed = transformer_output.squeeze(1)

        # Classify
        logits = self.mlp(transformer_output_squeezed)

        return logits

    def get_embeddings(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Get transformer output embeddings (useful for analysis).

        Args:
            drug: Drug/molecule embeddings
            target: Target/protein embeddings

        Returns:
            Transformer output embeddings
        """
        with torch.no_grad():
            drug_projection = self.drug_projector(drug)
            target_projection = self.target_projector(target)
            combined_features = torch.cat((target_projection, drug_projection), dim=1)
            projected_features = self.projection(combined_features)
            transformer_input = projected_features.unsqueeze(1)
            transformer_output = self.transformer_encoder(transformer_input)
            return transformer_output.squeeze(1)
