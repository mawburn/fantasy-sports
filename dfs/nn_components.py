"""Reusable neural network components for DFS models.

This module provides common building blocks used across position-specific models
to reduce code duplication and improve maintainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class StandardBlock(nn.Module):
    """Standard neural network block with normalization, activation, and dropout.

    This pattern appears frequently across all position models:
    Linear -> Normalization -> Activation -> Dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalization: str = "layer",  # "layer", "batch", or "none"
        activation: str = "leaky_relu",  # "relu", "leaky_relu", "gelu", "tanh", "sigmoid"
        dropout: float = 0.2,
        negative_slope: float = 0.01,  # for leaky_relu
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        # Normalization layer
        self.norm = None
        if normalization == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif normalization == "batch":
            self.norm = nn.BatchNorm1d(out_features)

        # Activation function
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection for deeper networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_features = hidden_features or in_features

        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class AttentionBlock(nn.Module):
    """Self-attention mechanism for feature importance weighting.

    Used in RB and WR models to weight important features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 1
    ):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 2

        if num_heads > 1:
            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
        else:
            # Simple attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, feature_dim),
                nn.Softmax(dim=-1)
            )

        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads > 1:
            # Multi-head attention expects (batch, seq, feature)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            attended, _ = self.attention(x, x, x)
            return attended.squeeze(1) if attended.dim() == 3 else attended
        else:
            # Simple attention - element-wise multiplication
            attention_weights = self.attention(x)
            return x * attention_weights


class MultiHeadOutput(nn.Module):
    """Multi-head output for mean and uncertainty predictions.

    Common pattern in position models for predicting both point estimates
    and uncertainty/variance.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        predict_std: bool = True,
        predict_ceiling: bool = False,
        predict_floor: bool = False,
    ):
        super().__init__()

        # Mean prediction head (always present)
        self.mean_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1)
        )

        # Standard deviation head
        self.std_head = None
        if predict_std:
            self.std_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive std
            )

        # Ceiling prediction head
        self.ceiling_head = None
        if predict_ceiling:
            self.ceiling_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden_dim, 1)
            )

        # Floor prediction head
        self.floor_head = None
        if predict_floor:
            self.floor_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden_dim, 1)
            )

        self.predict_std = predict_std
        self.predict_ceiling = predict_ceiling
        self.predict_floor = predict_floor

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {"mean": self.mean_head(x).squeeze(-1)}

        if self.std_head is not None:
            outputs["std"] = self.std_head(x).squeeze(-1)

        if self.ceiling_head is not None:
            outputs["ceiling"] = self.ceiling_head(x).squeeze(-1)

        if self.floor_head is not None:
            outputs["floor"] = self.floor_head(x).squeeze(-1)

        return outputs


class PositionSpecificBranch(nn.Module):
    """Branch network for position-specific features.

    Used to create specialized processing branches for different aspects
    of a position (e.g., passing vs rushing for QBs).
    """

    def __init__(
        self,
        in_features: int,
        branch_sizes: list,
        dropout_rates: Optional[list] = None,
        activation: str = "gelu",
        normalization: str = "layer"
    ):
        super().__init__()

        if dropout_rates is None:
            dropout_rates = [0.2] * len(branch_sizes)

        layers = []
        current_size = in_features

        for i, (out_size, dropout) in enumerate(zip(branch_sizes, dropout_rates)):
            # Don't add activation/dropout to last layer
            is_last = i == len(branch_sizes) - 1

            layers.append(
                StandardBlock(
                    current_size,
                    out_size,
                    normalization=normalization if not is_last else "none",
                    activation=activation if not is_last else None,
                    dropout=dropout if not is_last else 0
                )
            )
            current_size = out_size

        self.branch = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class FeatureExtractor(nn.Module):
    """Common feature extraction backbone used across position models.

    Provides a consistent way to extract features with optional residual connections.
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: list,
        dropout_rates: Optional[list] = None,
        use_residual: bool = False,
        normalization: str = "layer",
        activation: str = "leaky_relu"
    ):
        super().__init__()

        if dropout_rates is None:
            dropout_rates = [0.25, 0.2, 0.15][:len(layer_sizes)]

        self.use_residual = use_residual
        self.input_norm = nn.LayerNorm(input_size) if normalization == "layer" else None

        layers = []
        current_size = input_size

        for size, dropout in zip(layer_sizes, dropout_rates):
            if use_residual and current_size == size:
                layers.append(ResidualBlock(size, dropout=dropout))
            else:
                layers.append(
                    StandardBlock(
                        current_size, size,
                        normalization=normalization,
                        activation=activation,
                        dropout=dropout
                    )
                )
            current_size = size

        self.feature_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        return self.feature_layers(x)


class PositionEmbedding(nn.Module):
    """Embedding layer for position-specific information.

    Can be used to embed categorical features like team, opponent, or week.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        combine_method: str = "add"  # "add", "concat", or "multiply"
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.combine_method = combine_method

    def forward(
        self,
        features: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(indices)

        if self.combine_method == "concat":
            return torch.cat([features, embedded], dim=-1)
        elif self.combine_method == "multiply":
            # Need to match dimensions
            if features.shape[-1] != embedded.shape[-1]:
                raise ValueError(
                    f"Feature dim {features.shape[-1]} != embedding dim {embedded.shape[-1]}"
                )
            return features * embedded
        else:  # add
            if features.shape[-1] != embedded.shape[-1]:
                raise ValueError(
                    f"Feature dim {features.shape[-1]} != embedding dim {embedded.shape[-1]}"
                )
            return features + embedded


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining multiple feature branches.

    Used to intelligently combine outputs from different processing branches.
    """

    def __init__(self, *branch_dims: int):
        super().__init__()

        total_dim = sum(branch_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, len(branch_dims)),
            nn.Softmax(dim=-1)
        )

        # Project branches to same dimension if needed
        self.projections = nn.ModuleList()
        output_dim = max(branch_dims)
        for dim in branch_dims:
            if dim != output_dim:
                self.projections.append(nn.Linear(dim, output_dim))
            else:
                self.projections.append(nn.Identity())

        self.output_dim = output_dim

    def forward(self, *branches: torch.Tensor) -> torch.Tensor:
        # Concatenate all branches
        concatenated = torch.cat(branches, dim=-1)

        # Compute gating weights
        gate_weights = self.gate(concatenated)

        # Project branches to same dimension and apply gating
        output = None
        for i, (branch, proj) in enumerate(zip(branches, self.projections)):
            projected = proj(branch)
            weighted = projected * gate_weights[:, i:i+1]
            output = weighted if output is None else output + weighted

        return output
