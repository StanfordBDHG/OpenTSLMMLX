#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the OpenTSLMMLX open-source project.
#
# SPDX-License-Identifier: MIT
#

"""MLX port of OpenTSLM's TransformerCNNEncoder."""

import mlx.core as mx
import mlx.nn as nn


class TransformerEncoderLayer(nn.Module):
    """Matches PyTorch nn.TransformerEncoderLayer(norm_first=False, activation='gelu')."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.linear2(nn.gelu(self.linear1(x))))
        return x


class TransformerCNNEncoder(nn.Module):
    """MLX port of OpenTSLM's TransformerCNNEncoder.

    Takes raw time series [B, L] and produces patch embeddings [B, N, output_dim]
    where N = L // patch_size.
    """

    def __init__(
        self,
        output_dim: int = 128,
        transformer_input_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 4,
        ff_dim: int = 1024,
        max_patches: int = 2600,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Conv1d patch embedding
        # MLX Conv1d: input [B, L, C_in] → output [B, L//stride, C_out]
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # Learnable positional embeddings
        self.pos_embed = mx.zeros((1, max_patches, transformer_input_dim))

        # Input normalization
        self.input_norm = nn.LayerNorm(transformer_input_dim)

        # Transformer encoder layers
        self.layers = [
            TransformerEncoderLayer(transformer_input_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        B, L = x.shape

        # [B, L] → [B, L, 1] (channel-last for MLX Conv1d)
        x = x[:, :, None]

        # Conv1d → [B, N, transformer_input_dim]
        x = self.patch_embed(x)

        # Add positional embeddings
        N = x.shape[1]
        x = x + self.pos_embed[:, :N, :]

        # Layer norm
        x = self.input_norm(x)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x)

        return x
