"""MLX port of OpenTSLM's MLPProjector."""

import mlx.core as mx
import mlx.nn as nn


class MLPProjector(nn.Module):
    """Projects encoder output [B, N, input_dim] to LLM hidden size [B, N, output_dim]."""

    def __init__(self, input_dim: int = 128, output_dim: int = 2048):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.linear(self.norm(x)))
