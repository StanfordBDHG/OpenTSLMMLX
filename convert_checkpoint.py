#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the OpenTSLMMLX open-source project.
#
# SPDX-License-Identifier: MIT
#

"""One-time conversion from OpenTSLM PyTorch checkpoints to safetensors.

Usage:
    python convert_checkpoint.py \
        --input checkpoints/model_checkpoint.pt \
        --output-prefix checkpoints/model_checkpoint

Outputs:
    checkpoints/model_checkpoint.encoder.safetensors
    checkpoints/model_checkpoint.projector.safetensors
    checkpoints/model_checkpoint.lora.safetensors
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def _convert_encoder_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Convert PyTorch TransformerCNNEncoder state_dict to MLX key/shape format."""
    weights: dict[str, np.ndarray] = {}

    # Conv1d: PyTorch [C_out, C_in, K] -> MLX [C_out, K, C_in]
    weights["patch_embed.weight"] = state_dict["patch_embed.weight"].float().cpu().numpy().transpose(0, 2, 1)
    weights["pos_embed"] = state_dict["pos_embed"].float().cpu().numpy()
    weights["input_norm.weight"] = state_dict["input_norm.weight"].float().cpu().numpy()
    weights["input_norm.bias"] = state_dict["input_norm.bias"].float().cpu().numpy()

    num_layers = sum(
        1 for k in state_dict if k.startswith("encoder.layers.") and k.endswith(".norm1.weight")
    )
    for i in range(num_layers):
        pt_pfx = f"encoder.layers.{i}"
        mlx_pfx = f"layers.{i}"

        # Split fused QKV into separate Q, K, V projections.
        in_proj_w = state_dict[f"{pt_pfx}.self_attn.in_proj_weight"].float().cpu().numpy()
        in_proj_b = state_dict[f"{pt_pfx}.self_attn.in_proj_bias"].float().cpu().numpy()
        d = in_proj_w.shape[1]

        weights[f"{mlx_pfx}.self_attn.query_proj.weight"] = in_proj_w[:d]
        weights[f"{mlx_pfx}.self_attn.key_proj.weight"] = in_proj_w[d : 2 * d]
        weights[f"{mlx_pfx}.self_attn.value_proj.weight"] = in_proj_w[2 * d :]
        weights[f"{mlx_pfx}.self_attn.query_proj.bias"] = in_proj_b[:d]
        weights[f"{mlx_pfx}.self_attn.key_proj.bias"] = in_proj_b[d : 2 * d]
        weights[f"{mlx_pfx}.self_attn.value_proj.bias"] = in_proj_b[2 * d :]

        weights[f"{mlx_pfx}.self_attn.out_proj.weight"] = (
            state_dict[f"{pt_pfx}.self_attn.out_proj.weight"].float().cpu().numpy()
        )
        weights[f"{mlx_pfx}.self_attn.out_proj.bias"] = (
            state_dict[f"{pt_pfx}.self_attn.out_proj.bias"].float().cpu().numpy()
        )

        for name in ["linear1", "linear2"]:
            for param in ["weight", "bias"]:
                weights[f"{mlx_pfx}.{name}.{param}"] = (
                    state_dict[f"{pt_pfx}.{name}.{param}"].float().cpu().numpy()
                )

        for name in ["norm1", "norm2"]:
            for param in ["weight", "bias"]:
                weights[f"{mlx_pfx}.{name}.{param}"] = (
                    state_dict[f"{pt_pfx}.{name}.{param}"].float().cpu().numpy()
                )

    return weights


def _convert_projector_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Convert PyTorch MLPProjector state_dict to MLX key format."""
    return {
        "norm.weight": state_dict["projector.0.weight"].float().cpu().numpy(),
        "norm.bias": state_dict["projector.0.bias"].float().cpu().numpy(),
        "linear.weight": state_dict["projector.1.weight"].float().cpu().numpy(),
        "linear.bias": state_dict["projector.1.bias"].float().cpu().numpy(),
    }


def _convert_lora_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Convert PEFT LoRA keys/shapes to MLX LoRA keys/shapes.

    PEFT stores:
      - lora_A as [r, in]  -> MLX expects [in, r]
      - lora_B as [out, r] -> MLX expects [r, out]
    """
    weights: dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        m = re.match(r"base_model\.model\.(.+)\.(lora_[AB])\.default\.weight", key)
        if not m:
            continue
        mlx_key = f"{m.group(1)}.{m.group(2).lower()}"
        weights[mlx_key] = tensor.float().cpu().numpy().T
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .pt checkpoint")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix (writes .encoder/.projector/.lora.safetensors)",
    )
    args = parser.parse_args()

    import torch

    ckpt_path = Path(args.input)
    out_prefix = Path(args.output_prefix)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "encoder_state" not in checkpoint or "projector_state" not in checkpoint:
        raise ValueError("Checkpoint is missing 'encoder_state' or 'projector_state'")

    encoder_weights = _convert_encoder_weights(checkpoint["encoder_state"])
    projector_weights = _convert_projector_weights(checkpoint["projector_state"])
    lora_weights = _convert_lora_weights(checkpoint.get("lora_state", {}))

    encoder_path = str(out_prefix) + ".encoder.safetensors"
    projector_path = str(out_prefix) + ".projector.safetensors"
    lora_path = str(out_prefix) + ".lora.safetensors"

    save_file(encoder_weights, encoder_path)
    save_file(projector_weights, projector_path)
    if lora_weights:
        save_file(lora_weights, lora_path)

    print(f"Wrote {encoder_path} ({len(encoder_weights)} tensors)")
    print(f"Wrote {projector_path} ({len(projector_weights)} tensors)")
    if lora_weights:
        print(f"Wrote {lora_path} ({len(lora_weights)} tensors)")
    else:
        print("No LoRA tensors found; skipped .lora.safetensors output")


if __name__ == "__main__":
    main()
