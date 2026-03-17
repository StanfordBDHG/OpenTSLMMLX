"""MLX OpenTSLMSP: end-to-end time-series language model."""

import re

import numpy as np
import mlx.core as mx
import mlx.utils
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

from ts_encoder import TransformerCNNEncoder
from ts_projector import MLPProjector


class OpenTSLMSP:
    """MLX port of OpenTSLM's SP (Soft Prompt) variant.

    Encodes time series with TransformerCNNEncoder, projects to LLM hidden size,
    interleaves with text embeddings, and generates text via Llama.
    """

    def __init__(self, llm_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        self.llm, self.tokenizer = load(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        hidden_size = self.llm.args.hidden_size
        self.encoder = TransformerCNNEncoder()
        self.projector = MLPProjector(128, hidden_size)
        self.patch_size = 4

    def load_from_file(self, path: str):
        """Load trained encoder/projector/LoRA weights from a PyTorch checkpoint.

        TODO: Remove torch dependency by pre-converting checkpoints to safetensors.
          Write a conversion script that applies all weight transforms.
        """
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "encoder_state" not in checkpoint or "projector_state" not in checkpoint:
            raise ValueError("Checkpoint is missing 'encoder_state' or 'projector_state'")

        max_patches = checkpoint["encoder_state"]["pos_embed"].shape[1]
        self.encoder = TransformerCNNEncoder(max_patches=max_patches)
        encoder_weights = _convert_encoder_weights(checkpoint["encoder_state"])
        self.encoder.load_weights(list(encoder_weights.items()))

        projector_weights = _convert_projector_weights(checkpoint["projector_state"])
        self.projector.load_weights(list(projector_weights.items()))

        if checkpoint.get("lora_state"):
            self._apply_lora(checkpoint["lora_state"])

        print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")

    def _apply_lora(
        self,
        lora_state: dict,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        """Load LoRA adapter weights into the LLM using LoRALinear layers (MLX).
        """
        scale = lora_alpha / lora_r

        num_layers = len(self.llm.model.layers)
        linear_to_lora_layers(
            self.llm.model,
            num_layers,
            {"rank": lora_r, "scale": scale, "dropout": 0.0},
        )

        # TODO: Remove this conversion once checkpoints are pre-converted to safetensors.
        lora_weights = []
        for key, tensor in lora_state.items():
            m = re.match(r"base_model\.model\.(.+)\.(lora_[AB])\.default\.weight", key)
            if not m:
                continue

            # PEFT lora_A [r, in] → MLX lora_a [in, r]
            # PEFT lora_B [out, r] → MLX lora_b [r, out]
            mlx_key = f"{m.group(1)}.{m.group(2).lower()}"
            lora_weights.append((mlx_key, mx.array(tensor.float().numpy()).T))

        self.llm.load_weights(lora_weights, strict=False)
        print(f"  Applied LoRA: {len(lora_weights) // 2} modules, rank={lora_r}, alpha={lora_alpha}")

    def pad_and_apply_batch(
        self,
        batch: list[dict],
    ) -> tuple[mx.array, mx.array]:
        """Embed and interleave all text and time-series inputs for a batch.

        This mirrors the original OpenTSLMSP.pad_and_apply_batch:
          1. Gather all text segments (pre_prompt, time_series_text, post_prompt)
          2. Tokenize & embed all texts in one batch
          3. Batch encode & project all time series
          4. Interleave per sample: [pre_prompt, ts_text_1, ts_1, ..., post_prompt]
          5. Pad all sequences to uniform length

        Args:
            batch: list of dicts with keys:
                pre_prompt, time_series_text, time_series, post_prompt

        Returns:
            inputs_embeds:  [B, L_max, H]
            attention_mask: [B, L_max]
        """
        H = self.llm.args.hidden_size

        # 1) Gather all texts
        all_texts = []
        text_ptrs = []
        ts_counts = []
        for sample in batch:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            text_ptrs.append((start, len(all_texts)))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) Tokenize & embed all texts
        tok = self.tokenizer._tokenizer(
            all_texts, return_tensors="np", padding=True, truncation=True
        )
        input_ids = mx.array(tok.input_ids)
        attn_mask = tok.attention_mask  # keep as numpy for .sum() / actually padding mask
        text_embeds = self.llm.model.embed_tokens(input_ids)

        # 3) Batch encode & project time series
        ts_list = []
        for sample in batch:
            for ts in sample["time_series"]:
                ts_np = np.array(ts, dtype=np.float32)
                # Original uses pad_sequence which needs [T, 1], so it unsqueezes 1D -> 2D.
                # We pad manually into a numpy array as 1D directly, so we do the
                # reverse: squeeze 2D [T, 1] down to [T]. Same end result.
                if ts_np.ndim == 2:
                    ts_np = ts_np.squeeze(-1)
                ts_list.append(ts_np)

        if ts_list:
            max_len = max(len(t) for t in ts_list)
            rem = max_len % self.patch_size
            if rem:
                max_len += self.patch_size - rem

            ts_padded = np.zeros((len(ts_list), max_len), dtype=np.float32)
            for i, t in enumerate(ts_list):
                ts_padded[i, : len(t)] = t

            ts_enc = self.encoder(mx.array(ts_padded))
            ts_proj = self.projector(ts_enc)
        else:
            ts_proj = mx.zeros((0, 0, H))

        # 4) Re-assemble per sample
        all_seq_embeds, all_seq_masks = [], []
        ts_offset = 0

        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]
            sample_masks = attn_mask[start:end]
            seq_embeds, seq_masks = [], []

            # pre_prompt
            length = int(sample_masks[0].sum())
            seq_embeds.append(sample_embeds[0, :length, :])
            seq_masks.append(mx.ones((length,)))

            # each (text_i, ts_i) pair
            for i in range(n_ts):
                idx = 1 + i
                length = int(sample_masks[idx].sum())
                seq_embeds.append(sample_embeds[idx, :length, :])
                seq_masks.append(mx.ones((length,)))

                proj = ts_proj[ts_offset + i]
                seq_embeds.append(proj)
                seq_masks.append(mx.ones((proj.shape[0],)))

            ts_offset += n_ts

            # post_prompt
            length = int(sample_masks[-1].sum())
            seq_embeds.append(sample_embeds[-1, :length, :])
            seq_masks.append(mx.ones((length,)))

            all_seq_embeds.append(mx.concatenate(seq_embeds, axis=0))
            all_seq_masks.append(mx.concatenate(seq_masks, axis=0))

        # 5) Batch-pad the final sequences
        max_len = max(s.shape[0] for s in all_seq_embeds)
        padded_embeds = []
        padded_masks = []

        for emb, mask in zip(all_seq_embeds, all_seq_masks):
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                emb = mx.concatenate([emb, mx.zeros((pad_len, H))], axis=0)
                mask = mx.concatenate([mask, mx.zeros((pad_len,))], axis=0)
            padded_embeds.append(emb)
            padded_masks.append(mask)

        inputs_embeds = mx.stack(padded_embeds, axis=0)
        attention_mask = mx.stack(padded_masks, axis=0)

        return inputs_embeds, attention_mask

    def generate(self, batch: list[dict], max_new_tokens: int = 200) -> list[str]:
        """Generate text for a batch of samples.

        Each sample is a dict with keys:
            pre_prompt:        str
            time_series_text:  list[str]
            time_series:       list of 1D arrays
            post_prompt:       str
        """
        results = []
        for sample in batch:
            inputs_embeds, _ = self.pad_and_apply_batch([sample])
            mx.eval(inputs_embeds)
            text = self._generate_from_embeddings(inputs_embeds[0], max_new_tokens)
            results.append(text)
        return results

    def _generate_from_embeddings(self, input_embeddings: mx.array, max_new_tokens: int) -> str:
        """Generate text from pre-computed interleaved embeddings."""
        tokens = []
        eos_ids = self.tokenizer.eos_token_ids

        for token, _ in generate_step(
            prompt=mx.array([]),
            model=self.llm,
            input_embeddings=input_embeddings,
            max_tokens=max_new_tokens,
        ):
            tokens.append(token)
            if token in eos_ids:
                break

        return self.tokenizer.decode(tokens)


def _convert_encoder_weights(state_dict: dict) -> dict:
    """Convert PyTorch TransformerCNNEncoder state_dict to MLX format.

    TODO: Remove once checkpoints are pre-converted to safetensors.
    """
    weights = {}

    # Conv1d: PyTorch [C_out, C_in, K] -> MLX [C_out, K, C_in]
    weights["patch_embed.weight"] = mx.array(
        state_dict["patch_embed.weight"].numpy().transpose(0, 2, 1)
    )
    weights["pos_embed"] = mx.array(state_dict["pos_embed"].numpy())
    weights["input_norm.weight"] = mx.array(state_dict["input_norm.weight"].numpy())
    weights["input_norm.bias"] = mx.array(state_dict["input_norm.bias"].numpy())

    num_layers = sum(
        1 for k in state_dict if k.startswith("encoder.layers.") and k.endswith(".norm1.weight")
    )
    for i in range(num_layers):
        pt_pfx = f"encoder.layers.{i}"
        mx_pfx = f"layers.{i}"

        # Split combined QKV -> separate Q, K, V
        in_proj_w = state_dict[f"{pt_pfx}.self_attn.in_proj_weight"].numpy()
        in_proj_b = state_dict[f"{pt_pfx}.self_attn.in_proj_bias"].numpy()
        d = in_proj_w.shape[1]

        weights[f"{mx_pfx}.self_attn.query_proj.weight"] = mx.array(in_proj_w[:d])
        weights[f"{mx_pfx}.self_attn.key_proj.weight"] = mx.array(in_proj_w[d : 2 * d])
        weights[f"{mx_pfx}.self_attn.value_proj.weight"] = mx.array(in_proj_w[2 * d :])
        weights[f"{mx_pfx}.self_attn.query_proj.bias"] = mx.array(in_proj_b[:d])
        weights[f"{mx_pfx}.self_attn.key_proj.bias"] = mx.array(in_proj_b[d : 2 * d])
        weights[f"{mx_pfx}.self_attn.value_proj.bias"] = mx.array(in_proj_b[2 * d :])

        weights[f"{mx_pfx}.self_attn.out_proj.weight"] = mx.array(
            state_dict[f"{pt_pfx}.self_attn.out_proj.weight"].numpy()
        )
        weights[f"{mx_pfx}.self_attn.out_proj.bias"] = mx.array(
            state_dict[f"{pt_pfx}.self_attn.out_proj.bias"].numpy()
        )

        for name in ["linear1", "linear2"]:
            for param in ["weight", "bias"]:
                weights[f"{mx_pfx}.{name}.{param}"] = mx.array(
                    state_dict[f"{pt_pfx}.{name}.{param}"].numpy()
                )

        for name in ["norm1", "norm2"]:
            for param in ["weight", "bias"]:
                weights[f"{mx_pfx}.{name}.{param}"] = mx.array(
                    state_dict[f"{pt_pfx}.{name}.{param}"].numpy()
                )

    return weights


def _convert_projector_weights(state_dict: dict) -> dict:
    """Convert PyTorch MLPProjector state_dict to MLX format.

    TODO: Remove once checkpoints are pre-converted to safetensors.
    """
    return {
        "norm.weight": mx.array(state_dict["projector.0.weight"].numpy()),
        "norm.bias": mx.array(state_dict["projector.0.bias"].numpy()),
        "linear.weight": mx.array(state_dict["projector.1.weight"].numpy()),
        "linear.bias": mx.array(state_dict["projector.1.bias"].numpy()),
    }
