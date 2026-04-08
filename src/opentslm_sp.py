#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the OpenTSLMMLX open-source project.
#
# SPDX-License-Identifier: MIT
#

"""MLX OpenTSLMSP: end-to-end time-series language model."""

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
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
        """Load pre-converted encoder/projector/LoRA safetensors weights.

        Args:
            path: prefix path used for:
              {path}.encoder.safetensors
              {path}.projector.safetensors
              {path}.lora.safetensors (optional)
        """
        encoder_state = mx.load(f"{path}.encoder.safetensors")
        projector_state = mx.load(f"{path}.projector.safetensors")

        max_patches = encoder_state["pos_embed"].shape[1]
        self.encoder = TransformerCNNEncoder(max_patches=max_patches)
        self.encoder.load_weights(list(encoder_state.items()))

        self.projector.load_weights(list(projector_state.items()))

        try:
            lora_state = mx.load(f"{path}.lora.safetensors")
            self._apply_lora(lora_state)
        except Exception:
            pass

        print(f"  Loaded model from prefix: {path}")

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

        lora_weights = list(lora_state.items())
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
