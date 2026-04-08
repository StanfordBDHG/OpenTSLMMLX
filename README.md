# OpenTSLM SP — MLX

MLX port of [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM)'s SP (Soft Prompt) variant for inference on Apple Silicon.

## Project Structure

```
src/                  # Python MLX implementation
  ts_encoder.py       # TransformerCNNEncoder
  ts_projector.py     # MLPProjector
  opentslm_sp.py      # End-to-end model (includes interleave logic)
  sleep_dataset.py    # Sleep-EDF dataset loader (auto-downloads data)
checkpoints/          # Converted safetensors weights
models/               # LLM base weights
```

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the base LLM

Download [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) in **bf16** (full precision) and place it in `models/`:

```bash
hf download meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B-bf16
```

The bf16 model is required because the LoRA adapters in the checkpoints were trained against
full-precision weights. Quantized models (e.g. 4-bit) have different weight shapes and cannot
be combined with these LoRA weights.

### 3. Download a checkpoint

Download a trained `.pt` checkpoint and place it in `checkpoints/`:

| Checkpoint            | Task                             | Source                                                      |
| --------------------- | -------------------------------- | ----------------------------------------------------------- |
| `model_checkpoint.pt` | EEG / sleep stage classification | [HuggingFace](https://huggingface.co/OpenTSLM) |

Then run one-time conversion to safetensors:

```bash
# One-time conversion dependencies
pip install torch peft safetensors

python convert_checkpoint.py \
  --input checkpoints/model_checkpoint.pt \
  --output-prefix checkpoints/model_checkpoint
```

This writes:

- `checkpoints/model_checkpoint.encoder.safetensors`
- `checkpoints/model_checkpoint.projector.safetensors`
- `checkpoints/model_checkpoint.lora.safetensors` (if LoRA exists)

## Running Inference

The Sleep-EDF dataset is auto-downloaded on first run.

```bash
source .venv/bin/activate

# Single sample from Sleep-EDF test set
python inference.py

# Pick a specific sample
python inference.py --sample-idx 5

# Control generation length
python inference.py --max-new-tokens 500
```
