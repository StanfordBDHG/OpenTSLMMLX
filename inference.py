"""Run OpenTSLMSP inference on a single Sleep-EDF sample."""

import sys
import argparse

sys.path.insert(0, "src")

from opentslm_sp import OpenTSLMSP
from sleep_dataset import SleepEDFDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_checkpoint.pt")
    parser.add_argument("--model", default="models/Llama-3.2-1B-bf16")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()

    print("Loading model...")
    model = OpenTSLMSP(args.model)
    model.load_from_file(args.checkpoint)

    print(f"Loading Sleep-EDF dataset (split={args.split})...")
    dataset = SleepEDFDataset(split=args.split)

    sample = dataset[args.sample_idx]
    print(f"Running inference on sample {args.sample_idx}...")
    result = model.generate([sample], max_new_tokens=args.max_new_tokens)[0]

    print(f"\nLabel: {sample['label']}")
    print(f"Output: {result}")


if __name__ == "__main__":
    main()
