"""Minimal Sleep-EDF CoT dataset loader for MLX inference."""

import ast
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://polybox.ethz.ch/index.php/s/ZryWSdCFJZ9JR3R/download"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sleep")
DATA_PATH = os.path.join(DATA_DIR, "sleep_cot.csv")

PRE_PROMPT = (
    "You are given a 30-second EEG time series segment. "
    "Your task is to classify the sleep stage based on the signal's characteristics.\n\n"
)

POST_PROMPT = (
    "Possible sleep stages are:\n"
    "        Wake, Non-REM stage 1, Non-REM stage 2, Non-REM stage 3, REM sleep, Movement\n\n"
    "First, describe the key features of this EEG signal (amplitude, frequency, "
    "presence of specific waveforms). Then, based on your analysis, classify the "
    "sleep stage.\n\nAnswer: "
)


def _download_csv():
    """Download the Sleep-EDF CoT CSV if not already cached."""
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading Sleep-EDF CoT data to {DATA_PATH}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def _parse_time_series(ts_str: str) -> list[float]:
    """Parse string-encoded time series '[[v1, v2, ...]]' → flat list."""
    parsed = ast.literal_eval(ts_str)
    if isinstance(parsed[0], list):
        parsed = parsed[0]
    return parsed


class SleepEDFDataset:
    """Sleep-EDF CoT QA dataset for MLX inference.

    Each sample is a 30-second EEG segment (1500 data points) with a sleep stage label.
    Data is auto-downloaded on first use.
    """

    _cache = {}

    def __init__(self, split: str = "test"):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be 'train', 'validation', or 'test', got '{split}'")

        if split not in SleepEDFDataset._cache:
            _download_csv()
            self._load_splits()

        self.samples = SleepEDFDataset._cache[split]

    def _load_splits(self):
        df = pd.read_csv(DATA_PATH)

        # Stratified 80/10/10 split by label
        train_df, test_df = train_test_split(
            df, test_size=0.1, stratify=df["label"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.1 / 0.9, stratify=train_df["label"], random_state=42
        )

        for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            SleepEDFDataset._cache[name] = [
                self._format_sample(row) for _, row in split_df.iterrows()
            ]

    def _format_sample(self, row) -> dict:
        raw = _parse_time_series(row["time_series"])
        series = np.array(raw, dtype=np.float32)

        # Z-normalize
        mean = float(series.mean())
        std = max(float(series.std()), 1e-6)
        normalized = (series - mean) / std

        return {
            "pre_prompt": PRE_PROMPT,
            "time_series_text": [
                f"The following is the EEG time series, "
                f"it has mean {mean:.4f} and std {std:.4f}:"
            ],
            "time_series": [normalized],
            "post_prompt": POST_PROMPT,
            "label": row["label"],
            "answer": row.get("rationale", ""),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
