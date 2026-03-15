"""Prepare placeholder parquet files for PoisonClaw VWA experiments.

The parquet files are NOT task data — verl-agent uses them only to determine
modality (visual) and batch size. The actual VWA tasks come from the
VisualWebArenaEnvManager at runtime (via Docker or task JSON file).

Usage:
    python examples/data_preprocess/prepare_vwa.py \\
        --train_data_size 16 \\
        --val_data_size 8 \\
        --local_dir ~/data/verl-agent/
"""

import argparse
import os

import numpy as np
from PIL import Image

# Write a single shared placeholder PNG that all rows point to
_PLACEHOLDER_PATH = os.path.expanduser("~/data/verl-agent/visual/placeholder.png")


def _ensure_placeholder() -> str:
    os.makedirs(os.path.dirname(_PLACEHOLDER_PATH), exist_ok=True)
    if not os.path.exists(_PLACEHOLDER_PATH):
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(_PLACEHOLDER_PATH)
    return _PLACEHOLDER_PATH


def make_placeholder_row(idx: int, split: str) -> dict:
    """Return one placeholder dataset row in visual-agent format."""
    return {
        "data_source": "visual",
        # A single <image> token — replaced at runtime by the VWA screenshot
        "prompt": [{"role": "user", "content": "<image>"}],
        # file:// URL — fetch_image() handles this without modification
        "images": [{"image": f"file://{_PLACEHOLDER_PATH}"}],
        "ability": "agent",
        "extra_info": {"split": split, "index": idx},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VWA placeholder parquet files")
    parser.add_argument("--local_dir", default="~/data/verl-agent/visual/")
    parser.add_argument("--train_data_size", type=int, default=16)
    parser.add_argument("--val_data_size", type=int, default=8)
    args = parser.parse_args()

    try:
        import datasets
    except ImportError:
        raise ImportError("datasets is required: pip install datasets")

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    _ensure_placeholder()

    train_rows = [make_placeholder_row(i, "train") for i in range(args.train_data_size)]
    val_rows = [make_placeholder_row(i, "test") for i in range(args.val_data_size)]

    train_dataset = datasets.Dataset.from_list(train_rows)
    val_dataset = datasets.Dataset.from_list(val_rows)

    train_path = os.path.join(local_dir, "train.parquet")
    val_path = os.path.join(local_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Wrote {args.train_data_size} train rows → {train_path}")
    print(f"Wrote {args.val_data_size}  val rows  → {val_path}")


if __name__ == "__main__":
    main()
