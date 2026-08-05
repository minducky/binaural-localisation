"""Precompute EI features for each train-subset ratio (data-efficiency experiments).

Reuses ``CochlearPipeline``/``precompute_split`` from ``precompute_ei.py`` to
run the same (non-trainable) cochlear pipeline over each ratio in
``dataset.TRAIN_SUBSET_RATIOS``' precomputed raw-audio subset (see
``data_analysis/build_train_subsets.py``, which must be run first), so the
EI features are computed on *exactly* the same samples Ducky-style models
train on at that ratio. Does not touch ``precompute_ei.py`` or the existing
``train_ei.hdf5``/``valid_ei.hdf5``/``eval_ei.hdf5`` — those keep covering
the full ('Total') train set and the unfiltered valid/eval sets.

Usage:
    python yang/precompute_subset_ei.py [--gpu 0] [--batch_size 16]
        [--config config.yaml]

Output:
    <EI_DATASET_DIR>/train_subsets/
        subset_2pct.hdf5
        subset_5pct.hdf5
        subset_10pct.hdf5
        subset_25pct.hdf5
        subset_50pct.hdf5
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import argparse  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

from dataset import TRAIN_SUBSET_RATIOS, BinauralDataset, subset_slug  # noqa: E402
from yang.precompute_ei import CochlearPipeline, precompute_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute EI features for each train-subset ratio"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for forward pass"
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="DataLoader workers"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "config.yaml"),
        help="Config file path",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline = CochlearPipeline().to(device)
    pipeline.eval()

    output_dir = os.path.join(config["EI_DATASET_DIR"], "train_subsets")

    for ratio_key in sorted(TRAIN_SUBSET_RATIOS, key=TRAIN_SUBSET_RATIOS.get):
        print(f"\n=== Train subset: {ratio_key} ===")
        ratio_config = dict(config, DATASET_MODE=ratio_key)
        train_subset_dataset = BinauralDataset(
            ratio_config, "train", cache_size=config["CACHE_SIZE"]["TRAIN"]
        )
        precompute_split(
            pipeline,
            train_subset_dataset,
            device,
            args.batch_size,
            args.num_workers,
            os.path.join(output_dir, f"subset_{subset_slug(ratio_key)}.hdf5"),
        )

    print("\nDone! EI features precomputed for every train-subset ratio.")


if __name__ == "__main__":
    main()
