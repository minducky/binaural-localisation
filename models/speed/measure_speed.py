"""Benchmark 1-epoch training wall-clock time across model variants.

Covers all 10 Saddler architectures, 9 Ducky variants (non-learnable
periphery), and the Yang model (full YangModel pipeline, not YangMLP).
Each target is built from its own config file/overrides, run through one
real training epoch over the actual dataloader, and timed via
``model_dissection.measure_train_epoch_time``. Results are appended to a
CSV incrementally, one row per model, so a partial run still leaves usable
output.

Must be run with the HPC dataset mounted (paths under ``/scratch/...`` in
the config files). Can be launched from any working directory - the
process chdir's to the repo root on startup, since config values like
``SADDLER.ARCH_DIR`` and ``FILTER_COEFF_DIR`` are repo-root-relative paths.

Usage:
    python models/speed/measure_speed.py
    python models/speed/measure_speed.py --debug --group saddler --only Saddler_arch01
    python models/speed/measure_speed.py --gpu 1 --group ducky
    python models/speed/measure_speed.py --batch-size 16 --group ducky
"""

import argparse
import csv
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from torch import nn, optim

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from model_dissection import measure_train_epoch_time  # noqa: E402

from dataset import setup_develop_dataloaders  # noqa: E402
from models.get_model import get_model  # noqa: E402

# %% Benchmark target definitions

CONFIGS_DIR = REPO_ROOT / "configs"

SADDLER_TEMPLATE = CONFIGS_DIR / "Saddler" / "Saddler_simplified_IHC3000_arch02.yaml"
SADDLER_ARCHS = [f"arch{i:02d}" for i in range(1, 11)]

# The 9 non-learnable-periphery Ducky variants (excludes Learnable_Ducky_*
# and the *_reproduce config), each mapping to a distinct model class via
# its own EXPERIMENT_CONFIGS.MODEL value.
DUCKY_CONFIGS = [
    "Ducky_IldOnly_MultiScale.yaml",
    "Ducky_IldOnly_Whole.yaml",
    "Ducky_IldOnly_Onset.yaml",
    "Ducky_ItdIld_Onset.yaml",
    "Ducky_ItdIld_Whole.yaml",
    "Ducky_ItdOnly_MultiScale.yaml",
    "Ducky_ItdOnly_Whole.yaml",
    "Ducky_ItdOnly_Onset.yaml",
    "Ducky_Itdild_MultiScale.yaml",
]

YANG_TEMPLATE = CONFIGS_DIR / "Yang" / "yang.yaml"
YANG_HIDDEN_DIM_1 = 1050
YANG_HIDDEN_DIM_2 = 350

CSV_FIELDS = [
    "model_group",
    "model_name",
    "config_source",
    "total_params",
    "trainable_params",
    "batch_size",
    "num_train_batches",
    "epoch_time_sec",
    "status",
]


def build_targets(group: str) -> list[dict]:
    """Builds the list of benchmark targets for the requested group.

    Args:
        group: One of ``"saddler"``, ``"ducky"``, ``"yang"``, ``"all"``.

    Returns:
        A list of target dicts, each with ``group``, ``name``,
        ``config_path``, and ``overrides`` (a ``{(key_path): value}`` dict
        applied on top of the loaded config after experiment overrides).
    """
    targets = []

    if group in ("saddler", "all"):
        for arch in SADDLER_ARCHS:
            targets.append(
                {
                    "group": "Saddler",
                    "name": f"Saddler_{arch}",
                    "config_path": SADDLER_TEMPLATE,
                    "overrides": {
                        ("SADDLER", "ARCH_DIR"): (
                            f"models/saddler_configs/simplified_IHC3000/{arch}"
                        ),
                    },
                }
            )

    if group in ("ducky", "all"):
        for fname in DUCKY_CONFIGS:
            targets.append(
                {
                    "group": "Ducky",
                    "name": Path(fname).stem,
                    "config_path": CONFIGS_DIR / "Ducky" / fname,
                    "overrides": {},
                }
            )

    if group in ("yang", "all"):
        targets.append(
            {
                "group": "Yang",
                "name": f"Yang_h{YANG_HIDDEN_DIM_1}_h{YANG_HIDDEN_DIM_2}",
                "config_path": YANG_TEMPLATE,
                "overrides": {
                    ("MODEL",): "Yang",
                    ("YANG", "HIDDEN_DIM_1"): YANG_HIDDEN_DIM_1,
                    ("YANG", "HIDDEN_DIM_2"): YANG_HIDDEN_DIM_2,
                },
            }
        )

    return targets


# %% Config preparation


def apply_experiment_overrides(config: dict) -> dict:
    """Applies the single-experiment EXPERIMENT_CONFIGS values onto config.

    Mirrors ``ExperimentManager.get_experiment_config`` for a sweep of
    length 1, since every config here defines single-element lists.

    Args:
        config: Freshly loaded yaml config.

    Returns:
        The same config dict, mutated in place.
    """
    for key, values in config["EXPERIMENT_CONFIGS"].items():
        config[key] = values[0]
    return config


def apply_debug_overrides(config: dict) -> dict:
    """Shrinks the dataset via DEBUG_CONFIG, mirroring ExperimentManager.

    Args:
        config: Config dict to mutate.

    Returns:
        The same config dict, mutated in place.
    """
    config["DEBUG"] = True
    config.update(config["DEBUG_CONFIG"])
    return config


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Sets nested config values from ``{key_path: value}`` pairs.

    Args:
        config: Config dict to mutate.
        overrides: Maps a tuple of nested keys (e.g. ``("YANG",
            "HIDDEN_DIM_1")``) to the value to set at that path.

    Returns:
        The same config dict, mutated in place.
    """
    for key_path, value in overrides.items():
        target = config
        for key in key_path[:-1]:
            target = target[key]
        target[key_path[-1]] = value
    return config


def prepare_config(
    target: dict, gpu: int | None, debug: bool, batch_size: int | None
) -> dict:
    """Loads and fully resolves the config for one benchmark target.

    Args:
        target: A target dict from `build_targets`.
        gpu: If given, overrides GPU_NUM for this run.
        debug: If True, applies DEBUG_CONFIG for a small smoke-test dataset.
        batch_size: If given, overrides BATCH_SIZE for this run (applied
            after the DEBUG_CONFIG merge, so it wins over both).

    Returns:
        The resolved config dict, ready for `setup_develop_dataloaders`
        and `get_model`.
    """
    with open(target["config_path"]) as f:
        config = yaml.safe_load(f)

    apply_experiment_overrides(config)
    apply_overrides(config, target["overrides"])
    config["PROFILER"] = False

    if gpu is not None:
        config["GPU_NUM"] = gpu
    if debug:
        apply_debug_overrides(config)
    else:
        config["DEBUG"] = False
    if batch_size is not None:
        config["BATCH_SIZE"] = batch_size

    return config


# %% Benchmark execution


def run_target(
    target: dict, gpu: int | None, debug: bool, batch_size: int | None
) -> dict:
    """Builds one model, times a real training epoch, and counts params.

    Any exception during setup/training is caught so one broken target
    doesn't stop the rest of the benchmark; it's recorded in the row's
    ``status`` column instead.

    Args:
        target: A target dict from `build_targets`.
        gpu: If given, overrides GPU_NUM for this run.
        debug: If True, uses each config's small DEBUG_CONFIG dataset.
        batch_size: If given, overrides BATCH_SIZE for this run.

    Returns:
        A CSV row dict matching `CSV_FIELDS`.
    """
    row = {
        "model_group": target["group"],
        "model_name": target["name"],
        "config_source": str(target["config_path"].relative_to(REPO_ROOT)),
        "total_params": "",
        "trainable_params": "",
        "batch_size": "",
        "num_train_batches": "",
        "epoch_time_sec": "",
        "status": "ok",
    }

    model, optimizer, dataloaders = None, None, None
    try:
        config = prepare_config(target, gpu, debug, batch_size)

        dataloaders, class_mapping = setup_develop_dataloaders(config)
        config["NUM_CLASSES"] = len(class_mapping["class_to_angle"])
        model = get_model(config)

        device = torch.device(
            f"cuda:{config['GPU_NUM']}" if torch.cuda.is_available() else "cpu"
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
        )
        criterion = nn.CrossEntropyLoss()

        epoch_time = measure_train_epoch_time(
            model, dataloaders["train"], optimizer, criterion, device
        )

        row["total_params"] = sum(p.numel() for p in model.parameters())
        row["trainable_params"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        row["batch_size"] = config["BATCH_SIZE"]
        row["num_train_batches"] = len(dataloaders["train"])
        row["epoch_time_sec"] = round(epoch_time, 3)
    except Exception:
        row["status"] = "ERROR: " + traceback.format_exc(limit=5).replace(
            "\n", " | "
        )
    finally:
        # Each target builds its own model/dataloaders in one long-lived
        # process; without an explicit release, cached CUDA allocations
        # from one target can fragment memory and starve the next one.
        del model, optimizer, dataloaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return row


def run_benchmark(
    targets: list[dict],
    gpu: int | None,
    debug: bool,
    batch_size: int | None,
    output_path: Path,
) -> None:
    """Runs every target in order, writing each result to CSV as it finishes.

    Args:
        targets: Targets to benchmark, in order.
        gpu: If given, overrides GPU_NUM for every target.
        debug: If True, uses each config's small DEBUG_CONFIG dataset.
        batch_size: If given, overrides BATCH_SIZE for every target.
        output_path: CSV file to write (created/overwritten).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(targets)} target(s), writing results to {output_path}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()

        for i, target in enumerate(targets, start=1):
            print(f"\n[{i}/{len(targets)}] {target['group']}/{target['name']}")
            row = run_target(target, gpu, debug, batch_size)
            writer.writerow(row)
            f.flush()
            print(f"  -> {row['status']} | epoch_time_sec={row['epoch_time_sec']}")


# %% CLI


def main() -> None:
    """Parses CLI args and runs the requested slice of the benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="Override GPU_NUM for every target."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use each config's small DEBUG_CONFIG dataset for a quick smoke test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override BATCH_SIZE for every target (e.g. to work around OOM "
        "on a smaller GPU). Applied after --debug, so it wins over both.",
    )
    parser.add_argument(
        "--group",
        choices=["saddler", "ducky", "yang", "all"],
        default="all",
        help="Restrict to one model family (default: all).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only run the target whose model_name matches exactly.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: models/speed/epoch_speed_<timestamp>.csv).",
    )
    args = parser.parse_args()

    targets = build_targets(args.group)
    if args.only:
        targets = [t for t in targets if t["name"] == args.only]
        if not targets:
            raise SystemExit(f"No target named {args.only!r} found.")

    output_path = args.output or (
        Path(__file__).resolve().parent
        / f"epoch_speed_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )

    run_benchmark(targets, args.gpu, args.debug, args.batch_size, output_path)


if __name__ == "__main__":
    main()
