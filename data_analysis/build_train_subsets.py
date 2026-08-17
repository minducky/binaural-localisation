"""Build nested, class-balanced train-set subsets for data-efficiency experiments.

Scans the training split's raw HDF5 stimulus files, groups samples by
(azim, elev) class, and for each ratio in ``dataset.TRAIN_SUBSET_RATIOS``
saves a class-balanced, nested subset (each class's samples are shuffled
once with a fixed seed, then every ratio takes a prefix of that same
shuffle, so smaller ratios are always subsets of larger ones) to
``DATASET_DIR/train_subsets/subset_{slug}.hdf5``. Also reports the
index_foreground / foreground_index_brir distribution of each subset, to
check that stratifying by (azim, elev) alone doesn't collapse source-sound
or room diversity, and the per-(azim, elev)-class sample counts, to check
that every localisation class is still represented even at the smallest
ratios (each class is guaranteed at least 1 sample by construction).

Must be run before ``yang/precompute_subset_ei.py``, which consumes the
subset files this script produces.

Usage:
    python data_analysis/build_train_subsets.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import yaml  # noqa: E402

from dataset import TRAIN_SUBSET_RATIOS, subset_slug  # noqa: E402


def scan_train_records(train_dir: str) -> dict[str, np.ndarray]:
    """Reads per-sample metadata from every ``stim*`` file in ``train_dir``.

    Args:
        train_dir: Directory containing the training split's HDF5 files.

    Returns:
        A dict of parallel arrays (one entry per training sample, in file
        order): ``h5_fname``, ``sample_idx``, ``azim``, ``elev``,
        ``index_foreground``, ``foreground_index_brir``.
    """
    h5_fnames = sorted(f for f in os.listdir(train_dir) if f.startswith("stim"))

    all_h5_fname = []
    all_sample_idx = []
    all_azim = []
    all_elev = []
    all_index_foreground = []
    all_foreground_index_brir = []

    for h5_fname in h5_fnames:
        with h5py.File(os.path.join(train_dir, h5_fname), "r") as f:
            n = len(f["signal"])
            all_h5_fname.extend([h5_fname] * n)
            all_sample_idx.extend(range(n))
            all_azim.extend(f["foreground_azim"][:].tolist())
            all_elev.extend(f["foreground_elev"][:].tolist())
            all_index_foreground.extend(f["index_foreground"][:].tolist())
            all_foreground_index_brir.extend(f["foreground_index_brir"][:].tolist())

    return {
        "h5_fname": np.array(all_h5_fname, dtype=object),
        "sample_idx": np.array(all_sample_idx, dtype=np.int64),
        "azim": np.array(all_azim, dtype=np.float64),
        "elev": np.array(all_elev, dtype=np.float64),
        "index_foreground": np.array(all_index_foreground, dtype=np.int64),
        "foreground_index_brir": np.array(all_foreground_index_brir, dtype=np.int64),
    }


def build_nested_subsets(
    class_to_entries: dict[tuple[float, float], list[int]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Builds nested, class-balanced subsets of sample entry-indices.

    Each class's samples are shuffled once (seeded), and every ratio takes
    a prefix of that same shuffled order, sized
    ``min(n, max(1, round(n * ratio)))`` (at least 1 sample per class, even
    for very small ratios). Because every ratio's selection is a prefix of
    the same per-class shuffle, smaller ratios are always subsets of larger
    ones.

    Args:
        class_to_entries: Maps each (azim, elev) class to the list of flat
            sample indices belonging to it.
        ratios: Maps a subset key (e.g. '2%') to its target fraction.
        seed: RNG seed for the per-class shuffle, shared across all ratios.

    Returns:
        A dict mapping each ratio key to a sorted int array of selected
        flat sample indices.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(ratios.items(), key=lambda kv: kv[1])
    selected: dict[str, list[int]] = {key: [] for key, _ in ordered}

    for entries in class_to_entries.values():
        shuffled = np.array(entries)
        rng.shuffle(shuffled)
        n = len(shuffled)
        for key, ratio in ordered:
            k = min(n, max(1, round(n * ratio)))
            selected[key].extend(shuffled[:k].tolist())

    return {
        key: np.array(sorted(idxs), dtype=np.int64) for key, idxs in selected.items()
    }


def save_subset(
    records: dict[str, np.ndarray], indices: np.ndarray, output_path: str
) -> None:
    """Saves a subset's (h5_fname, sample_idx) pairs to an HDF5 index file.

    Args:
        records: Full record arrays from ``scan_train_records``.
        indices: Flat sample indices (into ``records``) to include.
        output_path: Destination HDF5 path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5_fname = np.array(records["h5_fname"][indices].tolist(), dtype=object)
        f.create_dataset("h5_fname", data=h5_fname, dtype=string_dtype)
        f.create_dataset("sample_idx", data=records["sample_idx"][indices])
    print(f" - Saved {output_path}: {len(indices)} samples")


def report_distribution(
    records: dict[str, np.ndarray],
    indices: np.ndarray,
    ratio_key: str,
    figure_dir: str,
) -> None:
    """Prints and plots the index_foreground/foreground_index_brir distribution.

    Args:
        records: Full record arrays from ``scan_train_records``.
        indices: Flat sample indices (into ``records``) selected for this
            ratio.
        ratio_key: Label for this subset (e.g. '2%' or 'Total'), used in
            printed output and output filenames.
        figure_dir: Directory to save the plotly histogram HTML files to.
    """
    slug = "total" if ratio_key == "Total" else subset_slug(ratio_key)
    for key in ("index_foreground", "foreground_index_brir"):
        values = records[key][indices]
        unique_vals, counts = np.unique(values, return_counts=True)
        n_unique_total = len(np.unique(records[key]))
        print(
            f"   [{ratio_key}] {key}: {len(unique_vals)} unique values "
            f"(of {n_unique_total} in full train set)"
        )
        fig = go.Figure(go.Bar(x=unique_vals, y=counts))
        fig.update_layout(
            title=f"{key} distribution — {ratio_key}",
            xaxis_title=key,
            yaxis_title="count",
        )
        fig.write_html(os.path.join(figure_dir, f"{key}_{slug}.html"))


def report_class_coverage(
    class_to_entries: dict[tuple[float, float], list[int]],
    subsets: dict[str, np.ndarray],
    ratios: dict[str, float],
    figure_dir: str,
) -> None:
    """Prints and plots per-(azim, elev)-class sample counts for each ratio.

    ``build_nested_subsets`` guarantees at least 1 sample per class via its
    floor, but that guarantee is otherwise invisible - this makes it
    visible and flags how many classes are sitting right at the floor,
    which matters most for the smallest ratios (0.25% / 0.5% / 1%).

    Args:
        class_to_entries: Maps each (azim, elev) class to the list of flat
            sample indices belonging to it.
        subsets: Maps each ratio key to its selected flat sample indices,
            as returned by ``build_nested_subsets``.
        ratios: Maps a subset key (e.g. '2%') to its target fraction.
        figure_dir: Directory to save the plotly bar chart HTML files to.
    """
    entry_to_class: dict[int, tuple[float, float]] = {}
    for cls, entries in class_to_entries.items():
        for entry in entries:
            entry_to_class[entry] = cls

    classes = sorted(class_to_entries)

    for ratio_key in sorted(ratios, key=ratios.get):
        counts_by_class = dict.fromkeys(classes, 0)
        for entry in subsets[ratio_key]:
            counts_by_class[entry_to_class[int(entry)]] += 1
        counts = np.array([counts_by_class[cls] for cls in classes])
        n_floor = int(np.sum(counts == 1))

        print(
            f"   [{ratio_key}] class coverage: {len(classes)} classes, "
            f"samples/class min={counts.min()} max={counts.max()} "
            f"mean={counts.mean():.2f}, {n_floor} classes at the 1-sample floor"
        )

        fig = go.Figure(
            go.Bar(x=[f"{azim},{elev}" for azim, elev in classes], y=counts)
        )
        fig.update_layout(
            title=f"Samples per (azim, elev) class — {ratio_key}",
            xaxis_title="(azim, elev) class",
            yaxis_title="sample count",
        )
        fig.write_html(
            os.path.join(figure_dir, f"class_coverage_{subset_slug(ratio_key)}.html")
        )


def main() -> None:
    with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)

    train_dir = os.path.join(config["DATASET_DIR"], "train")
    seed = config.get("RANDOM_SEED", 42)

    print(f"Scanning {train_dir} ...")
    records = scan_train_records(train_dir)
    n_total = len(records["h5_fname"])
    print(f"Found {n_total} training samples.")

    class_to_entries: dict[tuple[float, float], list[int]] = {}
    for i, (azim, elev) in enumerate(
        zip(records["azim"], records["elev"], strict=True)
    ):
        class_to_entries.setdefault((azim, elev), []).append(i)
    print(f"Found {len(class_to_entries)} (azim, elev) classes.")

    subsets = build_nested_subsets(class_to_entries, TRAIN_SUBSET_RATIOS, seed)

    output_dir = os.path.join(config["DATASET_DIR"], "train_subsets")
    figure_dir = os.path.join(SCRIPT_DIR, "train_subsets")
    os.makedirs(figure_dir, exist_ok=True)

    print("\n=== Distribution report ===")
    report_distribution(records, np.arange(n_total), "Total", figure_dir)

    for ratio_key in sorted(TRAIN_SUBSET_RATIOS, key=TRAIN_SUBSET_RATIOS.get):
        indices = subsets[ratio_key]
        output_path = os.path.join(output_dir, f"subset_{subset_slug(ratio_key)}.hdf5")
        save_subset(records, indices, output_path)
        report_distribution(records, indices, ratio_key, figure_dir)

    print("\n=== Class coverage report ===")
    report_class_coverage(class_to_entries, subsets, TRAIN_SUBSET_RATIOS, figure_dir)

    print("\nDone! All train subsets built.")


if __name__ == "__main__":
    main()
