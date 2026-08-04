"""
Compare precomputed EI patterns (from HDF5) vs on-the-fly EI patterns (from CochlearPipeline).

Picks 10 random samples from the training set, computes EI on-the-fly from raw audio,
and plots side-by-side with the precomputed version.

Usage:
    python compare_ei.py [--gpu 0] [--num_samples 10] [--seed 42]
"""

import os
import argparse
import yaml
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

from experiment_manager import BinauralDataset
from precompute_ei import CochlearPipeline


def main():
    parser = argparse.ArgumentParser(description='Compare precomputed vs on-the-fly EI patterns')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='./ei_comparison')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load precomputed EI
    ei_dir = config['EI_DATASET_DIR']
    precomputed_path = os.path.join(ei_dir, 'train_ei.hdf5')
    print(f"Loading precomputed EI from {precomputed_path}...")
    with h5py.File(precomputed_path, 'r') as f:
        precomputed_ei = f['signal_ei']  # (N, 31, 24, 12)
        precomputed_azim = f['azim'][:]
        precomputed_elev = f['elev'][:]
        n_precomputed = precomputed_ei.shape[0]

        # Pick random sample indices
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.choice(n_precomputed, size=args.num_samples, replace=False)
        sample_indices = np.sort(sample_indices)

        # Read only selected samples
        pre_ei = np.stack([precomputed_ei[i] for i in sample_indices])  # (num_samples, 31, 24, 12)
        pre_azim = precomputed_azim[sample_indices]
        pre_elev = precomputed_elev[sample_indices]

    print(f"Selected {args.num_samples} samples: {sample_indices}")

    # Load raw audio dataset (cache only 1 file at a time to save memory)
    print("Loading raw audio dataset...")
    raw_dataset = BinauralDataset(config, "train", cache_size=1)

    # Build cochlear pipeline
    pipeline = CochlearPipeline().to(device)
    pipeline.eval()

    # Compute on-the-fly EI for selected samples
    print("Computing on-the-fly EI...")
    onthefly_ei = []
    otf_azim = []
    otf_elev = []

    for idx in sample_indices:
        signal, class_idx, azim, elev, snr = raw_dataset[idx]
        with torch.no_grad():
            ei = pipeline(signal.unsqueeze(0).to(device))  # (1, 31, 24, 12)
        onthefly_ei.append(ei.cpu().numpy()[0])
        otf_azim.append(azim.item())
        otf_elev.append(elev.item())

    onthefly_ei = np.stack(onthefly_ei)  # (num_samples, 31, 24, 12)
    otf_azim = np.array(otf_azim)
    otf_elev = np.array(otf_elev)

    # Verify same samples (azim/elev should match)
    print("\nSample verification (azim/elev match):")
    for i in range(args.num_samples):
        match = (pre_azim[i] == otf_azim[i]) and (pre_elev[i] == otf_elev[i])
        print(f"  [{i}] idx={sample_indices[i]:6d} | "
              f"pre=({pre_azim[i]:.1f}, {pre_elev[i]:.1f}) "
              f"otf=({otf_azim[i]:.1f}, {otf_elev[i]:.1f}) | "
              f"{'OK' if match else 'MISMATCH'}")

    # Compute difference stats
    diff = onthefly_ei - pre_ei
    print(f"\nDifference stats:")
    print(f"  Max abs diff:  {np.abs(diff).max():.6e}")
    print(f"  Mean abs diff: {np.abs(diff).mean():.6e}")
    print(f"  RMS diff:      {np.sqrt(np.mean(diff**2)):.6e}")

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Flatten (31, 24, 12) -> (31, 24*12) for 2D visualization
        # Each row = frequency channel, columns = tau x alpha
        pre_flat = pre_ei[i].reshape(31, -1)       # (31, 288)
        otf_flat = onthefly_ei[i].reshape(31, -1)   # (31, 288)
        diff_flat = diff[i].reshape(31, -1)          # (31, 288)

        vmin = min(pre_flat.min(), otf_flat.min())
        vmax = max(pre_flat.max(), otf_flat.max())

        im0 = axes[0].imshow(pre_flat, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('Precomputed EI')
        axes[0].set_ylabel('Frequency channel')
        axes[0].set_xlabel('Tau x Alpha')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(otf_flat, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('On-the-fly EI')
        axes[1].set_xlabel('Tau x Alpha')
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(diff_flat, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[2].set_title(f'Difference (max={np.abs(diff_flat).max():.2e})')
        axes[2].set_xlabel('Tau x Alpha')
        plt.colorbar(im2, ax=axes[2])

        fig.suptitle(f'Sample {sample_indices[i]} | azim={pre_azim[i]:.1f}, elev={pre_elev[i]:.1f}',
                     fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(args.output_dir, f'compare_{i:02d}_idx{sample_indices[i]}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved {save_path}")

    print(f"\nDone! All plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
