"""
Precompute EI features from raw binaural audio using Yang model's cochlear pipeline.

The cochlear pipeline (OuterMiddle -> Basilar -> IHC -> EI) is non-trainable,
so outputs are identical every epoch. Computing once and saving to HDF5
reduces per-epoch training time from ~2h to minutes.

Usage:
    python precompute_ei.py [--gpu 0] [--batch_size 16] [--config config.yaml]

Output:
    /scratch/dm1u25/dataset/Saddler_EI/
        train_ei.hdf5
        valid_ei.hdf5
        eval_ei.hdf5
"""

import os
import argparse
import yaml
import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from experiment_manager import BinauralDataset
from models.yang import OuterMiddleEarLayer, BasilarMembraneLayer, InnerHairCellLayer, EiLayer


class CochlearPipeline(torch.nn.Module):
    """Forward-only cochlear pipeline (OM -> BM -> IHC -> EI)."""

    def __init__(self):
        super().__init__()
        filter_param_dir = './filter_param_dir'
        os.makedirs(filter_param_dir, exist_ok=True)
        sr = 44100
        sr_res = 4000
        self.outermiddle = OuterMiddleEarLayer(filter_param_dir=filter_param_dir, sr=sr, trainable=False)
        self.basilar = BasilarMembraneLayer(filter_param_dir=filter_param_dir, sr=sr, trainable=False)
        self.ihc = InnerHairCellLayer(filter_param_dir=filter_param_dir, orig_sr=sr, target_sr=sr_res, trainable=False)
        self.ei = EiLayer(sr=sr_res, trainable=False)

    def forward(self, x):
        """
        Args:
            x: (batch, samples, 2) binaural audio
        Returns:
            ei_out: (batch, 31, 24, 12) float32 EI features
        """
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)
        coch_l = self.ihc(self.basilar(self.outermiddle(x_l)))
        coch_r = self.ihc(self.basilar(self.outermiddle(x_r)))
        ei_out = self.ei(coch_l, coch_r)  # (batch, 31, 24, 12)
        return ei_out


def precompute_split(pipeline, dataset, device, batch_size, num_workers, output_path):
    """Precompute EI features for a single dataset split and save to HDF5."""
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=False
    )

    n_samples = len(dataset)
    # Preallocate output arrays
    all_ei = np.empty((n_samples, 31, 24, 12), dtype=np.float32)
    all_class_idx = np.empty(n_samples, dtype=np.int64)
    all_azim = np.empty(n_samples, dtype=np.float32)
    all_elev = np.empty(n_samples, dtype=np.float32)
    all_snr = np.empty(n_samples, dtype=np.float32)

    offset = 0
    pbar = tqdm(dataloader, desc=f"Computing EI -> {os.path.basename(output_path)}", ncols=120)
    for signal, class_idx, azim, elev, snr in pbar:
        bs = signal.shape[0]
        signal = signal.to(device)

        with torch.no_grad():
            ei = pipeline(signal)  # (bs, 31, 24, 12)

        all_ei[offset:offset + bs] = ei.cpu().numpy()
        all_class_idx[offset:offset + bs] = class_idx.numpy()
        all_azim[offset:offset + bs] = azim.numpy()
        all_elev[offset:offset + bs] = elev.numpy()
        all_snr[offset:offset + bs] = snr.numpy()
        offset += bs

    # Trim in case last batch was partial
    all_ei = all_ei[:offset]
    all_class_idx = all_class_idx[:offset]
    all_azim = all_azim[:offset]
    all_elev = all_elev[:offset]
    all_snr = all_snr[:offset]

    # Save to HDF5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('signal_ei', data=all_ei, compression='gzip', compression_opts=4)
        f.create_dataset('class_idx', data=all_class_idx)
        f.create_dataset('azim', data=all_azim)
        f.create_dataset('elev', data=all_elev)
        f.create_dataset('snr', data=all_snr)

    file_size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved {output_path}: {offset} samples, {file_size_mb:.1f} MB")


def save_class_mapping(mapping, output_path):
    """Save class mapping to HDF5."""
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('azim_values', data=np.array(mapping['azim_values']))
        f.create_dataset('elev_values', data=np.array(mapping['elev_values']))
        # angle_to_class: store as parallel arrays of (azim, elev, class_idx)
        keys = list(mapping['angle_to_class'].keys())
        azims = np.array([k[0] for k in keys])
        elevs = np.array([k[1] for k in keys])
        classes = np.array([mapping['angle_to_class'][k] for k in keys])
        f.create_dataset('mapping_azim', data=azims)
        f.create_dataset('mapping_elev', data=elevs)
        f.create_dataset('mapping_class', data=classes)
    print(f"Saved class mapping to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Precompute EI features')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for forward pass')
    parser.add_argument('--num_workers', type=int, default=12, help='DataLoader workers')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output_dir', type=str, default='/scratch/dm1u25/dataset/Saddler_EI',
                        help='Output directory for EI features')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build cochlear pipeline
    pipeline = CochlearPipeline().to(device)
    pipeline.eval()

    # --- Train split ---
    print("\n=== Train split ===")
    train_dataset = BinauralDataset(config, "train", cache_size=config['CACHE_SIZE']['TRAIN'])
    global_class_mapping = {
        'angle_to_class': train_dataset.angle_to_class,
        'class_to_angle': train_dataset.class_to_angle,
        'azim_values': train_dataset.azim_values,
        'elev_values': train_dataset.elev_values,
    }
    precompute_split(
        pipeline, train_dataset, device, args.batch_size, args.num_workers,
        os.path.join(args.output_dir, 'train_ei.hdf5')
    )

    # Save class mapping (from train set)
    save_class_mapping(global_class_mapping, os.path.join(args.output_dir, 'class_mapping.hdf5'))

    # --- Valid split ---
    print("\n=== Valid split ===")
    valid_dataset = BinauralDataset(config, "valid", cache_size=config['CACHE_SIZE']['VAL'],
                                    global_class_mapping=global_class_mapping)
    precompute_split(
        pipeline, valid_dataset, device, args.batch_size, args.num_workers,
        os.path.join(args.output_dir, 'valid_ei.hdf5')
    )

    # --- Eval split ---
    print("\n=== Eval split ===")
    eval_dataset = BinauralDataset(config, "evaluation/v01_eval_mit_bldg46room1004_tenoise",
                                   cache_size=config['CACHE_SIZE']['EVAL'],
                                   global_class_mapping=global_class_mapping)
    precompute_split(
        pipeline, eval_dataset, device, args.batch_size, args.num_workers,
        os.path.join(args.output_dir, 'eval_ei.hdf5')
    )

    print("\nDone! All EI features precomputed.")


if __name__ == '__main__':
    main()
