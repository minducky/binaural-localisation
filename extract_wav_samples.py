import os
import yaml
import h5py
import numpy as np
from scipy.io.wavfile import write as wav_write
from glob import glob
from collections import defaultdict

if __name__ == '__main__':
    # Config
    config_fpath = './config.yaml'
    with open(config_fpath, 'r') as f:
        config = yaml.safe_load(f)

    mode = 'eval_class'  # 'train' / 'val' / 'eval' / 'eval_class'
    num_samples_per_condition = 5
    azim_step = 30
    sr = 44100
    output_dir = f'/home/dm1u25/BSL_Saddler/binaural_samples/{mode}'
    os.makedirs(output_dir, exist_ok=True)

    # Dataset directory
    dataset_dir = config['DATASET_DIR']
    dataset_dirs = {
        'train': os.path.join(dataset_dir, 'train'),
        'val': os.path.join(dataset_dir, 'valid'),
        'eval': os.path.join(dataset_dir, 'evaluation/v01_eval_mit_bldg46room1004_tenoise'),
        'eval_class': os.path.join(dataset_dir, 'evaluation/v01_eval_mit_bldg46room1004_tenoise'),
    }
    dataset_dir_to_see = dataset_dirs[mode]
    h5_fpath_list = sorted(glob(os.path.join(dataset_dir_to_see, '*.hdf5')))
    print(f'Mode: {mode}, Found {len(h5_fpath_list)} h5 files in {dataset_dir_to_see}')

    if mode == 'eval_class':
        # eval_class: elev=0, azim=0, max SNR, extract all index_foreground
        # Structure: {index_fg: [(h5_fpath, idx, snr), ...]}
        fg_to_entries = defaultdict(list)

        for h5_fpath in h5_fpath_list:
            with h5py.File(h5_fpath, 'r') as f:
                azims = f['foreground_azim'][:]
                elevs = f['foreground_elev'][:]
                snrs = f['snr'][:]
                index_fgs = f['index_foreground'][:]
                n = len(azims)
                for i in range(n):
                    if float(elevs[i]) != 0 or float(azims[i]) != 0:
                        continue
                    snr_val = float(snrs[i])
                    if np.isinf(snr_val):
                        continue
                    fg_to_entries[int(index_fgs[i])].append((h5_fpath, i, snr_val))

        print(f'Found {len(fg_to_entries)} unique index_foreground at elev=0, azim=0')

    elif mode == 'eval':
        # eval: group by (elev, azim, snr), azim filtered to 30-degree intervals
        # Structure: {(elev, azim, snr): [(h5_fpath, idx, index_fg), ...]}
        condition_indices = defaultdict(list)

        for h5_fpath in h5_fpath_list:
            with h5py.File(h5_fpath, 'r') as f:
                azims = f['foreground_azim'][:]
                elevs = f['foreground_elev'][:]
                snrs = f['snr'][:]
                index_fgs = f['index_foreground'][:]
                n = len(azims)
                for i in range(n):
                    azim_val = float(azims[i])
                    if azim_val % azim_step != 0:
                        continue
                    snr_val = float(snrs[i])
                    if np.isinf(snr_val):
                        continue
                    key = (float(elevs[i]), azim_val, snr_val)
                    condition_indices[key].append((h5_fpath, i, int(index_fgs[i])))

        print(f'Found {len(condition_indices)} unique (elev, azim, snr) conditions (azim step={azim_step})')
    else:
        # train/val: group by (elev, azim), azim filtered to 30-degree intervals
        # Structure: {(elev, azim): [(h5_fpath, idx, snr, index_fg), ...]}
        condition_indices = defaultdict(list)

        for h5_fpath in h5_fpath_list:
            with h5py.File(h5_fpath, 'r') as f:
                azims = f['foreground_azim'][:]
                elevs = f['foreground_elev'][:]
                snrs = f['snr'][:]
                index_fgs = f['index_foreground'][:]
                n = len(azims)
                for i in range(n):
                    azim_val = float(azims[i])
                    if azim_val % azim_step != 0:
                        continue
                    key = (float(elevs[i]), azim_val)
                    condition_indices[key].append((h5_fpath, i, float(snrs[i]), int(index_fgs[i])))

        print(f'Found {len(condition_indices)} unique (elev, azim) conditions')

    # Extract and save samples
    total_saved = 0
    if mode == 'eval_class':
        for index_fg, entries in sorted(fg_to_entries.items()):
            # Pick the entry with max SNR
            best = max(entries, key=lambda x: x[2])
            h5_fpath, h5_idx, snr = best

            with h5py.File(h5_fpath, 'r') as f:
                signal = f['signal'][h5_idx]  # shape: (T, 2)

            signal = signal.astype(np.float32)
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val

            snr_str = f'{snr:.1f}'
            fname = f'if_{index_fg}_snr_{snr_str}.wav'
            fpath = os.path.join(output_dir, fname)
            wav_write(fpath, sr, signal)
            total_saved += 1
    elif mode == 'eval':
        num_fg_types = 460
        for (elev, azim, snr), indices in sorted(condition_indices.items()):
            # Pick 1 sample per unique index_foreground, up to num_fg_types
            seen_fg = set()
            selected = []
            for entry in indices:
                fg = entry[2]  # index_fg
                if fg not in seen_fg:
                    seen_fg.add(fg)
                    selected.append(entry)
                if len(selected) >= num_fg_types:
                    break

            snr_str = f'{snr:.1f}' if not np.isinf(snr) else 'inf'
            for sample_idx, (h5_fpath, h5_idx, index_fg) in enumerate(selected):
                with h5py.File(h5_fpath, 'r') as f:
                    signal = f['signal'][h5_idx]  # shape: (T, 2)

                signal = signal.astype(np.float32)
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal / max_val

                fname = f'azim_{int(azim)}_elev_{int(elev)}_sample_{sample_idx}_snr_{snr_str}_if_{index_fg}.wav'
                fpath = os.path.join(output_dir, fname)
                wav_write(fpath, sr, signal)
                total_saved += 1
    else:
        for (elev, azim), indices in sorted(condition_indices.items()):
            selected = indices[:num_samples_per_condition]
            for sample_idx, (h5_fpath, h5_idx, snr, index_fg) in enumerate(selected):
                with h5py.File(h5_fpath, 'r') as f:
                    signal = f['signal'][h5_idx]  # shape: (T, 2)

                signal = signal.astype(np.float32)
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal / max_val

                snr_str = f'{snr:.1f}' if not np.isinf(snr) else 'inf'
                fname = f'azim_{int(azim)}_elev_{int(elev)}_sample_{sample_idx}_snr_{snr_str}_if_{index_fg}.wav'
                fpath = os.path.join(output_dir, fname)
                wav_write(fpath, sr, signal)
                total_saved += 1

    print(f'Done! Saved {total_saved} wav files to {output_dir}')