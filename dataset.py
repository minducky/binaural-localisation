import os
import random

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, Subset


def print_cpu_mem(tag=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1e9
    total_gb = psutil.virtual_memory().used / 1e9
    print(f"[{tag}] Process: {mem_gb:.1f}GB / System: {total_gb:.1f}GB")


class BinauralDataset(Dataset):
    def __init__(self, config, mode, cache_size, global_class_mapping=None):
        self.data_dir = os.path.join(config['DATASET_DIR'], mode)
        self.h5_fnames = sorted(f for f in os.listdir(self.data_dir) if f.startswith('stim'))

        # DATASET_MODE: 'Total' (no filter) or 'Certain' (eval-range only, train/val only)
        dataset_mode = config['DATASET_MODE']
        if dataset_mode == 'Certain' and 'eval' not in mode:
            af = config['CERTAIN_ANGLE_FILTER']
            self.allowed_azims = self._build_allowed_azims(af['AZIM_RANGE'], af['AZIM_STEP'])
            self.allowed_elevs = set(float(e) for e in np.arange(af['ELEV_RANGE'][0], af['ELEV_RANGE'][1] + af['ELEV_STEP'], af['ELEV_STEP']))
        else:
            self.allowed_azims = None  # None = no filter
            self.allowed_elevs = None

        # Separate mapping dirs per mode so Total/Certain don't overwrite each other
        self.mapping_dir = os.path.join(config['DATASET_DIR'], f'global_class_mapping_{dataset_mode.lower()}')

        # Build or load angle-to-class mappings
        if global_class_mapping is not None:
            print(" - Using global class mapping from training set...")
            self.angle_to_class = global_class_mapping['angle_to_class']
            self.class_to_angle = global_class_mapping['class_to_angle']
            self.azim_values = global_class_mapping['azim_values']
            self.elev_values = global_class_mapping['elev_values']
            print(f"     - Loaded {len(self.class_to_angle)} classes from global mapping")
        else:
            self._load_or_build_angle_mappings()

        self.index_map = []
        signal_count = 0
        normal_snr_count = 0
        for h5_fname in self.h5_fnames:
            with h5py.File(os.path.join(self.data_dir, h5_fname), 'r') as f:
                # train/val mode
                if 'eval' not in mode:
                    n_samples = len(f['signal'])
                    azims = f['foreground_azim'][:]
                    elevs = f['foreground_elev'][:]
                    for i in range(n_samples):
                        if (self.allowed_azims is None or
                                (float(azims[i]) in self.allowed_azims and float(elevs[i]) in self.allowed_elevs)):
                            self.index_map.append((h5_fname, i))
                # evaluation mode
                elif 'eval' in mode:
                    n_samples = len(f['signal'])
                    snr = f['snr'][:]
                    for i in range(n_samples):
                        signal_count += 1
                        if not np.isinf(snr[i]):  # Exclude when SNR is inf
                            self.index_map.append((h5_fname, i))
                            normal_snr_count += 1
                    if h5_fname == self.h5_fnames[-1]:
                        print(f'- len of original evaluation set : {signal_count}\n'
                              f'    - len of normal snr evaluation set : {normal_snr_count}\n'
                              f'    - excluded inf snr sample num : {signal_count - normal_snr_count}\n')


        self.cache = {}
        self.cache_size = cache_size
        self.current_file = None
        self.current_h5 = None

        self.total_length = len(self.index_map)
        print(f" - Dataset mode : {mode} / {self.total_length} samples")
        print(f" - Number of classes: {len(self.class_to_angle)}")
        self._fill_cache()

    @staticmethod
    def _build_allowed_azims(azim_range, step):
        """Build allowed azimuth set from two ranges, e.g. [270, 360, 0, 90]."""
        result = set()
        for a in np.arange(azim_range[0], azim_range[1], step):   # e.g. 270–359
            result.add(float(a))
        for a in np.arange(azim_range[2], azim_range[3] + step, step):  # e.g. 0–90
            result.add(float(a))
        return result

    def _load_or_build_angle_mappings(self):
        """Load class mapping from file if exists, otherwise build and save"""
        mapping_fpath = os.path.join(self.mapping_dir, 'global_class_mapping.hdf5')

        if os.path.exists(mapping_fpath):
            print(f" - Loading class mapping from {mapping_fpath}...")
            with h5py.File(mapping_fpath, 'r') as f:
                self.azim_values = sorted(f['azim_values'][:].tolist())
                self.elev_values = sorted(f['elev_values'][:].tolist())
                mapping_azim = f['mapping_azim'][:]
                mapping_elev = f['mapping_elev'][:]
                mapping_class = f['mapping_class'][:]

            self.angle_to_class = {}
            self.class_to_angle = {}
            for a, e, c in zip(mapping_azim, mapping_elev, mapping_class):
                self.angle_to_class[(float(a), float(e))] = int(c)
                self.class_to_angle[int(c)] = (float(a), float(e))
        else:
            print(" - Building angle-to-class mappings...")
            unique_azims = set()
            unique_elevs = set()

            for h5_fname in self.h5_fnames:
                h5_fpath = os.path.join(self.data_dir, h5_fname)
                with h5py.File(h5_fpath, 'r') as f:
                    for a, e in zip(f['foreground_azim'][:], f['foreground_elev'][:]):
                        a, e = float(a), float(e)
                        if (self.allowed_azims is None or
                                (a in self.allowed_azims and e in self.allowed_elevs)):
                            unique_azims.add(a)
                            unique_elevs.add(e)

            self.azim_values = sorted(list(unique_azims))
            self.elev_values = sorted(list(unique_elevs))

            self.angle_to_class = {}
            self.class_to_angle = {}
            for elev_idx, elev in enumerate(self.elev_values):
                for azim_idx, azim in enumerate(self.azim_values):
                    class_idx = elev_idx * len(self.azim_values) + azim_idx
                    self.angle_to_class[(azim, elev)] = class_idx
                    self.class_to_angle[class_idx] = (azim, elev)

            # Save mapping
            os.makedirs(self.mapping_dir, exist_ok=True)
            mapping_azim = np.array([a for (a, e) in self.angle_to_class.keys()])
            mapping_elev = np.array([e for (a, e) in self.angle_to_class.keys()])
            mapping_class = np.array(list(self.angle_to_class.values()))
            with h5py.File(mapping_fpath, 'w') as f:
                f.create_dataset('azim_values', data=np.array(self.azim_values))
                f.create_dataset('elev_values', data=np.array(self.elev_values))
                f.create_dataset('mapping_azim', data=mapping_azim)
                f.create_dataset('mapping_elev', data=mapping_elev)
                f.create_dataset('mapping_class', data=mapping_class)
            print(f" - Saved class mapping to {mapping_fpath}")

        print(f"     - Unique azimuths: {len(self.azim_values)}")
        print(f"     - Unique elevations: {len(self.elev_values)}")
        print(f"     - Total classes: {len(self.class_to_angle)}")

    def build_eval_prior(self, n_classes):
        """Collect unique class indices from index_map and return a prior vector.

        Returns a numpy array of shape (n_classes,) with 1.0 for classes that
        actually appear in this dataset and 0.0 elsewhere.
        """
        unique_classes = set()
        for h5_fname, sample_idx in self.index_map:
            if h5_fname in self.cache:
                azim = float(self.cache[h5_fname]['azim'][sample_idx])
                elev = float(self.cache[h5_fname]['elev'][sample_idx])
            else:
                h5_fpath = os.path.join(self.data_dir, h5_fname)
                with h5py.File(h5_fpath, 'r') as f:
                    azim = float(f['foreground_azim'][sample_idx])
                    elev = float(f['foreground_elev'][sample_idx])
            unique_classes.add(self.angle_to_class[(azim, elev)])
        prior = np.zeros(n_classes)
        for cls in unique_classes:
            prior[cls] = 1.0
        return prior

    def _fill_cache(self):
        print("\n------------- Filling cache ------------")
        for f_idx, h5_fname in enumerate(self.h5_fnames[:self.cache_size]):
            with h5py.File(os.path.join(self.data_dir, h5_fname), 'r') as f:
                self.cache[h5_fname] = {
                    "signal": f["signal"][:],
                    "azim": f["foreground_azim"][:],
                    "elev": f["foreground_elev"][:],
                    "snr": f["snr"][:],
                }
            print_cpu_mem(f"- {f_idx + 1}/{self.cache_size} file")
        print("---------- Filling Cache Done ----------\n")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        h5_file, sample_idx = self.index_map[idx]

        if h5_file in self.cache:
            h5_data = self.cache[h5_file]
            signal = h5_data["signal"][sample_idx]
            azim = h5_data["azim"][sample_idx]
            elev = h5_data["elev"][sample_idx]
            snr = h5_data["snr"][sample_idx]
        else:
            with h5py.File(os.path.join(self.data_dir, h5_file), 'r') as f:
                signal = f["signal"][sample_idx]
                azim = f["foreground_azim"][sample_idx]
                elev = f["foreground_elev"][sample_idx]
                snr = f["snr"][sample_idx]

        signal = torch.tensor(signal, dtype=torch.float32)  # (1.3sec * 44100, 2)
        # target_len = 44100
        # offset = int(44100 * 0.1)
        # signal = signal[offset:offset + target_len, :] # (0.1-1.1sec * 44100, 2)

        class_idx = self.angle_to_class[(azim, elev)]
        return (
            signal,
            torch.tensor(class_idx).long(),
            torch.tensor(azim),
            torch.tensor(elev),
            torch.tensor(snr)
        )


class EIDataset(Dataset):
    def __init__(self, h5_path, config=None, mode=None, class_mapping_path=None, global_class_mapping=None):
        """
        Args:
            h5_path: Path to precomputed EI HDF5 file (e.g. train_ei.hdf5)
            config: Config dict; if provided, DATASET_MODE / Certain filtering is applied
            mode: Dataset split string (e.g. 'train', 'valid', 'eval'); required when config is given
            class_mapping_path: Path to class_mapping.hdf5 (used if global_class_mapping is None)
            global_class_mapping: Optional dict with class mappings (same format as BinauralDataset)
        """
        print(f" - Loading precomputed EI features from '{h5_path}'...")
        with h5py.File(h5_path, 'r') as f:
            self.signal_ei = f['signal_ei'][:]      # (N, 31, 24, 12) float32
            self.class_idx = f['class_idx'][:]      # (N,) int64
            self.azim = f['azim'][:]                # (N,) float32
            self.elev = f['elev'][:]                # (N,) float32
            self.snr = f['snr'][:]                  # (N,) float32

        print(f" - Loaded {len(self.class_idx)} samples, EI shape: {self.signal_ei.shape}")

        # DATASET_MODE: 'Total' (no filter) or 'Certain' (eval-range only, train/val only)
        if config is not None and config['DATASET_MODE'] == 'Certain' and mode is not None and 'eval' not in mode:
            af = config['CERTAIN_ANGLE_FILTER']
            allowed_azims = BinauralDataset._build_allowed_azims(af['AZIM_RANGE'], af['AZIM_STEP'])
            allowed_elevs = set(float(e) for e in np.arange(af['ELEV_RANGE'][0], af['ELEV_RANGE'][1] + af['ELEV_STEP'], af['ELEV_STEP']))
            mask = np.array([
                float(a) in allowed_azims and float(e) in allowed_elevs
                for a, e in zip(self.azim, self.elev)
            ])
            self.signal_ei = self.signal_ei[mask]
            self.class_idx = self.class_idx[mask]
            self.azim      = self.azim[mask]
            self.elev      = self.elev[mask]
            self.snr       = self.snr[mask]
            print(f" - Certain mode applied: {mask.sum()} / {len(mask)} samples kept")

        self.total_length = len(self.class_idx)
        print(f" - Dataset mode : {mode} / {self.total_length} samples")

        # Load or use class mapping
        if global_class_mapping is not None:
            self.angle_to_class = global_class_mapping['angle_to_class']
            self.class_to_angle = global_class_mapping['class_to_angle']
            self.azim_values = global_class_mapping['azim_values']
            self.elev_values = global_class_mapping['elev_values']
        elif class_mapping_path is not None:
            self._load_class_mapping(class_mapping_path)
        else:
            raise ValueError("Either global_class_mapping or class_mapping_path must be provided")

    def _load_class_mapping(self, path):
        """Load class mapping from HDF5 file."""
        with h5py.File(path, 'r') as f:
            self.azim_values = sorted(f['azim_values'][:].tolist())
            self.elev_values = sorted(f['elev_values'][:].tolist())
            mapping_azim = f['mapping_azim'][:]
            mapping_elev = f['mapping_elev'][:]
            mapping_class = f['mapping_class'][:]

        self.angle_to_class = {}
        self.class_to_angle = {}
        for a, e, c in zip(mapping_azim, mapping_elev, mapping_class):
            self.angle_to_class[(float(a), float(e))] = int(c)
            self.class_to_angle[int(c)] = (float(a), float(e))

    def build_eval_prior(self, n_classes):
        """Return a prior vector: 1.0 for classes present in this dataset."""
        unique_classes = set(self.class_idx.tolist())
        prior = np.zeros(n_classes)
        for cls in unique_classes:
            prior[cls] = 1.0
        return prior

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.signal_ei[idx], dtype=torch.float32),
            torch.tensor(self.class_idx[idx]).long(),
            torch.tensor(self.azim[idx]),
            torch.tensor(self.elev[idx]),
            torch.tensor(self.snr[idx])
        )


class FileGroupedSampler(Sampler):
    """ Sampler which samples from grouped h5 files that reducing h5 I/O"""
    def __init__(self, dataset, shuffle=True):
        # Subset wrapping
        base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else range(len(dataset))

        file_groups = {}
        for idx in indices:
            h5_file, _ = base.index_map[idx]
            file_groups.setdefault(h5_file, []).append(idx)

        self.groups = list(file_groups.values())
        self.shuffle = shuffle

    def __iter__(self):
        groups = [list(g) for g in self.groups]
        if self.shuffle:
            random.shuffle(groups)
            for g in groups:
                random.shuffle(g)
        for g in groups:
            yield from g

    def __len__(self):
        return sum(len(g) for g in self.groups)


# %% Setup Dataloaders for develop (train/val) and evaluation
def setup_develop_dataloaders(config):
    """Setup train/val dataloaders and return class mapping"""
    use_ei = config['MODEL'] == 'YangMLP'
    print(f"\n{'=' * 70}")
    print(f"Cache status:")
    if use_ei:
        ei_dir = config['EI_DATASET_DIR']
        dataset_mode = config['DATASET_MODE']
        class_mapping_path = os.path.join(ei_dir, f'class_mapping_{dataset_mode.lower()}.hdf5')
        if not os.path.exists(class_mapping_path):  # fallback to default mapping
            class_mapping_path = os.path.join(ei_dir, 'class_mapping.hdf5')
        print(f"\n [[[ Training ]]] ")
        print_cpu_mem(" - Before Setting EI Dataset")
        train_dataset = EIDataset(os.path.join(ei_dir, 'train_ei.hdf5'), config=config, mode='train', class_mapping_path=class_mapping_path)

        global_class_mapping = {
            'angle_to_class': train_dataset.angle_to_class,
            'class_to_angle': train_dataset.class_to_angle,
            'azim_values': train_dataset.azim_values,
            'elev_values': train_dataset.elev_values,
        }
        print(f"\n [[[ Validation ]]] ")
        val_dataset = EIDataset(os.path.join(ei_dir, 'valid_ei.hdf5'), config=config, mode='valid', global_class_mapping=global_class_mapping)
        print_cpu_mem(" - After Setting EI Dataset")
    else:
        print(f"\n [[[ Training ]]] ")
        print_cpu_mem(" - Before Setting Binaural Dataset")
        train_dataset = BinauralDataset(config, "train", config["CACHE_SIZE"]["TRAIN"])

        base_train_dataset = train_dataset

        global_class_mapping = {
            'angle_to_class': base_train_dataset.angle_to_class,
            'class_to_angle': base_train_dataset.class_to_angle,
            'azim_values': base_train_dataset.azim_values,
            'elev_values': base_train_dataset.elev_values,
        }
        print(f"\n [[[ Validation ]]]")
        val_dataset = BinauralDataset(config, "valid", config["CACHE_SIZE"]["VAL"], global_class_mapping)
        print_cpu_mem(" - After Setting Binaural Dataset")

    print(f"{'=' * 70}")
    batch_size = config["BATCH_SIZE"]
    num_workers = config["NUM_WORKERS"]

    if config['DEBUG']:
        train_dataset = Subset(train_dataset, range(config['DEBUG_CONFIG']['TRAIN_DATASET_SIZE']))
        val_dataset = Subset(val_dataset, range(config['DEBUG_CONFIG']['VAL_DATASET_SIZE']))
        print(f"\n{'-' * 70}")
        print(f'Debug mode : \n'
              f'- ************* train dataset into {config["DEBUG_CONFIG"]["TRAIN_DATASET_SIZE"]} samples\n'
              f'- ************* val dataset into {config["DEBUG_CONFIG"]["VAL_DATASET_SIZE"]} samples\n')
        print(f"{'-' * 70}")

    sampler_mode = config['SAMPLER_MODE']
    if sampler_mode == 'grouped' and hasattr(train_dataset, 'index_map'):
        train_sampler = FileGroupedSampler(train_dataset, shuffle=True)
        train_loader_kwargs = dict(sampler=train_sampler, shuffle=False)
    else:
        train_loader_kwargs = dict(shuffle=True)

    print(f"\n{'=' * 70}")
    print(f"Train DataLoader sampler mode: {sampler_mode}")
    print(f"{'=' * 70}\n\n")

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, **train_loader_kwargs),
        'val': DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, shuffle=False)
    }

    return dataloaders, global_class_mapping

def setup_eval_dataloaders(config, global_class_mapping):
    """Setup eval dataloaders with global class mapping from training"""
    use_ei = config['MODEL'] == 'YangMLP'

    if use_ei:
        ei_dir = config['EI_DATASET_DIR']
        print_cpu_mem("Before Setting EI Evaluation Dataset")
        eval_dataset = EIDataset(os.path.join(ei_dir, 'eval_ei.hdf5'), config=config, mode='eval', global_class_mapping=global_class_mapping)
        print_cpu_mem("After Setting EI Evaluation Dataset")
    else:
        print_cpu_mem("Before Setting Evaluation Dataset")
        eval_dataset = BinauralDataset(config,"evaluation/v01_eval_mit_bldg46room1004_tenoise", config["CACHE_SIZE"]["EVAL"], global_class_mapping)
        print_cpu_mem("After Setting Evaluation Dataset")
    batch_size = config["BATCH_SIZE"]
    num_workers = config["NUM_WORKERS"]

    if config['DEBUG']:
        eval_dataset = Subset(eval_dataset, range(config['DEBUG_CONFIG']['EVAL_DATASET_SIZE']))
        print(f'Debug mode : \n'
              f'- eval dataset into {config["DEBUG_CONFIG"]["EVAL_DATASET_SIZE"]} samples\n')
    return {'eval': DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)}