import os
import yaml
import h5py
from glob import glob
import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':
    # Read config
    config_fpath = './config.yaml'
    with open(config_fpath, 'r') as f:
        config = yaml.safe_load(f)

    ''' (Must) Choose among 'train' / 'val' / 'yang' '''
    mode = 'val'
    print(f'- Chosen Mode : {mode}')
    if mode == 'yang':
        key_excluded = ['signal']
    else:
        key_excluded = ['signal', 'list_background_index_brir']

    # Directory to save data_analysis
    figure_dir = f'data_analysis/{mode}'
    os.makedirs(figure_dir, exist_ok=True)

    # Directory Setting
    dataset_dir = config['DATASET_DIR']
    train_dataset_dir = os.path.join(dataset_dir, 'train')
    val_dataset_dir = os.path.join(dataset_dir, 'valid')
    eval_dataset_dir = os.path.join(dataset_dir, 'evaluation/v01_eval_mit_bldg46room1004_tenoise')

    dataset_dirs = {'train': train_dataset_dir, 'val': val_dataset_dir, 'yang': eval_dataset_dir}
    dataset_dir_to_see = dataset_dirs[mode]

    print(f'- Dataset directory : {dataset_dir_to_see}')

    # h5 Setting
    h5_fpath_list = glob(os.path.join(dataset_dir_to_see, '*.hdf5'))
    # print(f'h5_fpath_list: {h5_fpath_list}')
    h5_fpath = h5_fpath_list[1]

    # Check key and just see one example for each key
    with h5py.File(h5_fpath_list[0]) as f:
        all_keys = list(f.keys())
        print(f'Available keys: {all_keys}')
        print('## Just show example')
        for key in all_keys:
            print(f'- key : {key} / shape : {f[key].shape} / type : {f[key].dtype} / val : {f[key][0]} ')

    # Key dictionary
    key_data = {key: [] for key in all_keys if key not in key_excluded}

    # Collect data for each key
    for h5_fpath in h5_fpath_list:
        with h5py.File(h5_fpath) as f:
            for key in key_data.keys():
                key_data[key].extend(f[key][:])

    # np.unique so useful to see distribution of metadata
    for key, values in key_data.items():
        unique_vals, counts = np.unique(values, return_counts=True)
        print(f'\n{'=' * 50}')
        print(f'Key : {key}')
        print(f'Unique values : {unique_vals}')
        print(f'Total unique count : {counts}')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=unique_vals, y=counts))
        fig.write_html(os.path.join(figure_dir, f'{key}.html'))
