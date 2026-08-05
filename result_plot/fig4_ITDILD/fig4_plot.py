"""
fig4_plot.py — ITD (Correlagram) and ILD map visualization.

Scans binaural_samples/eval/ for azim=90, 0, 270 (all elevations, all SNRs).
Saves individual PDFs:
  ITD90/   ILD90/   — azim 90°
  ITD0/    ILD0/    — azim 0°
  ITD-90/  ILD-90/  — azim 270° (labeled as -90°)

Each PDF: single heatmap (F × ITD_bins or F × T_avg).
Colormap: cividis for ITD, RdBu_r for ILD.
"""

import os
import re
from glob import glob
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.ducky import BM, IHC, Correlagram, ILD
from auditory_layers.cochlear import audspace_bw

FILTER_COEFF_DIR = os.path.join(PROJECT_ROOT, 'models', 'filter_coeff_dir')
EVAL_DIR         = os.path.join(PROJECT_ROOT, 'binaural_samples', 'train')
OUTPUT_DIR       = os.path.dirname(__file__)

# ── font ───────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']
TICK_FS  = 7
LABEL_FS = 8
TITLE_FS = 9

# ── figure size (single heatmap per PDF) ───────────────────────────────────────
FIG_W = 80 / 25.4   # mm → inches
FIG_H = 65 / 25.4

# ── peripheral model ───────────────────────────────────────────────────────────
SR    = 44100
FMIN, FMAX, BW, ORDER = 125, 16000, 0.5, 4
IHC_FCUT, IHC_ORDER   = 4000, 1

PHY_ITD_RANGE = [-1.5e-3, 1.5e-3]
ILD_STRIDE    = int(0.005 * SR)   # 220 samples = 5 ms

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    bm_model   = BM(fmin=FMIN, fmax=FMAX, bw=BW, sr=SR, order=ORDER,
                    filter_coeff_dir=FILTER_COEFF_DIR, learnable_coefficients=False)
    ihc_model  = IHC(fcut=IHC_FCUT, sr=SR, order=IHC_ORDER,
                     filter_coeff_dir=FILTER_COEFF_DIR)
    corr_model = Correlagram(sr=SR, phy_ITD_range=PHY_ITD_RANGE)
    ild_model  = ILD()

bm_model.eval()
ihc_model.eval()
corr_model.eval()
ild_model.eval()

fc, _ = audspace_bw(FMIN, FMAX, BW)   # (71,) center frequencies in Hz
N_FREQ = len(fc)

# frequency tick positions (bin indices) and labels
_TICK_IDX = [int(np.argmin(np.abs(fc - f))) for f in [1000, 8000]]
_TICK_LBL = ['1k', '8k']

# ITD axis: -1.5 ms to +1.5 ms, 133 bins
_itd_range_ms = [PHY_ITD_RANGE[0] * 1e3, PHY_ITD_RANGE[1] * 1e3]

# ── output subdirectories ──────────────────────────────────────────────────────
SUBDIRS = {
    'ITD90':  os.path.join(OUTPUT_DIR, 'ITD90'),
    'ITD0':   os.path.join(OUTPUT_DIR, 'ITD0'),
    'ITD-90': os.path.join(OUTPUT_DIR, 'ITD-90'),
    'ILD90':  os.path.join(OUTPUT_DIR, 'ILD90'),
    'ILD0':   os.path.join(OUTPUT_DIR, 'ILD0'),
    'ILD-90': os.path.join(OUTPUT_DIR, 'ILD-90'),
}
for d in SUBDIRS.values():
    os.makedirs(d, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def run_ihc(audio_mono):
    """(T,) → ihc_out (1, F, T) torch tensor"""
    x = torch.tensor(audio_mono, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        bm  = bm_model(x)
        ihc = ihc_model(bm)
    return ihc   # (1, F, T)


def compute_itd(ihc_l, ihc_r):
    """(1, F, T), (1, F, T) → (F, ITD_bins) numpy"""
    with torch.no_grad():
        corr = corr_model(ihc_l, ihc_r)   # (1, 1, F, ITD_bins)
    return corr.squeeze(0).squeeze(0).numpy()   # (F, ITD_bins)


def compute_ild(ihc_l, ihc_r):
    """(1, F, T), (1, F, T) → (F, T_avg) numpy"""
    avg_l = F.avg_pool2d(ihc_l.unsqueeze(1),
                         kernel_size=(1, ILD_STRIDE),
                         stride=(1, ILD_STRIDE)).squeeze(1)
    avg_r = F.avg_pool2d(ihc_r.unsqueeze(1),
                         kernel_size=(1, ILD_STRIDE),
                         stride=(1, ILD_STRIDE)).squeeze(1)
    with torch.no_grad():
        ild = ild_model(avg_l, avg_r)   # (1, 1, F, T_avg)
    return ild.squeeze(0).squeeze(0).numpy()   # (F, T_avg)


def save_itd_plot(itd_map, stem, out_dir, azim_label):
    """itd_map: (F, ITD_bins). Saves PDF to out_dir/stem.pdf"""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    extent = [_itd_range_ms[0], _itd_range_ms[1], -0.5, N_FREQ - 0.5]
    im = ax.imshow(itd_map, aspect='auto', origin='lower', extent=extent,
                   cmap='cividis', interpolation='antialiased',
                   rasterized=True)

    ax.set_yticks(_TICK_IDX)
    ax.set_yticklabels(_TICK_LBL, fontsi데ze=TICK_FS)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_xticklabels(['-1', '0', '1'], fontsize=TICK_FS)
    ax.set_xlabel('ITD (ms)', fontsize=LABEL_FS)
    ax.set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
    ax.set_title(f'ITD  azim={azim_label}', fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)

    fig.tight_layout(pad=0.4)
    out_path = os.path.join(out_dir, f'{stem}.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return out_path


def save_ild_plot(ild_map, stem, out_dir, azim_label, T_samp):
    """ild_map: (F, T_avg). Saves PDF to out_dir/stem.pdf"""
    T_avg = ild_map.shape[1]
    t_max = T_samp / SR

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    extent = [0.0, t_max, -0.5, N_FREQ - 0.5]
    im = ax.imshow(ild_map, aspect='auto', origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                   interpolation='antialiased', rasterized=True)

    ax.set_yticks(_TICK_IDX)
    ax.set_yticklabels(_TICK_LBL, fontsize=TICK_FS)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=TICK_FS)
    ax.set_xlabel('Time (s)', fontsize=LABEL_FS)
    ax.set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
    ax.set_title(f'ILD  azim={azim_label}', fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)

    fig.tight_layout(pad=0.4)
    out_path = os.path.join(out_dir, f'{stem}.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return out_path


# ── scan and process ──────────────────────────────────────────────────────────
TARGETS = {
    0:   ('ITD0',   'ILD0',   '0°'),
    90:  ('ITD90',  'ILD90',  '90°'),
    270: ('ITD-90', 'ILD-90', '-90°'),
}

_pat = re.compile(r'azim_(\d+)_')

counts = {azim: 0 for azim in TARGETS}

all_files = sorted(glob(os.path.join(EVAL_DIR, 'azim_*.wav')))
print(f'Total eval files: {len(all_files)}')

for fpath in all_files:
    fname = os.path.basename(fpath)
    m = _pat.search(fname)
    if not m:
        continue
    azim = int(m.group(1))
    if azim not in TARGETS:
        continue

    itd_dir_key, ild_dir_key, label = TARGETS[azim]
    stem = os.path.splitext(fname)[0]

    try:
        audio, sr = sf.read(fpath)
    except Exception as e:
        print(f'  SKIP {fname}: {e}')
        continue
    if sr != SR or audio.ndim < 2:
        print(f'  SKIP {fname}: unexpected format (sr={sr}, shape={audio.shape})')
        continue
    T_samp = audio.shape[0]

    ihc_l = run_ihc(audio[:, 0])   # (1, F, T)
    ihc_r = run_ihc(audio[:, 1])

    itd_map = compute_itd(ihc_l, ihc_r)   # (F, 133)
    ild_map = compute_ild(ihc_l, ihc_r)   # (F, T_avg)

    p_itd = save_itd_plot(itd_map, stem, SUBDIRS[itd_dir_key], label)
    p_ild = save_ild_plot(ild_map, stem, SUBDIRS[ild_dir_key], label, T_samp)

    counts[azim] += 1
    if counts[azim] % 10 == 0:
        print(f'  azim={label}: {counts[azim]} files processed …')

print('\n── Done ──')
for azim, (itd_k, ild_k, label) in TARGETS.items():
    print(f'  azim={label}: {counts[azim]} files → {SUBDIRS[itd_k]}  /  {SUBDIRS[ild_k]}')
