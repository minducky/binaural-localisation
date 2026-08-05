"""
fig4_composite.py — 3 × 2 composite: ITD and ILD maps.

Rows:   azim=90°  /  azim=0°  /  azim=270° (-90°)
Cols:   ITD (Correlagram)  /  ILD map

Figure width: 161 mm (single-column paper).
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.ducky import BM, IHC
from auditory_layers.cochlear import audspace_bw
from auditory_layers.midbrain import Correlagram, ILD, ILDNormaliser

FILTER_COEFF_DIR = os.path.join(PROJECT_ROOT, 'models', 'filter_coeff_dir')
TRAIN_DIR        = os.path.join(PROJECT_ROOT, 'binaural_samples', 'train')
OUTPUT_DIR       = os.path.dirname(__file__)

# ── font ───────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']
TICK_FS      = 7
LABEL_FS     = 8
TITLE_FS     = 9
ROW_LABEL_FS = 9

# ── figure size ─────────────────────────────────────────────────────────────────
FIG_W = 161 / 25.4
FIG_H = 112 / 25.4

# ── peripheral model ────────────────────────────────────────────────────────────
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
    corr_model = Correlagram(sr_res=SR, phy_itd_range=PHY_ITD_RANGE, normalise=True)
    ild_model  = ILD()
    ild_normaliser = ILDNormaliser()

for m in (bm_model, ihc_model, corr_model, ild_model):
    m.eval()

fc, _ = audspace_bw(FMIN, FMAX, BW)
N_FREQ = len(fc)

_TICK_IDX = [int(np.argmin(np.abs(fc - f))) for f in [1000, 8000]]
_TICK_LBL = ['1k', '8k']
_itd_ms   = [PHY_ITD_RANGE[0] * 1e3, PHY_ITD_RANGE[1] * 1e3]

# ── selected examples ──────────────────────────────────────────────────────────
# (row, col) : filename
EXAMPLES = {
    (0, 0): 'azim_90_elev_30_sample_0_snr_21.1_if_2609.wav',
    (1, 0): 'azim_0_elev_60_sample_3_snr_inf_if_13600.wav',
    (2, 0): 'azim_270_elev_30_sample_0_snr_9.6_if_13831.wav',
    (0, 1): 'azim_90_elev_30_sample_0_snr_21.1_if_2609.wav',
    (1, 1): 'azim_0_elev_30_sample_2_snr_-12.1_if_7885.wav',
    (2, 1): 'azim_270_elev_30_sample_0_snr_9.6_if_13831.wav',
}
ROW_LABELS = ['90°', '0°', '−90°']

# ── helpers ────────────────────────────────────────────────────────────────────
def run_ihc(audio_mono):
    x = torch.tensor(audio_mono, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return ihc_model(bm_model(x))   # (1, F, T)


def get_itd(ihc_l, ihc_r):
    with torch.no_grad():
        corr = corr_model(ihc_l, ihc_r)   # (1, F, ITD_bins)
    return corr.squeeze(0).numpy()   # (F, ITD_bins)


def get_ild(ihc_l, ihc_r):
    avg_l = F.avg_pool2d(ihc_l.unsqueeze(1),
                         kernel_size=(1, ILD_STRIDE), stride=(1, ILD_STRIDE)).squeeze(1)
    avg_r = F.avg_pool2d(ihc_r.unsqueeze(1),
                         kernel_size=(1, ILD_STRIDE), stride=(1, ILD_STRIDE)).squeeze(1)
    with torch.no_grad():
        ild = ild_normaliser(ild_model(avg_l, avg_r))   # (1, F, T_avg)
    return ild.squeeze(0).numpy()   # (F, T_avg)


# ── precompute maps ────────────────────────────────────────────────────────────
_cache = {}
maps = {}   # (row, col) → numpy array

for (row, col), fname in EXAMPLES.items():
    fpath = os.path.join(TRAIN_DIR, fname)
    if fname not in _cache:
        audio, _ = sf.read(fpath)
        ihc_l = run_ihc(audio[:, 0])
        ihc_r = run_ihc(audio[:, 1])
        T_samp = audio.shape[0]
        _cache[fname] = (ihc_l, ihc_r, T_samp)
        print(f'Loaded: {fname}')
    ihc_l, ihc_r, T_samp = _cache[fname]
    if col == 0:
        maps[(row, col)] = (get_itd(ihc_l, ihc_r), T_samp)
    else:
        maps[(row, col)] = (get_ild(ihc_l, ihc_r), T_samp)

# ── figure ─────────────────────────────────────────────────────────────────────
ILD_T_MAX   = 1.0                          # crop ILD to 1 s
ILD_FRAMES  = int(ILD_T_MAX * SR / ILD_STRIDE)   # 200 frames

fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=True)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.28, wspace=0.12)
ax  = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

for r in range(3):
    for c in range(2):
        data, T_samp = maps[(r, c)]
        a = ax[r][c]

        if c == 0:  # ITD
            extent = [_itd_ms[0], _itd_ms[1], -0.5, N_FREQ - 0.5]
            a.imshow(data, aspect='auto', origin='lower', extent=extent,
                     cmap='cividis', interpolation='antialiased', rasterized=True)
            a.set_xticks([-1.5, 0.0, 1.5])
            if r == 2:
                a.set_xticklabels(['-1.5', '0', '1.5'], fontsize=TICK_FS)
                a.set_xlabel('ITD (ms)', fontsize=LABEL_FS)
            else:
                a.tick_params(axis='x', labelbottom=False)
        else:  # ILD — crop to 1 s
            data_crop = data[:, :ILD_FRAMES]
            extent = [0.0, ILD_T_MAX, -0.5, N_FREQ - 0.5]
            a.imshow(data_crop, aspect='auto', origin='lower', extent=extent,
                     cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                     interpolation='antialiased', rasterized=True)
            a.set_xticks([0.0, 0.5, 1.0])
            if r == 2:
                a.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=TICK_FS)
                a.set_xlabel('Time (s)', fontsize=LABEL_FS)
            else:
                a.tick_params(axis='x', labelbottom=False)

        # y-axis
        a.set_yticks(_TICK_IDX)
        if c == 0:
            a.set_yticklabels(_TICK_LBL, fontsize=TICK_FS)
            a.set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
        else:
            a.tick_params(axis='y', labelleft=False)

        a.tick_params(labelsize=TICK_FS)

# column titles on top row
ax[0][0].set_title('Correlagram (ITD)', fontsize=TITLE_FS, loc='center', pad=5)
ax[0][1].set_title('ILD',               fontsize=TITLE_FS, loc='center', pad=5)

# ── save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'fig4_composite.pdf')
fig.savefig(out_path, dpi=300)
print(f'\nSaved → {out_path}')
plt.close(fig)
