"""
fig6_plot.py — Effect of onset: 2 × 2 per example.

Layout (per PDF):
  Col 0: ITD (Correlagram)   Col 1: ILD
  Row 0: Whole signal        Row 1: Onset window (50 ms)

Saves one PDF per example to fig6_onset/examples/.
Picks 10 files from binaural_samples/train/.
"""

import os
import sys
import warnings
from glob import glob
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

from models.ducky import BM, IHC, Correlagram, ILD, detect_onset_and_slice
from auditory_layers.cochlear import audspace_bw

FILTER_COEFF_DIR = os.path.join(PROJECT_ROOT, 'models', 'filter_coeff_dir')
TRAIN_DIR        = os.path.join(PROJECT_ROOT, 'binaural_samples', 'train')
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), 'examples')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_EXAMPLES = 10

# ── font ───────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']
TICK_FS      = 7
LABEL_FS     = 8
TITLE_FS     = 9
ROW_LABEL_FS = 9

# ── figure size ─────────────────────────────────────────────────────────────────
FIG_W = 161 / 25.4
FIG_H = 100 / 25.4

# ── peripheral model ────────────────────────────────────────────────────────────
SR    = 44100
FMIN, FMAX, BW, ORDER = 125, 16000, 0.5, 4
IHC_FCUT, IHC_ORDER   = 4000, 1
PHY_ITD_RANGE = [-1.5e-3, 1.5e-3]
ILD_STRIDE    = int(0.005 * SR)    # 220 samples = 5 ms
ONSET_WINDOW  = int(0.050 * SR)    # 2205 samples = 50 ms
ONSET_FRAMES  = ONSET_WINDOW // ILD_STRIDE   # 10 frames

ILD_T_MAX    = 1.0                           # crop whole ILD to 1 s
ILD_FRAMES   = int(ILD_T_MAX * SR / ILD_STRIDE)  # 200 frames

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    bm_model   = BM(fmin=FMIN, fmax=FMAX, bw=BW, sr=SR, order=ORDER,
                    filter_coeff_dir=FILTER_COEFF_DIR, learnable_coefficients=False)
    ihc_model  = IHC(fcut=IHC_FCUT, sr=SR, order=IHC_ORDER,
                     filter_coeff_dir=FILTER_COEFF_DIR)
    corr_model = Correlagram(sr=SR, phy_ITD_range=PHY_ITD_RANGE)
    ild_model  = ILD()

for m in (bm_model, ihc_model, corr_model, ild_model):
    m.eval()

fc, _ = audspace_bw(FMIN, FMAX, BW)
N_FREQ = len(fc)

_TICK_IDX = [int(np.argmin(np.abs(fc - f))) for f in [1000, 8000]]
_TICK_LBL = ['1k', '8k']
_itd_ms   = [PHY_ITD_RANGE[0] * 1e3, PHY_ITD_RANGE[1] * 1e3]

# ── helpers ────────────────────────────────────────────────────────────────────
def run_ihc(audio_mono):
    x = torch.tensor(audio_mono, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return ihc_model(bm_model(x))   # (1, F, T)


def compute_maps(ihc_l, ihc_r):
    """Returns whole_itd, whole_ild, onset_itd, onset_ild — all numpy (F, *)."""
    # ── whole ──
    with torch.no_grad():
        whole_itd = corr_model(ihc_l, ihc_r).squeeze(0).squeeze(0).numpy()  # (F, 133)

    avg_l = F.avg_pool2d(ihc_l.unsqueeze(1),
                         kernel_size=(1, ILD_STRIDE), stride=(1, ILD_STRIDE)).squeeze(1)
    avg_r = F.avg_pool2d(ihc_r.unsqueeze(1),
                         kernel_size=(1, ILD_STRIDE), stride=(1, ILD_STRIDE)).squeeze(1)
    with torch.no_grad():
        whole_ild = ild_model(avg_l, avg_r).squeeze(0).squeeze(0).numpy()   # (F, T_avg)
    whole_ild = whole_ild[:, :ILD_FRAMES]   # crop to 1 s

    # ── onset ──
    x_l_onset, x_r_onset, x_l_avg_onset, x_r_avg_onset = detect_onset_and_slice(
        ihc_l, ihc_r, stride=ILD_STRIDE, t_window=ONSET_WINDOW)

    with torch.no_grad():
        onset_itd = corr_model(x_l_onset, x_r_onset).squeeze(0).squeeze(0).numpy()  # (F, 133)
        onset_ild = ild_model(x_l_avg_onset, x_r_avg_onset).squeeze(0).squeeze(0).numpy()  # (F, 10)

    return whole_itd, whole_ild, onset_itd, onset_ild


def plot_and_save(whole_itd, whole_ild, onset_itd, onset_ild, stem):
    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.12)
    ax  = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    ROW_LABELS = ['Whole', 'Onset']

    for r, (itd_map, ild_map) in enumerate([
            (whole_itd, whole_ild),
            (onset_itd, onset_ild),
    ]):
        # ── col 0: ITD ──
        a = ax[r][0]
        extent_itd = [_itd_ms[0], _itd_ms[1], -0.5, N_FREQ - 0.5]
        a.imshow(itd_map, aspect='auto', origin='lower', extent=extent_itd,
                 cmap='cividis', interpolation='antialiased', rasterized=True)
        a.set_xticks([-1.5, 0.0, 1.5])
        a.set_yticks(_TICK_IDX)
        a.set_yticklabels(_TICK_LBL, fontsize=TICK_FS)
        a.set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
        if r == 1:
            a.set_xticklabels(['-1.5', '0', '1.5'], fontsize=TICK_FS)
            a.set_xlabel('ITD (ms)', fontsize=LABEL_FS)
        else:
            a.tick_params(axis='x', labelbottom=False)
        a.tick_params(labelsize=TICK_FS)
        # row label
        a.set_title(ROW_LABELS[r], fontsize=ROW_LABEL_FS, loc='left', pad=3)

        # ── col 1: ILD ──
        a = ax[r][1]
        if r == 0:   # whole: 0~1 s
            t_max  = ILD_T_MAX
            xticks = [0.0, 0.5, 1.0]
            xlbls  = ['0', '0.5', '1']
            xlabel = 'Time (s)'
        else:        # onset: 0~50 ms
            t_max  = ONSET_FRAMES * ILD_STRIDE / SR * 1e3   # ms
            xticks = [0, 25, 50]
            xlbls  = ['0', '25', '50']
            xlabel = 'Time (ms)'

        extent_ild = [0.0, t_max, -0.5, N_FREQ - 0.5]
        a.imshow(ild_map, aspect='auto', origin='lower', extent=extent_ild,
                 cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                 interpolation='antialiased', rasterized=True)
        a.set_xticks(xticks)
        a.set_yticks(_TICK_IDX)
        a.tick_params(axis='y', labelleft=False)
        if r == 1:
            a.set_xticklabels([str(x) for x in xlbls], fontsize=TICK_FS)
            a.set_xlabel(xlabel, fontsize=LABEL_FS)
        else:
            a.tick_params(axis='x', labelbottom=False)
        a.tick_params(labelsize=TICK_FS)

    # column titles on top row
    ax[0][0].set_title('Correlagram (ITD)', fontsize=TITLE_FS, loc='center', pad=10)
    ax[0][1].set_title('ILD',               fontsize=TITLE_FS, loc='center', pad=10)

    out_path = os.path.join(OUTPUT_DIR, f'{stem}.pdf')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ── pick 10 samples from train ─────────────────────────────────────────────────
all_files = sorted(glob(os.path.join(TRAIN_DIR, 'azim_90_elev_*_snr_*.wav')))[:N_EXAMPLES]

print(f'Processing {len(all_files)} examples …')
for fpath in all_files:
    fname = os.path.basename(fpath)
    stem  = os.path.splitext(fname)[0]
    try:
        audio, sr = sf.read(fpath)
    except Exception as e:
        print(f'  SKIP {fname}: {e}')
        continue

    ihc_l = run_ihc(audio[:, 0])
    ihc_r = run_ihc(audio[:, 1])

    whole_itd, whole_ild, onset_itd, onset_ild = compute_maps(ihc_l, ihc_r)

    out = plot_and_save(whole_itd, whole_ild, onset_itd, onset_ild, stem)
    print(f'  Saved: {os.path.basename(out)}')

print(f'\nDone. PDFs in {OUTPUT_DIR}')
