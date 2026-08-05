"""
fig3_plot.py — Peripheral processing visualization.

Layout: 2 rows × 3 columns (Left / Right).
  Col 0: raw binaural waveform
  Col 1: basilar membrane output  — raw BM               [cividis]
  Col 2: inner hair cell output   — relu + sqrt + LP      [cividis]

Figure width: 161 mm (single-column paper width).
Saves 10 PDFs (5 × azim=90, 5 × azim=270) to fig3_peripheral/example/.
"""

import os
from glob import glob
import sys
import warnings
import numpy as np
import torch
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

FILTER_COEFF_DIR = os.path.join(PROJECT_ROOT, 'models', 'filter_coeff_dir')
SAMPLES_DIR      = os.path.join(PROJECT_ROOT, 'binaural_samples', 'train')

OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), 'example')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── font ───────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']
TICK_FS       = 7
LABEL_FS      = 8
TITLE_FS      = 9
ROW_LABEL_FS  = 9   # Left / Right row labels

# ── figure size ────────────────────────────────────────────────────────────────
FIG_W = 161 / 25.4
FIG_H =  80 / 25.4

# ── waveform colours ──────────────────────────────────────────────────────────
COLOR_L = '#1f77b4'   # blue  — Left
COLOR_R = '#ff7f0e'   # orange — Right

# ── peripheral model ───────────────────────────────────────────────────────────
SR    = 44100
FMIN, FMAX, BW, ORDER = 125, 16000, 0.5, 4
IHC_FCUT, IHC_ORDER   = 4000, 1

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    bm_model  = BM(fmin=FMIN, fmax=FMAX, bw=BW, sr=SR, order=ORDER,
                   filter_coeff_dir=FILTER_COEFF_DIR, learnable_coefficients=False)
    ihc_model = IHC(fcut=IHC_FCUT, sr=SR, order=IHC_ORDER, filter_coeff_dir=FILTER_COEFF_DIR)

bm_model.eval()
ihc_model.eval()

fc, _ = audspace_bw(FMIN, FMAX, BW)   # (71,) center frequencies in Hz
N_FREQ = len(fc)

# frequency tick positions (bin indices) and labels
_TICK_IDX = [int(np.argmin(np.abs(fc - f))) for f in [1000, 8000]]
_TICK_LBL = ['1k', '8k']

# ── IHC detail options ────────────────────────────────────────────────────────
IHC_DETAIL       = True
IHC_DETAIL_FC    = 500    # Hz  — target frequency band
IHC_DETAIL_T_CTR = 0.27  # sec — center of zoom window
IHC_DETAIL_T_WIN = 0.2   # sec — total window duration

# ── samples ────────────────────────────────────────────────────────────────────
sample_list = []
for azim in [90]:
    f_query = 'azim_90_elev_30_sample_0_snr_21.1_if_2609.wav'
    # f_query = f'azim_{azim}_elev_*_sample_*_snr_*.wav'
    fpath_list = glob(os.path.join(SAMPLES_DIR, f_query))

print(fpath_list)
print(len(fpath_list))
sample_list = fpath_list
# sample_list = fpath_list[10:20]


# ── helpers ───────────────────────────────────────────────────────────────────
def run_peripheral(audio_mono):
    """(T,) → bm_raw (F,T) float32,  ihc_out (F,T) float32"""
    x = torch.tensor(audio_mono, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        bm_raw  = bm_model(x)
        ihc_out = ihc_model(bm_raw)
    return bm_raw.squeeze(0).numpy(), ihc_out.squeeze(0).numpy()


def pct(arr, lo=1.0, hi=99.0):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


# ── main loop ─────────────────────────────────────────────────────────────────
for fpath in sample_list:
    audio, sr = sf.read(fpath)
    assert sr == SR
    T_samp = audio.shape[0]

    bm_l, ihc_l = run_peripheral(audio[:, 0])
    bm_r, ihc_r = run_peripheral(audio[:, 1])

    # ── IHC detail: 1D zoom on a single frequency band ───────────────────────
    if IHC_DETAIL:
        fc_idx  = int(np.argmin(np.abs(fc - IHC_DETAIL_FC)))
        t0 = max(0.0, IHC_DETAIL_T_CTR - IHC_DETAIL_T_WIN / 2)
        t1 = min(T_samp / SR, IHC_DETAIL_T_CTR + IHC_DETAIL_T_WIN / 2)
        s0, s1  = int(t0 * SR), int(t1 * SR)
        sig     = ihc_r[fc_idx, s0:s1]
        t_sig   = np.arange(len(sig)) / SR + t0

        print(f'IHC detail: fc[{fc_idx}] = {fc[fc_idx]:.1f} Hz,  '
              f't = [{t0:.3f}, {t1:.3f}] s  ({len(sig)} samples)')

        fig_d, ax_d = plt.subplots(figsize=(4, 1.5))
        ax_d.plot(t_sig, sig, color='black', lw=0.3)
        ax_d.axis('off')
        fig_d.tight_layout(pad=0.1)

        stem     = os.path.splitext(os.path.basename(fpath))[0]
        out_det  = os.path.join(OUTPUT_DIR, f'{stem}_ihc_detail_{int(IHC_DETAIL_FC)}Hz.pdf')
        fig_d.savefig(out_det, bbox_inches='tight', dpi=300)
        plt.close(fig_d)
        print(f'Saved IHC detail → {out_det}')

    #
    # # shared color limits (L and R same scale for fair comparison)
    # bm_lo,  bm_hi  = pct(np.stack([bm_l,  bm_r]),  lo=1,  hi=99)
    # ihc_lo, ihc_hi = pct(np.stack([ihc_l, ihc_r]), lo=1,  hi=99)
    #
    # t      = np.arange(T_samp) / SR
    # extent = [0.0, T_samp / SR, -0.5, N_FREQ - 0.5]
    #
    # # ── figure ────────────────────────────────────────────────────────────────
    # fig = plt.figure(figsize=(FIG_W, FIG_H))
    # gs  = gridspec.GridSpec(2, 3, figure=fig)
    # ax  = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    #
    # # ── col 0: waveform ───────────────────────────────────────────────────────
    # for r, (ch, color) in enumerate([(0, COLOR_L), (1, COLOR_R)]):
    #     ax[r][0].plot(t, audio[:, ch], color=color, lw=0.5)
    #     ax[r][0].set_xlim(0.0, T_samp / SR)
    #     ax[r][0].set_ylim(-1.05, 1.05)
    #     ax[r][0].set_yticks([-1.0, 0.0, 1.0])
    #     ax[r][0].set_yticklabels(['-1', '0', '1'], fontsize=TICK_FS)
    #     ax[r][0].tick_params(labelsize=TICK_FS)
    #
    # # ── col 1: BM (raw) — imshow ─────────────────────────────────────────────
    # for r, bm_ch in enumerate([bm_l, bm_r]):
    #     ax[r][1].imshow(bm_ch, aspect='auto', origin='lower', extent=extent,
    #                     cmap='cividis', vmin=bm_lo, vmax=bm_hi)
    #     ax[r][1].set_ylim(-0.5, N_FREQ - 0.5)
    #     ax[r][1].set_yticks(_TICK_IDX)
    #     ax[r][1].set_yticklabels(_TICK_LBL, fontsize=TICK_FS)
    #     ax[r][1].tick_params(labelsize=TICK_FS)
    #
    # # ── col 2: IHC (raw) — imshow ─────────────────────────────────────────────
    # for r, ihc_ch in enumerate([ihc_l, ihc_r]):
    #     ax[r][2].imshow(ihc_ch, aspect='auto', origin='lower', extent=extent,
    #                     cmap='cividis', vmin=ihc_lo, vmax=ihc_hi)
    #     ax[r][2].set_ylim(-0.5, N_FREQ - 0.5)
    #     ax[r][2].set_yticks(_TICK_IDX)
    #     # hide y-axis tick labels (same as col 1) — tick_params, NOT set_yticklabels([])
    #     # ax[r][2].tick_params(axis='y', labelleft=False, labelsize=TICK_FS)
    #     ax[r][2].set_yticklabels(_TICK_LBL, fontsize=TICK_FS)
    #
    # # ── x-axis ticks: 0.0, 0.5, 1.0 on all columns ───────────────────────────
    # for r in range(2):
    #     for c in range(3):
    #         ax[r][c].set_xlim(0.0, T_samp / SR)
    #         ax[r][c].set_xticks([0.0, 0.5, 1.0])
    #         if r == 0:
    #             ax[r][c].tick_params(axis='x', labelbottom=False)
    #         else:
    #             ax[r][c].set_xticklabels(['0.0', '0.5', '1.0'], fontsize=TICK_FS)
    #
    # # ── axis labels (bottom row) ───────────────────────────────────────────────
    # for c in range(3):
    #     ax[1][c].set_xlabel('Time (s)', fontsize=LABEL_FS)
    #
    # ax[0][2].set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
    # ax[1][2].set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
    # ax[0][1].set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
    # ax[1][1].set_ylabel('Frequency (Hz)', fontsize=LABEL_FS)
    # ax[0][0].set_ylabel('Amplitude', fontsize=LABEL_FS)
    # ax[1][0].set_ylabel('Amplitude', fontsize=LABEL_FS)
    #
    # # ── Left / Right row labels (loc='left') + column titles (loc='center') ───
    # # Using different loc values so both titles coexist in the same title line
    # # ax[0][0].set_title('Left',  fontsize=ROW_LABEL_FS, loc='left',   pad=3)
    # # ax[1][0].set_title('Right', fontsize=ROW_LABEL_FS, loc='left',   pad=3)
    #
    # ax[0][0].set_title('Binaural Audio',   fontsize=TITLE_FS, loc='center', pad=5)
    # ax[0][1].set_title('Basilar Membrane', fontsize=TITLE_FS, loc='center', pad=5)
    # ax[0][2].set_title('Inner Hair Cell',  fontsize=TITLE_FS, loc='center', pad=5)
    #
    # # ── layout & save ──────────────────────────────────────────────────────────
    # fig.tight_layout(pad=0.4, h_pad=0.3, w_pad=0.5)
    # fname = os.path.basename(fpath)
    # out_path = os.path.join(OUTPUT_DIR, f'{fname}.pdf')
    # fig.savefig(out_path, bbox_inches='tight', dpi=300)
    # plt.close(fig)
    # print(f'Saved: {out_path}.pdf')

print(f'\nDone. {len(sample_list)} PDFs in {OUTPUT_DIR}')
