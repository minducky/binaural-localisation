"""
fig7_spherical.py — Spherical error plots.

Saves two PDFs (both 161 mm wide):
  fig7_spherical_baseline.pdf  — 1 subplot: arch / yang / human bands
  fig7_spherical_ducky.pdf     — 3 subplots: Ducky models only (whole, onset, multiscale)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DIR      = os.path.join(os.path.dirname(__file__), 'eval_result')
OUT_DIR  = os.path.dirname(__file__)

# ── font ───────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']
TICK_FS  = 7
LABEL_FS = 8
TITLE_FS = 9

FIG_W = 161 / 25.4

# ── colours ────────────────────────────────────────────────────────────────────
C_ARCH  = '#999999'
C_YANG  = '#D55E00'
C_HUMAN = '#CC79A7'
C_ITD   = '#56B4E9'
C_ILD   = '#E69F00'
C_BOTH  = '#009E73'   # teal for ITD+ILD
ALPHA_BAND = 0.25
LW = 1.5

# ── data ───────────────────────────────────────────────────────────────────────
SNR_ORDER  = [-13.6, -6.8, 0.0, 6.8, 13.6]
SNR_LABELS = ['-13.6', '-6.8', '0', '6.8', '13.6']

def normalize_snr(s):
    if str(s) in ('Infinity', 'inf', 'Inf'): return float('inf')
    return round(float(s), 1)

err_df = pd.read_csv(os.path.join(DIR, 'Spherical_error.csv'))
err_df['snr'] = err_df['snr'].apply(normalize_snr)
pivot = err_df.groupby(['name', 'snr'])['spherical_error'].mean().unstack('snr')

def model_err(name):
    return np.array([pivot.loc[name, s] if s in pivot.columns else np.nan
                     for s in SNR_ORDER])

def band(names):
    mat = np.array([model_err(n) for n in names if n in pivot.index])
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    return mean - std, mean, mean + std

# ── model groups ───────────────────────────────────────────────────────────────
arch_models  = [f'eval_arch{i:02d}' for i in range(1, 11)]
yang_models  = ['exp0_Yang2025_lr1e-3', 'exp0_Yang2025_lr1e-3_2', 'exp0_Yang2025_lr1e-3_3',
                'exp0_Yang2025_lr1e-4', 'exp0_Yang2025_lr5e-4']   # exclude lr1e-5
human_models = sorted(n for n in pivot.index if n.startswith('human_'))

DUCKY_WHOLE = [
    ('exp0_Ducky_ITDOnly_NoOnset', C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_NoOnset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ItdIld_Whole',    C_BOTH, 'ITD + ILD'),
]
DUCKY_ONSET = [
    ('exp0_DuckyITDOnly_Onset',  C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_Onset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_Onset',  C_BOTH, 'ITD + ILD'),
]
DUCKY_MULTI = [
    ('exp0_Ducky_ItdMultiscale',      C_ITD,  'ITD only'),
    ('exp0_Ducky_IldMultiscale',      C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_MultiScale',  C_BOTH, 'ITD + ILD'),
]

x = np.arange(len(SNR_ORDER))

# ── shared y limits ────────────────────────────────────────────────────────────
all_vals = pivot[SNR_ORDER].values.flatten()
ymin = max(0, np.nanmin(all_vals) - 2)
ymax = np.nanmax(all_vals) + 2

def _setup(ax, title):
    ax.set_xticks(x)
    ax.set_xticklabels(SNR_LABELS, fontsize=TICK_FS)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('SNR (dB)', fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=TICK_FS)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Baselines  (1 subplot, 161 mm)
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(1, 1, figsize=(FIG_W, 70/25.4), constrained_layout=True)

_setup(ax1, 'Baselines')
ax1.set_ylabel('Spherical Error (°)', fontsize=LABEL_FS)

lo, mid, hi = band(arch_models)
ax1.fill_between(x, lo, hi, color=C_ARCH, alpha=ALPHA_BAND)
ax1.plot(x, mid, color=C_ARCH, lw=LW, label='Saddler (arch)')

lo, mid, hi = band(yang_models)
ax1.fill_between(x, lo, hi, color=C_YANG, alpha=ALPHA_BAND)
ax1.plot(x, mid, color=C_YANG, lw=LW, label='Yang 2025')

lo, mid, hi = band(human_models)
ax1.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax1.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

err = model_err('exp0_Ducky_ITDILD_MultiScale')
ax1.plot(x, err, color=C_BOTH, lw=LW, marker='o', ms=3, label='Ducky ITD+ILD MultiScale')

ax1.legend(fontsize=TICK_FS, loc='upper left', framealpha=0.8)

out1 = os.path.join(OUT_DIR, 'fig7_spherical_baseline.pdf')
fig1.savefig(out1, dpi=300)
plt.close(fig1)
print(f'Saved → {out1}')

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Ducky models  (3 subplots, 161 mm)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(FIG_W, 70/25.4),
                          constrained_layout=True, sharey=True)

for ax, title, ducky_list in zip(
        axes,
        ['Whole / No Onset', 'Onset', 'MultiScale'],
        [DUCKY_WHOLE, DUCKY_ONSET, DUCKY_MULTI],
):
    _setup(ax, title)
    for name, color, label in ducky_list:
        if name not in pivot.index:
            print(f'  WARNING: {name} not found in data')
            continue
        err = model_err(name)
        ax.plot(x, err, color=color, lw=LW, marker='o', ms=3, label=label)
    ax.legend(fontsize=TICK_FS, loc='upper left', framealpha=0.8)

axes[0].set_ylabel('Spherical Error (°)', fontsize=LABEL_FS)

out2 = os.path.join(OUT_DIR, 'fig7_spherical_ducky.pdf')
fig2.savefig(out2, dpi=300)
plt.close(fig2)
print(f'Saved → {out2}')
