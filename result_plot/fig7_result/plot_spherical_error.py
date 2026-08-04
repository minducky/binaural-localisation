
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "wandb_export_2026-05-20T14_42_59.600+01_00.csv")

SNR_ORDER = [-13.6, -6.8, 0.0, 6.8, 13.6]
SNR_LABELS = ['-13.6', '-6.8', '0', '6.8', '13.6']

def normalize_snr(s):
    v = float(s)
    if np.isinf(v):
        return float('inf')
    return round(v, 1)

df = pd.read_csv(CSV_PATH)
df['snr'] = df['snr'].apply(normalize_snr)
df = df[df['snr'] != float('inf')]
df = df.groupby(['name', 'snr'])['spherical_error'].mean().reset_index()

arch_models  = sorted(n for n in df['name'].unique() if n.startswith('eval_arch'))
yang_models  = sorted(n for n in df['name'].unique() if 'Yang' in n)
human_models = sorted(n for n in df['name'].unique() if n.startswith('human_'))

def model_err(name, snrs):
    sub = df[df['name'] == name].set_index('snr')['spherical_error']
    return np.array([sub.get(s, np.nan) for s in snrs])

def band(names, snrs):
    mat = np.array([model_err(n, snrs) for n in names])
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    return mean - std, mean, mean + std

ymin = max(0.0, np.floor(df['spherical_error'].min() / 5) * 5 - 2)
ymax = np.ceil(df['spherical_error'].max() / 5) * 5 + 2

C_ARCH  = '#999999'   # gray
C_YANG  = '#D55E00'   # vermillion
C_HUMAN = '#CC79A7'   # pinkish purple
C_ITD   = '#56B4E9'   # sky blue
C_ILD   = '#E69F00'   # amber
C_BOTH  = '#222222'   # near-black

ALPHA_BAND = 0.25
LW = 2.0

fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
fig.subplots_adjust(wspace=0.08)

x5 = np.arange(len(SNR_ORDER))

def _setup(ax, title):
    ax.set_xticks(x5)
    ax.set_xticklabels(SNR_LABELS, fontsize=9)
    ax.set_xlim(-0.5, x5[-1] + 0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('SNR (dB)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.grid(True, alpha=0.25, linestyle='--')

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 1 – Baselines
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_ylabel('Spherical Error (°)', fontsize=10)
_setup(ax, 'Baselines')

lo, mid, hi = band(arch_models, SNR_ORDER)
ax.fill_between(x5, lo, hi, color=C_ARCH, alpha=ALPHA_BAND)
ax.plot(x5, mid, color=C_ARCH, lw=LW, label='Saddler (arch)')

lo, mid, hi = band(yang_models, SNR_ORDER)
ax.fill_between(x5, lo, hi, color=C_YANG, alpha=ALPHA_BAND)
ax.plot(x5, mid, color=C_YANG, lw=LW, label='Yang 2025')

lo, mid, hi = band(human_models, SNR_ORDER)
ax.fill_between(x5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x5, mid, color=C_HUMAN, lw=LW, label='Human')

ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 2 – Whole / No Onset
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
_setup(ax, 'Whole (No Onset)')

lo, mid, hi = band(human_models, SNR_ORDER)
ax.fill_between(x5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x5, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_Ducky_ITDOnly_NoOnset', C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_NoOnset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ItdIld_Whole',    C_BOTH, 'ITD + ILD'),
]:
    err = model_err(name, SNR_ORDER)
    ax.plot(x5, err, color=color, lw=LW, marker='o', ms=5, label=label)

ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 3 – Onset
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
_setup(ax, 'Onset')

lo, mid, hi = band(human_models, SNR_ORDER)
ax.fill_between(x5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x5, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_DuckyITDOnly_Onset',  C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_Onset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_Onset',  C_BOTH, 'ITD + ILD'),
]:
    err = model_err(name, SNR_ORDER)
    ax.plot(x5, err, color=color, lw=LW, marker='o', ms=5, label=label)

ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 4 – MultiScale
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[3]
_setup(ax, 'MultiScale')

lo, mid, hi = band(human_models, SNR_ORDER)
ax.fill_between(x5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x5, mid, color=C_HUMAN, lw=LW, label='Human')

err = model_err('exp0_Ducky_ITDILD_MultiScale', SNR_ORDER)
ax.plot(x5, err, color=C_BOTH, lw=LW, marker='o', ms=5, label='ITD + ILD')

ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

out_path = os.path.join(os.path.dirname(__file__), 'result_plot_spherical_error.pdf')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()