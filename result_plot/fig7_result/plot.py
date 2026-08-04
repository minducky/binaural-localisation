
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "wandb_export_2026-05-19T17_35_00.153+01_00.csv")

SNR_ORDER = [-13.6, -6.8, 0.0, 6.8, 13.6, float('inf')]
SNR_LABELS = ['-13.6', '-6.8', '0', '6.8', '13.6', '∞']

def normalize_snr(s):
    if s == 'Infinity':
        return float('inf')
    return round(float(s), 1)

df = pd.read_csv(CSV_PATH)
df['snr'] = df['snr'].apply(normalize_snr)
df = df.groupby(['name', 'snr'])['combined_accuracy'].mean().reset_index()

arch_models  = sorted(n for n in df['name'].unique() if n.startswith('eval_arch'))
yang_models  = sorted(n for n in df['name'].unique() if 'Yang' in n)
human_models = sorted(n for n in df['name'].unique() if n.startswith('human_'))

def model_acc(name, snrs):
    sub = df[df['name'] == name].set_index('snr')['combined_accuracy']
    return np.array([sub.get(s, np.nan) for s in snrs])

def band(names, snrs):
    mat = np.array([model_acc(n, snrs) for n in names])
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    return mean - std, mean, mean + std

# ── y-axis range (same for all subplots) ──────────────────────────────────────
ymin = max(0.0, np.floor(df['combined_accuracy'].min() * 20) / 20 - 0.02)
ymax = min(1.0, np.ceil(df['combined_accuracy'].max() * 20) / 20 + 0.02)

# ── colour palette ─────────────────────────────────────────────────────────────
C_ARCH  = '#999999'   # gray            – Saddler arch baselines
C_YANG  = '#D55E00'   # vermillion      – Yang baselines
C_HUMAN = '#CC79A7'   # pinkish purple  – Human baselines
C_ITD   = '#56B4E9'   # sky blue        – ITD only
C_ILD   = '#E69F00'   # amber           – ILD only
C_BOTH  = '#222222'   # near-black      – ITD + ILD

ALPHA_BAND = 0.25
LW = 2.0

fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
fig.subplots_adjust(wspace=0.08)

x_full = np.arange(len(SNR_ORDER))          # 6 ticks (includes ∞)
x_5    = np.arange(len(SNR_ORDER) - 1)      # 5 ticks (no ∞ for model results)

# ── helper: set up an axis ────────────────────────────────────────────────────
def _setup(ax, title, include_inf=True):
    xs = x_full if include_inf else x_5
    labels = SNR_LABELS if include_inf else SNR_LABELS[:-1]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('SNR (dB)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.25, linestyle='--')

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 1 – Baselines
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_ylabel('Accuracy', fontsize=10)
_setup(ax, 'Baselines', include_inf=False)

lo, mid, hi = band(arch_models, SNR_ORDER[:-1])
ax.fill_between(x_5, lo, hi, color=C_ARCH, alpha=ALPHA_BAND)
ax.plot(x_5, mid, color=C_ARCH, lw=LW, label='Saddler (arch)')

lo, mid, hi = band(yang_models, SNR_ORDER[:-1])
ax.fill_between(x_5, lo, hi, color=C_YANG, alpha=ALPHA_BAND)
ax.plot(x_5, mid, color=C_YANG, lw=LW, label='Yang 2025')

lo, mid, hi = band(human_models, SNR_ORDER[:-1])
ax.fill_between(x_5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x_5, mid, color=C_HUMAN, lw=LW, label='Human')

ax.legend(fontsize=8, loc='upper left', framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 2 – Whole / No Onset
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
_setup(ax, 'Whole (No Onset)', include_inf=False)

lo, mid, hi = band(human_models, SNR_ORDER[:-1])
ax.fill_between(x_5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x_5, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_Ducky_ITDOnly_NoOnset', C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_NoOnset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ItdIld_Whole',    C_BOTH, 'ITD + ILD'),
]:
    acc = model_acc(name, SNR_ORDER[:-1])
    ax.plot(x_5, acc, color=color, lw=LW, marker='o', ms=5, label=label)

ax.legend(fontsize=8, loc='upper left', framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 3 – Onset
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
_setup(ax, 'Onset', include_inf=False)

lo, mid, hi = band(human_models, SNR_ORDER[:-1])
ax.fill_between(x_5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x_5, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_DuckyITDOnly_Onset',  C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_Onset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_Onset',  C_BOTH, 'ITD + ILD'),
]:
    acc = model_acc(name, SNR_ORDER[:-1])
    ax.plot(x_5, acc, color=color, lw=LW, marker='o', ms=5, label=label)

ax.legend(fontsize=8, loc='upper left', framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Subplot 4 – MultiScale
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[3]
_setup(ax, 'MultiScale', include_inf=False)

lo, mid, hi = band(human_models, SNR_ORDER[:-1])
ax.fill_between(x_5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
ax.plot(x_5, mid, color=C_HUMAN, lw=LW, label='Human')

acc = model_acc('exp0_Ducky_ITDILD_MultiScale', SNR_ORDER[:-1])
ax.plot(x_5, acc, color=C_BOTH, lw=LW, marker='o', ms=5, label='ITD + ILD')

ax.legend(fontsize=8, loc='upper left', framealpha=0.8)

# ── save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'result_plot.pdf')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()