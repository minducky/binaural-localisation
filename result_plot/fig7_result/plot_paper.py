import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))

SNR_ORDER  = [-13.6, -6.8, 0.0, 6.8, 13.6]
SNR_LABELS = ['-13.6', '-6.8', '0', '6.8', '13.6']

def normalize_snr(s):
    if str(s) == 'Infinity':
        return float('inf')
    return round(float(s), 1)

acc_df = pd.read_csv(os.path.join(BASE, 'eval_result/Combined_accuracy.csv'))
err_df = pd.read_csv(os.path.join(BASE, 'eval_result/Spherical_error.csv'))

for df in [acc_df, err_df]:
    df['snr'] = df['snr'].apply(normalize_snr)

acc_df = acc_df[acc_df['snr'] != float('inf')]
err_df = err_df[err_df['snr'] != float('inf')]

acc_df = acc_df.groupby(['name', 'snr'])['combined_accuracy'].mean().reset_index()
err_df = err_df.groupby(['name', 'snr'])['spherical_error'].mean().reset_index()

def get_groups(names):
    arch  = sorted(n for n in names if n.startswith('eval_arch'))
    yang  = sorted(n for n in names if 'Yang' in n)
    human = sorted(n for n in names if n.startswith('human_'))
    return arch, yang, human

acc_arch, acc_yang, acc_human = get_groups(acc_df['name'].unique())
err_arch, err_yang, err_human = get_groups(err_df['name'].unique())

def get_vals(df, col, name):
    sub = df[df['name'] == name].set_index('snr')[col]
    return np.array([sub.get(s, np.nan) for s in SNR_ORDER])

def get_band(df, col, names):
    mat  = np.array([get_vals(df, col, n) for n in names])
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    return mean - std, mean, mean + std

C_ARCH  = '#999999'
C_YANG  = '#D55E00'
C_HUMAN = '#CC79A7'
C_ITD   = '#56B4E9'
C_ILD   = '#E69F00'
C_BOTH  = '#222222'

ALPHA = 0.25
LW    = 1.2
MS    = 3

FS_LABEL = 8
FS_TICK  = 7
FS_LEG   = 7
FS_TITLE = 9

x = np.arange(len(SNR_ORDER))

fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0),
                          sharey='row', sharex=True)

# ── titles (top row only) ─────────────────────────────────────────────────────
for col, title in enumerate(['Baselines', 'Whole (No Onset)', 'Onset', 'MultiScale']):
    axes[0, col].set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=3)

# ── y-labels (leftmost column only per row) ───────────────────────────────────
axes[0, 0].set_ylabel('Combined Accuracy', fontsize=FS_LABEL)
axes[1, 0].set_ylabel('Spherical Error (°)', fontsize=FS_LABEL)

# ── shared x-label ────────────────────────────────────────────────────────────
fig.text(0.5, 0.01, 'SNR (dB)', ha='center', va='bottom', fontsize=FS_LABEL,
         family='Times New Roman')

# ── common axis setup ─────────────────────────────────────────────────────────
ymin_acc = 0.0
ymax_acc = min(1.0, np.ceil(acc_df['combined_accuracy'].max() / 0.05) * 0.05 + 0.04)
ymin_err = max(0.0, np.floor(err_df['spherical_error'].min() / 5) * 5 - 2)
ymax_err = np.ceil(err_df['spherical_error'].max() / 5) * 5 + 2

for col in range(4):
    for row, (ymin, ymax) in enumerate([(ymin_acc, ymax_acc), (ymin_err, ymax_err)]):
        ax = axes[row, col]
        ax.set_xlim(-0.5, x[-1] + 0.5)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25, linestyle='--', lw=0.5)
        ax.tick_params(axis='both', labelsize=FS_TICK)

# x-tick labels on bottom row only (sharex handles hiding top row automatically)
for col in range(4):
    axes[1, col].set_xticks(x)
    axes[1, col].set_xticklabels(SNR_LABELS, fontsize=FS_TICK)

# y-tick formatters
for col in range(4):
    axes[0, col].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axes[1, col].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# hide y-tick labels for non-leftmost columns (sharey='row' handles limits,
# but tick label visibility must be set explicitly)
for row in range(2):
    for col in range(1, 4):
        axes[row, col].tick_params(labelleft=False)

# ══════════════════════════════════════════════════════════════════════════════
# Row 0 — Combined Accuracy
# ══════════════════════════════════════════════════════════════════════════════

# [0,0] Baselines
ax = axes[0, 0]
lo, mid, hi = get_band(acc_df, 'combined_accuracy', acc_arch)
ax.fill_between(x, lo, hi, color=C_ARCH, alpha=ALPHA)
ax.plot(x, mid, color=C_ARCH, lw=LW, label='Saddler (arch)')

lo, mid, hi = get_band(acc_df, 'combined_accuracy', acc_yang)
ax.fill_between(x, lo, hi, color=C_YANG, alpha=ALPHA)
ax.plot(x, mid, color=C_YANG, lw=LW, label='Yang 2025')

lo, mid, hi = get_band(acc_df, 'combined_accuracy', acc_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

ax.legend(fontsize=FS_LEG, loc='upper left', framealpha=0.8, handlelength=1.5)

# [0,1] Whole / No Onset
ax = axes[0, 1]
lo, mid, hi = get_band(acc_df, 'combined_accuracy', acc_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_Ducky_ITDOnly_NoOnset', C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_NoOnset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ItdIld_Whole',    C_BOTH, 'ITD+ILD'),
]:
    ax.plot(x, get_vals(acc_df, 'combined_accuracy', name),
            color=color, lw=LW, marker='o', ms=MS, label=label)

ax.legend(fontsize=FS_LEG, loc='upper left', framealpha=0.8, handlelength=1.5)

# [0,2] Onset
ax = axes[0, 2]
lo, mid, hi = get_band(acc_df, 'combined_accuracy', acc_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_DuckyITDOnly_Onset',   C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_Onset',  C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_Onset',   C_BOTH, 'ITD+ILD'),
]:
    ax.plot(x, get_vals(acc_df, 'combined_accuracy', name),
            color=color, lw=LW, marker='o', ms=MS, label=label)

ax.legend(fontsize=FS_LEG, loc='upper left', framealpha=0.8, handlelength=1.5)

# [0,3] MultiScale
ax = axes[0, 3]
lo, mid, hi = get_band(acc_df, 'combined_accuracy', acc_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_Ducky_ItdMultiscale',       C_ITD,  'ITD only'),
    ('exp0_Ducky_IldMultiscale',       C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_MultiScale',   C_BOTH, 'ITD+ILD'),
]:
    ax.plot(x, get_vals(acc_df, 'combined_accuracy', name),
            color=color, lw=LW, marker='o', ms=MS, label=label)

ax.legend(fontsize=FS_LEG, loc='upper left', framealpha=0.8, handlelength=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — Spherical Error
# ══════════════════════════════════════════════════════════════════════════════

# [1,0] Baselines
ax = axes[1, 0]
lo, mid, hi = get_band(err_df, 'spherical_error', err_arch)
ax.fill_between(x, lo, hi, color=C_ARCH, alpha=ALPHA)
ax.plot(x, mid, color=C_ARCH, lw=LW, label='Saddler (arch)')

lo, mid, hi = get_band(err_df, 'spherical_error', err_yang)
ax.fill_between(x, lo, hi, color=C_YANG, alpha=ALPHA)
ax.plot(x, mid, color=C_YANG, lw=LW, label='Yang 2025')

lo, mid, hi = get_band(err_df, 'spherical_error', err_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

ax.legend(fontsize=FS_LEG, loc='upper right', framealpha=0.8, handlelength=1.5)

# [1,1] Whole / No Onset
ax = axes[1, 1]
lo, mid, hi = get_band(err_df, 'spherical_error', err_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_Ducky_ITDOnly_NoOnset', C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_NoOnset', C_ILD,  'ILD only'),
    ('exp0_Ducky_ItdIld_Whole',    C_BOTH, 'ITD+ILD'),
]:
    ax.plot(x, get_vals(err_df, 'spherical_error', name),
            color=color, lw=LW, marker='o', ms=MS, label=label)

ax.legend(fontsize=FS_LEG, loc='upper right', framealpha=0.8, handlelength=1.5)

# [1,2] Onset
ax = axes[1, 2]
lo, mid, hi = get_band(err_df, 'spherical_error', err_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_DuckyITDOnly_Onset',   C_ITD,  'ITD only'),
    ('exp0_Ducky_IldOnly_Onset',  C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_Onset',   C_BOTH, 'ITD+ILD'),
]:
    ax.plot(x, get_vals(err_df, 'spherical_error', name),
            color=color, lw=LW, marker='o', ms=MS, label=label)

ax.legend(fontsize=FS_LEG, loc='upper right', framealpha=0.8, handlelength=1.5)

# [1,3] MultiScale
ax = axes[1, 3]
lo, mid, hi = get_band(err_df, 'spherical_error', err_human)
ax.fill_between(x, lo, hi, color=C_HUMAN, alpha=ALPHA)
ax.plot(x, mid, color=C_HUMAN, lw=LW, label='Human')

for name, color, label in [
    ('exp0_Ducky_ItdMultiscale',       C_ITD,  'ITD only'),
    ('exp0_Ducky_IldMultiscale',       C_ILD,  'ILD only'),
    ('exp0_Ducky_ITDILD_MultiScale',   C_BOTH, 'ITD+ILD'),
]:
    ax.plot(x, get_vals(err_df, 'spherical_error', name),
            color=color, lw=LW, marker='o', ms=MS, label=label)

ax.legend(fontsize=FS_LEG, loc='upper right', framealpha=0.8, handlelength=1.5)

# ── save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.8, w_pad=0.4)

out_path = os.path.join(BASE, 'paper_plot.pdf')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()