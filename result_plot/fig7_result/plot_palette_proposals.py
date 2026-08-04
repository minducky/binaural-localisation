import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf as pdf_backend
import os

CSV_ACCURACY  = os.path.join(os.path.dirname(__file__), "wandb_export_2026-05-19T17_35_00.153+01_00.csv")
CSV_SPHERICAL = os.path.join(os.path.dirname(__file__), "wandb_export_2026-05-20T14_42_59.600+01_00.csv")

SNR_ORDER = [-13.6, -6.8, 0.0, 6.8, 13.6]
SNR_LABELS = ['-13.6', '-6.8', '0', '6.8', '13.6']

def normalize_snr(s):
    v = float(s)
    if np.isinf(v) or str(s) == 'Infinity':
        return float('inf')
    return round(v, 1)

def load_accuracy():
    df = pd.read_csv(CSV_ACCURACY)
    df['snr'] = df['snr'].apply(normalize_snr)
    return df.groupby(['name', 'snr'])['combined_accuracy'].mean().reset_index()

def load_spherical():
    df = pd.read_csv(CSV_SPHERICAL)
    df['snr'] = df['snr'].apply(normalize_snr)
    df = df[df['snr'] != float('inf')]
    return df.groupby(['name', 'snr'])['spherical_error'].mean().reset_index()

df_acc  = load_accuracy()
df_sph  = load_spherical()

arch_models  = sorted(n for n in df_acc['name'].unique() if n.startswith('eval_arch'))
yang_models  = sorted(n for n in df_acc['name'].unique() if 'Yang' in n)
human_models = sorted(n for n in df_acc['name'].unique() if n.startswith('human_'))

def model_vals(df, col, name, snrs):
    sub = df[df['name'] == name].set_index('snr')[col]
    return np.array([sub.get(s, np.nan) for s in snrs])

def band(df, col, names, snrs):
    mat = np.array([model_vals(df, col, n, snrs) for n in names])
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    return mean - std, mean, mean + std

# ── Palette definitions ────────────────────────────────────────────────────────
PALETTES = {
    'A  Okabe-Ito': {
        'human': '#CC79A7',   # pinkish purple
        'arch':  '#999999',   # gray
        'yang':  '#E69F00',   # amber
        'itd':   '#0072B2',   # deep blue
        'ild':   '#D55E00',   # vermillion
        'both':  '#009E73',   # bluish green
    },
    'B  Paul Tol Vibrant': {
        'human': '#AA3377',   # magenta
        'arch':  '#BBBBBB',   # silver
        'yang':  '#EE7733',   # orange
        'itd':   '#0077BB',   # blue
        'ild':   '#CC3311',   # red
        'both':  '#009988',   # teal
    },
    'C  Elegant Dark': {
        'human': '#7B2FBE',   # royal purple
        'arch':  '#6B6B6B',   # charcoal
        'yang':  '#C8960C',   # golden
        'itd':   '#1A5276',   # dark navy
        'ild':   '#B03A2E',   # dark red
        'both':  '#148F77',   # dark teal
    },
    'D  Fresh Modern': {
        'human': '#8E24AA',   # vivid purple
        'arch':  '#78909C',   # blue-gray
        'yang':  '#F57F17',   # deep amber
        'itd':   '#1565C0',   # indigo
        'ild':   '#BF360C',   # deep orange-red
        'both':  '#00695C',   # dark cyan
    },
}

ALPHA_BAND = 0.25
LW = 2.0
x5 = np.arange(len(SNR_ORDER))

def _setup(ax, title, ylabel=None):
    ax.set_xticks(x5)
    ax.set_xticklabels(SNR_LABELS, fontsize=8)
    ax.set_xlim(-0.5, x5[-1] + 0.5)
    ax.set_xlabel('SNR (dB)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

def draw_row(axes, C, df, col, ylabel, snrs, inv_y=False):
    """Draw all 4 subplots for one metric."""
    snr_vals = snrs

    # ── Subplot 1 – Baselines ──────────────────────────────────────────────────
    ax = axes[0]
    _setup(ax, 'Baselines', ylabel)

    lo, mid, hi = band(df, col, arch_models, snr_vals)
    ax.fill_between(x5, lo, hi, color=C['arch'], alpha=ALPHA_BAND)
    ax.plot(x5, mid, color=C['arch'], lw=LW, label='Saddler (arch)')

    lo, mid, hi = band(df, col, yang_models, snr_vals)
    ax.fill_between(x5, lo, hi, color=C['yang'], alpha=ALPHA_BAND)
    ax.plot(x5, mid, color=C['yang'], lw=LW, label='Yang 2025')

    lo, mid, hi = band(df, col, human_models, snr_vals)
    ax.fill_between(x5, lo, hi, color=C['human'], alpha=ALPHA_BAND)
    ax.plot(x5, mid, color=C['human'], lw=LW, label='Human')

    ax.legend(fontsize=7, framealpha=0.8)

    # ── Subplot 2 – Whole / No Onset ──────────────────────────────────────────
    ax = axes[1]
    _setup(ax, 'Whole (No Onset)')

    lo, mid, hi = band(df, col, human_models, snr_vals)
    ax.fill_between(x5, lo, hi, color=C['human'], alpha=ALPHA_BAND)
    ax.plot(x5, mid, color=C['human'], lw=LW, label='Human')

    for name, key, label in [
        ('exp0_Ducky_ITDOnly_NoOnset', 'itd',  'ITD only'),
        ('exp0_Ducky_IldOnly_NoOnset', 'ild',  'ILD only'),
        ('exp0_Ducky_ItdIld_Whole',    'both', 'ITD + ILD'),
    ]:
        v = model_vals(df, col, name, snr_vals)
        ax.plot(x5, v, color=C[key], lw=LW, marker='o', ms=4, label=label)

    ax.legend(fontsize=7, framealpha=0.8)

    # ── Subplot 3 – Onset ─────────────────────────────────────────────────────
    ax = axes[2]
    _setup(ax, 'Onset')

    lo, mid, hi = band(df, col, human_models, snr_vals)
    ax.fill_between(x5, lo, hi, color=C['human'], alpha=ALPHA_BAND)
    ax.plot(x5, mid, color=C['human'], lw=LW, label='Human')

    for name, key, label in [
        ('exp0_DuckyITDOnly_Onset',  'itd',  'ITD only'),
        ('exp0_Ducky_IldOnly_Onset', 'ild',  'ILD only'),
        ('exp0_Ducky_ITDILD_Onset',  'both', 'ITD + ILD'),
    ]:
        v = model_vals(df, col, name, snr_vals)
        ax.plot(x5, v, color=C[key], lw=LW, marker='o', ms=4, label=label)

    ax.legend(fontsize=7, framealpha=0.8)

    # ── Subplot 4 – MultiScale ────────────────────────────────────────────────
    ax = axes[3]
    _setup(ax, 'MultiScale')

    lo, mid, hi = band(df, col, human_models, snr_vals)
    ax.fill_between(x5, lo, hi, color=C['human'], alpha=ALPHA_BAND)
    ax.plot(x5, mid, color=C['human'], lw=LW, label='Human')

    v = model_vals(df, col, 'exp0_Ducky_ITDILD_MultiScale', snr_vals)
    ax.plot(x5, v, color=C['both'], lw=LW, marker='o', ms=4, label='ITD + ILD')

    ax.legend(fontsize=7, framealpha=0.8)

    for ax in axes:
        if inv_y:
            pass  # spherical error: higher = worse, keep default (low=good shown low)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f' if not inv_y else '%.1f'))

# ── Build PDF (one page per palette, each page = 2 rows: accuracy + spherical) ─
out_path = os.path.join(os.path.dirname(__file__), 'palette_proposals.pdf')
with pdf_backend.PdfPages(out_path) as pdf:
    for pname, C in PALETTES.items():
        fig, axes_grid = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle(f'Palette {pname}', fontsize=14, fontweight='bold', y=1.01)
        fig.subplots_adjust(wspace=0.08, hspace=0.35)

        # row 0 – accuracy
        axes_acc = axes_grid[0]
        axes_acc[0].set_title('')
        for ax in axes_acc[1:]:
            ax.sharey(axes_acc[0])
        draw_row(axes_acc, C, df_acc, 'combined_accuracy', 'Accuracy', SNR_ORDER)

        # row 1 – spherical error
        axes_sph = axes_grid[1]
        for ax in axes_sph[1:]:
            ax.sharey(axes_sph[0])
        draw_row(axes_sph, C, df_sph, 'spherical_error', 'Spherical Error (°)', SNR_ORDER, inv_y=True)

        # row labels
        axes_acc[0].annotate('Accuracy', xy=(0, 0.5), xytext=(-0.18, 0.5),
                              xycoords='axes fraction', textcoords='axes fraction',
                              fontsize=11, ha='center', va='center', rotation=90,
                              fontweight='bold')
        axes_sph[0].annotate('Spherical Error', xy=(0, 0.5), xytext=(-0.18, 0.5),
                              xycoords='axes fraction', textcoords='axes fraction',
                              fontsize=11, ha='center', va='center', rotation=90,
                              fontweight='bold')

        # colour swatch legend at bottom
        swatch_labels = [
            ('Human',        C['human']),
            ('Saddler arch', C['arch']),
            ('Yang 2025',    C['yang']),
            ('ITD only',     C['itd']),
            ('ILD only',     C['ild']),
            ('ITD + ILD',    C['both']),
        ]
        handles = [plt.Line2D([0], [0], color=col, lw=3, label=lbl)
                   for lbl, col in swatch_labels]
        fig.legend(handles=handles, loc='lower center', ncol=6,
                   fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.03))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Rendered palette {pname}")

print(f"Saved → {out_path}")