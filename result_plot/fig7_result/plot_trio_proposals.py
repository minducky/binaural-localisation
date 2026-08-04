import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf as pdf_backend
import os

CSV_SPHERICAL = os.path.join(os.path.dirname(__file__), "wandb_export_2026-05-20T14_42_59.600+01_00.csv")

SNR_ORDER = [-13.6, -6.8, 0.0, 6.8, 13.6]
SNR_LABELS = ['-13.6', '-6.8', '0', '6.8', '13.6']

def normalize_snr(s):
    v = float(s)
    return float('inf') if np.isinf(v) else round(v, 1)

df = pd.read_csv(CSV_SPHERICAL)
df['snr'] = df['snr'].apply(normalize_snr)
df = df[df['snr'] != float('inf')]
df = df.groupby(['name', 'snr'])['spherical_error'].mean().reset_index()

human_models = sorted(n for n in df['name'].unique() if n.startswith('human_'))

def model_vals(name, snrs):
    sub = df[df['name'] == name].set_index('snr')['spherical_error']
    return np.array([sub.get(s, np.nan) for s in snrs])

def band(names, snrs):
    mat = np.array([model_vals(n, snrs) for n in names])
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    return mean - std, mean, mean + std

# Fixed Okabe-Ito base colors
C_HUMAN = '#CC79A7'
ALPHA_BAND = 0.25
LW = 2.0
x5 = np.arange(len(SNR_ORDER))

# ITD / ILD / Both trio options  (all colorblind-safe, all from Okabe-Ito family)
TRIOS = {
    'A  Sky blue · Vermillion · Orange\n(lightest blue, warm contrast pair)': {
        'itd':  ('#56B4E9', 'ITD only'),    # sky blue
        'ild':  ('#D55E00', 'ILD only'),    # vermillion
        'both': ('#E69F00', 'ITD + ILD'),   # amber
    },
    'B  Blue · Orange · Vermillion\n(dark blue, medium amber, strong red)': {
        'itd':  ('#0072B2', 'ITD only'),    # deep blue
        'ild':  ('#E69F00', 'ILD only'),    # amber/orange
        'both': ('#D55E00', 'ITD + ILD'),   # vermillion
    },
    'C  Blue · Vermillion · Reddish-purple\n(cool-warm-pink spread)': {
        'itd':  ('#0072B2', 'ITD only'),    # deep blue
        'ild':  ('#D55E00', 'ILD only'),    # vermillion
        'both': ('#E69F00', 'ITD + ILD'),   # amber  (swap: both=amber)
    },
    'D  Sky blue · Orange · Black\n(maximum luminance contrast)': {
        'itd':  ('#56B4E9', 'ITD only'),    # sky blue
        'ild':  ('#E69F00', 'ILD only'),    # amber
        'both': ('#222222', 'ITD + ILD'),   # near-black
    },
}

SUBPLOTS = [
    ('Whole (No Onset)', [
        ('exp0_Ducky_ITDOnly_NoOnset', 'itd'),
        ('exp0_Ducky_IldOnly_NoOnset', 'ild'),
        ('exp0_Ducky_ItdIld_Whole',    'both'),
    ]),
    ('Onset', [
        ('exp0_DuckyITDOnly_Onset',  'itd'),
        ('exp0_Ducky_IldOnly_Onset', 'ild'),
        ('exp0_Ducky_ITDILD_Onset',  'both'),
    ]),
    ('MultiScale', [
        ('exp0_Ducky_ITDILD_MultiScale', 'both'),
    ]),
]

out_path = os.path.join(os.path.dirname(__file__), 'trio_proposals.pdf')
with pdf_backend.PdfPages(out_path) as pdf:
    for tname, TRIO in TRIOS.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle(f'Trio option {tname}', fontsize=12, fontweight='bold', y=1.03)
        fig.subplots_adjust(wspace=0.08)

        ymin = max(0, df['spherical_error'].min() - 3)
        ymax = df['spherical_error'].max() + 3

        for ax, (title, lines) in zip(axes, SUBPLOTS):
            ax.set_xticks(x5)
            ax.set_xticklabels(SNR_LABELS, fontsize=8)
            ax.set_xlim(-0.5, x5[-1] + 0.5)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel('SNR (dB)', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # human band
            lo, mid, hi = band(human_models, SNR_ORDER)
            ax.fill_between(x5, lo, hi, color=C_HUMAN, alpha=ALPHA_BAND)
            ax.plot(x5, mid, color=C_HUMAN, lw=LW, label='Human')

            for name, key in lines:
                color, label = TRIO[key]
                v = model_vals(name, SNR_ORDER)
                ax.plot(x5, v, color=color, lw=LW, marker='o', ms=5, label=label)

            ax.legend(fontsize=8, framealpha=0.8)

        axes[0].set_ylabel('Spherical Error (°)', fontsize=9)

        # swatch row
        swatch = [('Human', C_HUMAN)] + [(lbl, col) for col, lbl in TRIO.values()]
        handles = [plt.Line2D([0], [0], color=c, lw=3, label=l) for l, c in swatch]
        fig.legend(handles=handles, loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Rendered {tname.splitlines()[0]}")

print(f"Saved → {out_path}")