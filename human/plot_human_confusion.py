import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), 'raw_data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'human_confusion')

ELEV_MAP = {'A': 40, 'B': 30, 'C': 20, 'D': 10, 'E': 0}


def decode_label(series):
    elev = series.str[0].map(ELEV_MAP)
    azim = (series.str[1:].astype(int) - 10) * 10   # -90 … 90
    return elev, azim


def load_all():
    dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} responses from {df['subject_id'].nunique()} subjects")
    return df


def _global_labels(df):
    """Compute fixed label sets from the full dataset for consistent matrix shapes."""
    t_elev, t_azim = decode_label(df['response_true'])
    p_elev, p_azim = decode_label(df['response_pred'])

    azim_labels = sorted(t_azim.unique().tolist())   # -90 … 90
    elev_labels = sorted(t_elev.unique().tolist())

    class_to_angle = dict(zip(
        pd.concat([df['response_true'], df['response_pred']]),
        zip(pd.concat([t_azim, p_azim]), pd.concat([t_elev, p_elev])),
    ))
    all_classes = sorted(
        df['response_true'].unique().tolist(),
        key=lambda c: (class_to_angle[c][1], class_to_angle[c][0]),
    )
    class_tick_labels = [
        f"e{class_to_angle[c][1]:.0f}/a{class_to_angle[c][0]:.0f}"
        for c in all_classes
    ]
    return {
        'azim_labels': azim_labels,
        'elev_labels': elev_labels,
        'all_classes': all_classes,
        'class_tick_labels': class_tick_labels,
    }


def build_confusion_matrices(df, labels):
    t_elev, t_azim = decode_label(df['response_true'])
    p_elev, p_azim = decode_label(df['response_pred'])

    cm_azim = confusion_matrix(t_azim, p_azim, labels=labels['azim_labels'])
    cm_elev = confusion_matrix(t_elev, p_elev, labels=labels['elev_labels'])
    cm_class = confusion_matrix(
        df['response_true'], df['response_pred'], labels=labels['all_classes']
    )

    return {
        'cm_azim':  cm_azim,  'azim_labels':  [str(a) for a in labels['azim_labels']],
        'cm_elev':  cm_elev,  'elev_labels':  [str(e) for e in labels['elev_labels']],
        'cm_class': cm_class, 'class_labels': labels['class_tick_labels'],
    }


def plot_confusion_matrices(conf_matrices, save_dir, subject=None):
    pdf_dir = os.path.join(save_dir, 'pdf')
    html_dir = os.path.join(save_dir, 'html')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)

    tag = f" — {subject}" if subject else ""
    specs = [
        ('cm_azim',  'azim_labels',  f'Human Azimuth Confusion Matrix{tag}',                     'human_confusion_azim'),
        ('cm_elev',  'elev_labels',  f'Human Elevation Confusion Matrix{tag}',                    'human_confusion_elev'),
        ('cm_class', 'class_labels', f'Human Class Confusion Matrix (sorted by elev, azim){tag}', 'human_confusion_class'),
    ]

    for cm_key, label_key, title, fname in specs:
        cm = conf_matrices[cm_key].astype(float)
        labels = conf_matrices[label_key]

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        n = len(labels)
        size = max(400, min(1200, n * 18))

        fig = go.Figure(go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale='Blues',
            zmin=0, zmax=1,
            colorbar=dict(title='Recall', tickformat='.0%'),
            hovertemplate='True: %{y}<br>Pred: %{x}<br>Recall: %{z:.1%}<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(title='Predicted', tickfont=dict(size=max(6, 10 - n // 20)),
                       categoryorder='array', categoryarray=labels),
            yaxis=dict(title='True',      tickfont=dict(size=max(6, 10 - n // 20)),
                       categoryorder='array', categoryarray=labels,
                       autorange='reversed'),
            width=size + 100,
            height=size,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig.write_image(os.path.join(pdf_dir,  f"{fname}.pdf"))
        fig.write_html(os.path.join(html_dir,  f"{fname}.html"))

    print(f"  Saved → {save_dir}")


def main():
    df = load_all()
    labels = _global_labels(df)

    # All subjects combined
    conf_matrices = build_confusion_matrices(df, labels)
    plot_confusion_matrices(conf_matrices, os.path.join(OUT_DIR, 'all'))

    # Per subject
    for subject_id, group in df.groupby('subject_id'):
        conf_matrices = build_confusion_matrices(group.reset_index(drop=True), labels)
        plot_confusion_matrices(
            conf_matrices,
            os.path.join(OUT_DIR, subject_id),
            subject=subject_id,
        )

    print(f"\nDone. Output → {OUT_DIR}/")


if __name__ == '__main__':
    main()