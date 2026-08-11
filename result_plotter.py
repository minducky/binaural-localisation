"""Evaluation result figures (confusion matrices) as PDF."""

import os

from plotter import plot_confusion_matrix


class ResultPlotter:
    """Saves evaluation figures to an experiment's output directories."""

    def __init__(self, exp_dirs: dict):
        """Stores the experiment output directories to save figures under.

        Args:
            exp_dirs: Dict of experiment output directories (from
                ``ExperimentManager.create_experiment_dir``).
        """
        self.exp_dirs = exp_dirs

    def plot_confusion_matrices(self, conf_matrices: dict, save_dir: str) -> None:
        """Plot and save azimuth, elevation, and class confusion matrices."""
        pdf_dir = os.path.join(save_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        specs = [
            ("cm_azim", "azim_labels", "Azimuth Confusion Matrix", "confusion_azim"),
            ("cm_elev", "elev_labels", "Elevation Confusion Matrix", "confusion_elev"),
            (
                "cm_class",
                "class_labels",
                "Class Confusion Matrix (sorted by elev, azim)",
                "confusion_class",
            ),
        ]

        for cm_key, label_key, title, fname in specs:
            cm = conf_matrices[cm_key]
            labels = conf_matrices[label_key]

            n = len(labels)
            # Scale figure size (inches) with matrix dimension
            size_in = max(4.0, min(12.0, n * 0.18))

            # Confusion matrices can have hundreds of classes (up to ~504),
            # at which point the figure is already at its 12in size cap and
            # plotter's default 1200 DPI would rasterize a many-thousand-
            # pixel-per-side image -- huge file, slow to render. A
            # per-cell-content heatmap doesn't need that density regardless
            # of class count, so cap the longer image dimension at
            # MAX_RASTER_PX instead of using a flat DPI: small matrices
            # (few classes, small size_in) still get the full 1200 DPI
            # default (unchanged from before -- a small matrix's file size
            # was already fine), while large ones scale down automatically.
            MAX_RASTER_PX = 6000
            longer_side_in = size_in + 1  # width is size_in+1, height is size_in
            dpi = min(1200, int(MAX_RASTER_PX / longer_side_in))

            plot_confusion_matrix(
                cm,
                labels,
                title=title,
                width=size_in + 1,
                height=size_in,
                tick_fontsize=max(6, 10 - n // 20),
                dpi=dpi,
                download=True,
                download_fpath=os.path.join(pdf_dir, f"{fname}.pdf"),
            )

        print(f"Confusion matrix plots saved to {pdf_dir}/")
