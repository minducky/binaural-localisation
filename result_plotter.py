"""Evaluation result figures (confusion matrices) as PDF+HTML."""

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
        html_dir = os.path.join(save_dir, "html")
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(html_dir, exist_ok=True)

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
            # Scale figure size with matrix dimension
            size = max(400, min(1200, n * 18))

            plot_confusion_matrix(
                cm,
                labels,
                title=title,
                width=size + 100,
                height=size,
                tick_fontsize=max(6, 10 - n // 20),
                show=False,
                download=True,
                download_fpath=[
                    os.path.join(pdf_dir, f"{fname}.pdf"),
                    os.path.join(html_dir, f"{fname}.html"),
                ],
            )

        print(f"Confusion matrix plots saved to {save_dir}/")
