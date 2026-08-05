"""Evaluation result figures (SNR breakdown, confusion matrices) as PDF+HTML."""

import os

import numpy as np
import plotly.graph_objects as go


class ResultPlotter:
    """Saves evaluation figures to an experiment's output directories."""

    def __init__(self, exp_dirs: dict):
        """Stores the experiment output directories to save figures under.

        Args:
            exp_dirs: Dict of experiment output directories (from
                ``ExperimentManager.create_experiment_dir``).
        """
        self.exp_dirs = exp_dirs

    def plot_evaluation_results(
        self,
        results_by_snr: dict,
        overall_results: dict,
        save_dir: str,
        eval_metadata: dict | None = None,
    ) -> None:
        """Plot comprehensive evaluation results with SNR breakdown"""

        # Extract data for plotting
        snr_values = []
        spherical_errors = []
        azimuth_errors = []
        elevation_errors = []
        azimuth_accs = []
        elevation_accs = []
        combined_accs = []

        for snr_key in sorted(
            results_by_snr.keys(), key=lambda k: float(k.replace("SNR_", ""))
        ):
            snr_val = float(snr_key.replace("SNR_", ""))
            metrics = results_by_snr[snr_key]

            snr_values.append(snr_val)
            spherical_errors.append(metrics["spherical_error"])
            azimuth_errors.append(metrics["azimuth_error"])
            elevation_errors.append(metrics["elevation_error"])
            azimuth_accs.append(metrics["azimuth_accuracy"])
            elevation_accs.append(metrics["elevation_accuracy"])
            combined_accs.append(metrics["combined_accuracy"])

        # Add metadata to title if available
        title_suffix = ""
        if eval_metadata:
            n_azims = eval_metadata["num_unique_azims"]
            n_elevs = eval_metadata["num_unique_elevs"]
            n_classes = eval_metadata["num_unique_classes"]
            title_suffix = (
                f"<br><sub>{n_azims} azim × {n_elevs} elev = {n_classes} classes</sub>"
            )

        # Create subplots
        fig = go.Figure()

        # Plot 1: Angular Errors by SNR
        fig.add_trace(
            go.Scatter(
                x=snr_values,
                y=spherical_errors,
                mode="lines+markers",
                name="Spherical Error",
                line=dict(color="#D62728", width=2),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=snr_values,
                y=azimuth_errors,
                mode="lines+markers",
                name="Azimuth Error",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=snr_values,
                y=elevation_errors,
                mode="lines+markers",
                name="Elevation Error",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=8),
            )
        )

        # Add overall average lines
        fig.add_trace(
            go.Scatter(
                x=[min(snr_values), max(snr_values)],
                y=[overall_results["spherical_error"].item()] * 2,
                mode="lines",
                name="Overall Spherical",
                line=dict(color="#D62728", width=1, dash="dash"),
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[min(snr_values), max(snr_values)],
                y=[overall_results["azimuth_error"].item()] * 2,
                mode="lines",
                name="Overall Azimuth",
                line=dict(color="#1f77b4", width=1, dash="dash"),
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[min(snr_values), max(snr_values)],
                y=[overall_results["elevation_error"].item()] * 2,
                mode="lines",
                name="Overall Elevation",
                line=dict(color="#2ca02c", width=1, dash="dash"),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(text=f"Angular Errors by SNR{title_suffix}", font=dict(size=20)),
            xaxis=dict(
                title="SNR (dB)",
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
            ),
            yaxis=dict(
                title="Error (degrees)",
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor="lightgray",
            ),
            width=1000,
            height=600,
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(x=0.7, y=0.95),
        )

        # Save pdf / html separately
        pdf_save_dir = os.path.join(save_dir, "pdf")
        os.makedirs(pdf_save_dir)
        html_save_dir = os.path.join(save_dir, "html")
        os.makedirs(html_save_dir)

        fig.write_image(os.path.join(pdf_save_dir, "angular_errors_by_snr.pdf"))
        fig.write_html(os.path.join(html_save_dir, "angular_errors_by_snr.html"))

        # Plot 2: Classification Accuracies by SNR
        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=snr_values,
                y=combined_accs,
                mode="lines+markers",
                name="Combined (Azim+Elev)",
                line=dict(color="#9467bd", width=2),
                marker=dict(size=8),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=snr_values,
                y=azimuth_accs,
                mode="lines+markers",
                name="Azimuth Only",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=snr_values,
                y=elevation_accs,
                mode="lines+markers",
                name="Elevation Only",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=8),
            )
        )

        # Add overall average lines
        fig2.add_trace(
            go.Scatter(
                x=[min(snr_values), max(snr_values)],
                y=[overall_results["combined_accuracy"].item()] * 2,
                mode="lines",
                name="Overall Combined",
                line=dict(color="#9467bd", width=1, dash="dash"),
                showlegend=True,
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=[min(snr_values), max(snr_values)],
                y=[overall_results["azimuth_accuracy"].item()] * 2,
                mode="lines",
                name="Overall Azimuth",
                line=dict(color="#1f77b4", width=1, dash="dash"),
                showlegend=True,
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=[min(snr_values), max(snr_values)],
                y=[overall_results["elevation_accuracy"].item()] * 2,
                mode="lines",
                name="Overall Elevation",
                line=dict(color="#1f77b4", width=1, dash="dash"),
                showlegend=True,
            )
        )

        fig2.update_layout(
            title=dict(
                text=f"Classification Accuracy by SNR{title_suffix}", font=dict(size=20)
            ),
            xaxis=dict(
                title="SNR (dB)",
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
            ),
            yaxis=dict(
                title="Accuracy",
                tickfont=dict(size=12),
                tickformat=".0%",
                showgrid=True,
                gridcolor="lightgray",
            ),
            width=1000,
            height=600,
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(x=0.7, y=0.15),
        )

        fig2.write_image(os.path.join(pdf_save_dir, "accuracy_by_snr.pdf"))
        fig2.write_html(os.path.join(html_save_dir, "accuracy_by_snr.html"))

        print(f"Evaluation plots saved to {save_dir}/")

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
            cm = conf_matrices[cm_key].astype(float)
            labels = [str(v) for v in conf_matrices[label_key]]

            # Row-normalize to show recall per true class
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

            n = len(labels)
            # Scale figure size with matrix dimension
            size = max(400, min(1200, n * 18))

            fig = go.Figure(
                go.Heatmap(
                    z=cm_norm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="Recall", tickformat=".0%"),
                    hovertemplate=(
                        "True: %{y}<br>Pred: %{x}<br>Recall: %{z:.1%}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title=dict(text=title, font=dict(size=16)),
                xaxis=dict(
                    title="Predicted",
                    tickfont=dict(size=max(6, 10 - n // 20)),
                    categoryorder="array",
                    categoryarray=labels,
                ),
                yaxis=dict(
                    title="True",
                    tickfont=dict(size=max(6, 10 - n // 20)),
                    categoryorder="array",
                    categoryarray=labels,
                    autorange="reversed",
                ),
                width=size + 100,
                height=size,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )

            fig.write_image(os.path.join(pdf_dir, f"{fname}.pdf"))
            fig.write_html(os.path.join(html_dir, f"{fname}.html"))

        print(f"Confusion matrix plots saved to {save_dir}/")
