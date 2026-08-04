def _plot_azimuth_accuracy(self, azim_accuracy, save_dir, title):
    """Plot accuracy as polar chart"""
    azims = np.array(list(azim_accuracy.keys()))
    accs = np.array(list(azim_accuracy.values()))

    # Convert azimuth to 0-360 range for plotting
    theta_plot = np.where(azims < 0, azims + 360, azims)
    width = 360 / len(azims) if len(azims) > 0 else 10

    fig = go.Figure()

    fig.add_trace(
        go.Barpolar(
            r=accs,
            theta=theta_plot,
            width=[width] * len(azims),
            marker=dict(
                color=accs,
                colorscale="Cividis",
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(
                        text="Accuracy",
                        font=dict(size=15, color="black"),
                        side="top",
                    ),
                    len=0.7,
                    lenmode="fraction",
                    thickness=30,
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2,
                    tickformat=".2f",
                    tickfont=dict(size=12, color="black"),
                    outlinecolor="black",
                    outlinewidth=0.2,
                    bgcolor="white",
                ),
                line=dict(color="black", width=0.5),
            ),
            opacity=0.8,
            customdata=azims,
            hovertemplate="<b>Azimuth</b>: %{customdata}°<br><b>Accuracy</b>: %{r:.3f}<br><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="black")),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=[0, 1],
                showticklabels=False,
                showline=True,
                tickmode="array",
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                gridcolor="lightgray",
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=[
                    "0°",
                    "45°",
                    "90°",
                    "135°",
                    "±180°",
                    "-135°",
                    "-90°",
                    "-45°",
                ],
                linecolor="black",
                gridcolor="black",
                tickfont=dict(size=12, color="black"),
            ),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=800,
        height=700,
        showlegend=False,
    )

    fig.write_image(os.path.join(save_dir, "azimuth_accuracy.pdf"))
    fig.write_html(os.path.join(save_dir, "azimuth_accuracy.html"))


def _plot_confusion_matrix(self, cm, save_dir, title, std_cm=None, azim_labels=None):
    """Plot confusion matrix (with optional std annotation)"""

    # Text annotations
    if std_cm is not None:
        text_annotations = [
            [f"{cm[i, j]:.1f}±{std_cm[i, j]:.1f}" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]
        colorbar_title = "Mean Count"
    else:
        text_annotations = cm.astype(int).astype(str).tolist()
        colorbar_title = "Count"

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            colorscale="Cividis",
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(size=13, color="black"),
                    side="top",
                ),
                thickness=20,
                len=0.6,
                tickfont=dict(size=11, color="black"),
                outlinecolor="black",
                outlinewidth=1,
            ),
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=8 if std_cm is not None else 10),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z:.1f}<extra></extra>",
        )
    )
    xaxis_config = dict(title="Predicted", side="bottom")
    yaxis_config = dict(title="True", autorange="reversed")

    if azim_labels is not None:
        tick_labels = [f"{int(a)}°" for a in azim_labels]
        xaxis_config.update(
            tickvals=list(range(len(azim_labels))), ticktext=tick_labels
        )
        yaxis_config.update(
            tickvals=list(range(len(azim_labels))), ticktext=tick_labels
        )

    fig.update_layout(
        title=title,
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        width=800 if std_cm is None else 900,
        height=800 if std_cm is None else 900,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.write_image(os.path.join(save_dir, "confusion_matrix.pdf"))
    fig.write_html(os.path.join(save_dir, "confusion_matrix.html"))


def _plot_aggregated_azimuth_accuracy(self, all_azim_accs, save_dir):
    """Plot aggregated azimuth accuracy with mean ± std"""

    # Get all unique azimuths
    all_azims = set()
    for azim_acc in all_azim_accs:
        all_azims.update(azim_acc.keys())
    azims = np.array(sorted(all_azims))

    # Compute mean and std for each azimuth
    mean_accs = []
    std_accs = []
    for azim in azims:
        accs = [aa.get(azim, np.nan) for aa in all_azim_accs]
        accs = [a for a in accs if not np.isnan(a)]
        mean_accs.append(np.mean(accs) if accs else 0)
        std_accs.append(np.std(accs) if accs else 0)

    mean_accs = np.array(mean_accs)
    std_accs = np.array(std_accs)

    # Convert azimuth to 0-360 range
    theta_plot = np.where(azims < 0, azims + 360, azims)
    width = 360 / len(azims) if len(azims) > 0 else 10

    fig = go.Figure()

    fig.add_trace(
        go.Barpolar(
            r=mean_accs,
            theta=theta_plot,
            width=[width] * len(azims),
            marker=dict(
                color=mean_accs,
                colorscale="Cividis",
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(
                        text="Mean Accuracy",
                        font=dict(size=15, color="black"),
                        side="top",
                    ),
                    len=0.7,
                    thickness=30,
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2,
                    tickformat=".2f",
                    tickfont=dict(size=12, color="black"),
                    outlinecolor="black",
                    outlinewidth=0.2,
                    bgcolor="white",
                ),
                line=dict(color="black", width=0.5),
            ),
            opacity=0.8,
            customdata=np.column_stack([azims, std_accs]),
            hovertemplate="<b>Azimuth</b>: %{customdata[0]}°<br><b>Mean Acc</b>: %{r:.3f}<br><b>Std</b>: %{customdata[1]:.3f}<br><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Aggregated Accuracy by Azimuth ({len(all_azim_accs)} folds)",
            font=dict(size=18, color="black"),
        ),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=[0, 1],
                showticklabels=False,
                showline=True,
                tickmode="array",
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                gridcolor="lightgray",
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=[
                    "0°",
                    "45°",
                    "90°",
                    "135°",
                    "±180°",
                    "-135°",
                    "-90°",
                    "-45°",
                ],
                linecolor="black",
                gridcolor="black",
                tickfont=dict(size=12, color="black"),
            ),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=800,
        height=700,
        showlegend=False,
    )

    fig.write_image(os.path.join(save_dir, "azimuth_accuracy.pdf"))
    fig.write_html(os.path.join(save_dir, "azimuth_accuracy.html"))

    # Save CSV
    azim_df = pd.DataFrame(
        {"azimuth": azims, "mean_accuracy": mean_accs, "std_accuracy": std_accs}
    )
    azim_df.to_csv(os.path.join(save_dir, "azimuth_accuracy.csv"), index=False)