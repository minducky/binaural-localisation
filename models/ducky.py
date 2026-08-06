"""Ducky model family: ITD/ILD binaural cues, onset windowing, CNN encoders."""

import torch
import torch.nn as nn
from auditory_layers.cochlear import BM, IHC
from auditory_layers.midbrain import (
    ILD,
    AvgILD,
    Correlagram,
    ILDNormaliser,
    ITDNormaliser,
    OnsetSlicer,
)
from torch.profiler import record_function


class PeripheralFrontend(nn.Module):
    """Shared BM + IHC pipeline. Processes left and right channels together."""

    def __init__(self, config: dict):
        """Builds the BM/IHC stages selected by ``config['DUCKY']['LAYERS']``.

        Args:
            config: Experiment config dict with a ``DUCKY`` block containing
                ``BM``/``IHC`` sub-configs, ``SR``, and ``LAYERS`` (subset of
                ``{'BM', 'IHC'}``), plus a top-level ``FILTER_COEFF_DIR``.
        """
        super().__init__()
        bm_config = config["DUCKY"]["BM"]
        ihc_config = config["DUCKY"]["IHC"]
        sr = config["DUCKY"]["SR"]
        filter_coeff_dir = config["FILTER_COEFF_DIR"]
        layers = config["DUCKY"]["LAYERS"]

        self.module_list = nn.ModuleList()
        self.module_names = []

        if "BM" in layers:
            self.module_list.append(
                BM(
                    fmin=bm_config["fmin"],
                    fmax=bm_config["fmax"],
                    bw=bm_config["bw"],
                    sr=sr,
                    order=bm_config["order"],
                    filter_coeff_dir=filter_coeff_dir,
                    learnable_coefficients=bm_config["learnable_coefficients"],
                    fc_min_bound=bm_config["fc_min_bound"],
                    fc_max_bound=bm_config["fc_max_bound"],
                    ourbeta_min=bm_config["ourbeta_min"],
                    ourbeta_max=bm_config["ourbeta_max"],
                )
            )
            self.module_names.append("BM")
        if "IHC" in layers:
            self.module_list.append(
                IHC(
                    fcut=ihc_config["fcut"],
                    sr=sr,
                    order=ihc_config["order"],
                    filter_coeff_dir=filter_coeff_dir,
                )
            )
            self.module_names.append("IHC")

    def forward(
        self, x_l: torch.Tensor, x_r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the configured BM/IHC stages on both channels.

        Args:
            x_l: Left channel audio of shape (B, 1, T).
            x_r: Right channel audio of shape (B, 1, T).

        Returns:
            A tuple ``(x_l, x_r)`` of peripheral responses, each of shape
            (B, F, T).
        """
        for name, layer in zip(self.module_names, self.module_list, strict=True):
            with record_function(name):
                x_l = layer(x_l)
                x_r = layer(x_r)
        return x_l, x_r


def _compute_itd_map(
    corr_module: Correlagram,
    itd_normaliser: ITDNormaliser,
    x_l: torch.Tensor,
    x_r: torch.Tensor,
) -> torch.Tensor:
    """Computes a normalized ITD map ready for a 2D conv encoder.

    Args:
        corr_module: Cross-correlation module.
        itd_normaliser: Normaliser applied to ``corr_module``'s output.
        x_l: Left peripheral response of shape (B, F, T).
        x_r: Right peripheral response of shape (B, F, T).

    Returns:
        Normalized ITD map of shape (B, 1, F, ITD_bins).
    """
    corr = corr_module(x_l, x_r)
    return itd_normaliser(corr, x_l, x_r).unsqueeze(1)


def _compute_ild_map(
    ild_module: ILD | AvgILD,
    ild_normaliser: ILDNormaliser,
    x_l: torch.Tensor,
    x_r: torch.Tensor,
) -> torch.Tensor:
    """Computes a normalized ILD map ready for a 2D conv encoder.

    Args:
        ild_module: ``ILD`` (for already avg-pooled inputs) or ``AvgILD``
            (which avg-pools internally) instance.
        ild_normaliser: Normaliser applied to ``ild_module``'s output.
        x_l: Left peripheral response of shape (B, F, T).
        x_r: Right peripheral response of shape (B, F, T).

    Returns:
        Normalized ILD map of shape (B, 1, F, T') where T' depends on
        ``ild_module``.
    """
    ild = ild_module(x_l, x_r)
    return ild_normaliser(ild).unsqueeze(1)


# %% Encoder building blocks


def _make_itd_encoder() -> nn.Sequential:
    """Conv encoder for ITD map (B, 1, F, ITD_bins) → (B, 256)."""
    return nn.Sequential(
        # (B, 1, 71, 133)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 32, 35, 66)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 64, 17, 33)
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 128, 8, 16)
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 256, 4, 8)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 256, 2, 4)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        # (B, 256, 1, 1)
        nn.Flatten(),
        # (B, 256)
    )


def _make_ild_onset_encoder() -> nn.Sequential:
    """Conv encoder for onset ILD map (B, 1, F, t_window//stride) → (B, 256)."""
    return nn.Sequential(
        # (B, 1, 71, 10)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d((2, 1), (2, 1)),
        # (B, 32, 35, 10)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d((2, 1), (2, 1)),
        # (B, 64, 17, 10)
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AvgPool2d((2, 1), (2, 1)),
        # (B, 128, 8, 10)
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 256, 4, 5)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 256, 2, 2)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        # (B, 256, 1, 1)
        nn.Flatten(),
        # (B, 256)
    )


def _make_ild_whole_encoder() -> nn.Sequential:
    """Conv encoder for whole-signal ILD map (B, 1, F, T//stride) → (B, 256).
    AdaptiveAvgPool2d handles variable time length.
    """
    return nn.Sequential(
        # (B, 1, F, T//stride)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d((2, 2), (2, 2)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # (B, 256)
    )


# %% Models


class DuckyItdOnsetModel(nn.Module):
    """Experiment 2: ITD-only model using the onset window.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.onset_slicer = OnsetSlicer(sr=sr, avg_window=0.005, slice_window=0.050)
        self.CORR = Correlagram(
            sr_res=sr, phy_itd_range=config["DUCKY"]["CORR"]["phy_ITD_range"]
        )
        self.itd_normaliser = ITDNormaliser()

        self.ITD_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)  # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)  # (B, F, T)
        x_l, x_r, _, _ = self.onset_slicer(x_l, x_r)

        with record_function("CORR"):
            ITD = _compute_itd_map(self.CORR, self.itd_normaliser, x_l, x_r)

        with record_function("Classifier"):
            return self.Classifier(self.ITD_Encoder(ITD))


class DuckyItdWholeModel(nn.Module):
    """Experiment 1: ITD-only model using the whole signal.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(
            sr_res=sr, phy_itd_range=config["DUCKY"]["CORR"]["phy_ITD_range"]
        )
        self.itd_normaliser = ITDNormaliser()

        self.ITD_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)  # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)  # (B, F, T)

        with record_function("CORR"):
            ITD = _compute_itd_map(self.CORR, self.itd_normaliser, x_l, x_r)

        with record_function("Classifier"):
            return self.Classifier(self.ITD_Encoder(ITD))


class DuckyIldOnsetModel(nn.Module):
    """ILD-only model using onset window.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.onset_slicer = OnsetSlicer(sr=sr, avg_window=0.005, slice_window=0.050)
        self.ild_module = ILD(eps=1e-8)
        self.ild_normaliser = ILDNormaliser()
        self.ILD_Encoder = (
            _make_ild_onset_encoder()
        )  # (B, 1, F, t_window//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)

        _, _, x_l_avg_onset, x_r_avg_onset = self.onset_slicer(x_l, x_r)

        ILD_map = _compute_ild_map(
            self.ild_module, self.ild_normaliser, x_l_avg_onset, x_r_avg_onset
        )  # (B, 1, F, t_window//stride)

        with record_function("Classifier"):
            return self.Classifier(self.ILD_Encoder(ILD_map))


class DuckyIldWholeModel(nn.Module):
    """ILD-only model using whole signal.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.avg_ild = AvgILD(avg=int(0.005 * sr), eps=1e-8)
        self.ild_normaliser = ILDNormaliser()
        self.ILD_Encoder = _make_ild_whole_encoder()  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)

        ILD_map = _compute_ild_map(
            self.avg_ild, self.ild_normaliser, x_l, x_r
        )  # (B, 1, F, T//stride)

        with record_function("Classifier"):
            return self.Classifier(self.ILD_Encoder(ILD_map))


class DuckyItdIldWholeModel(nn.Module):
    """ITD + ILD, both computed on whole signal.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(
            sr_res=sr, phy_itd_range=config["DUCKY"]["CORR"]["phy_ITD_range"]
        )
        self.itd_normaliser = ITDNormaliser()
        self.avg_ild = AvgILD(avg=int(0.005 * sr), eps=1e-8)
        self.ild_normaliser = ILDNormaliser()

        self.ITD_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)
        self.ILD_Encoder = _make_ild_whole_encoder()  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)

        ITD_map = _compute_itd_map(
            self.CORR, self.itd_normaliser, x_l, x_r
        )  # (B, 1, F, ITD_bins)
        ILD_map = _compute_ild_map(
            self.avg_ild, self.ild_normaliser, x_l, x_r
        )  # (B, 1, F, T//stride)

        feat = torch.cat(
            [
                self.ITD_Encoder(ITD_map),
                self.ILD_Encoder(ILD_map),
            ],
            dim=-1,
        )  # (B, 512)

        with record_function("Classifier"):
            return self.Classifier(feat)


class DuckyModel(nn.Module):
    """Experiment 3: ITD + ILD, both computed on the onset window.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.onset_slicer = OnsetSlicer(sr=sr, avg_window=0.005, slice_window=0.050)
        self.CORR = Correlagram(
            sr_res=sr, phy_itd_range=config["DUCKY"]["CORR"]["phy_ITD_range"]
        )
        self.itd_normaliser = ITDNormaliser()
        self.ild_module = ILD(eps=1e-8)
        self.ild_normaliser = ILDNormaliser()

        self.ITD_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)
        self.ILD_Encoder = (
            _make_ild_onset_encoder()
        )  # (B, 1, F, t_window//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)  # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)  # (B, F, T)

        with record_function("CORR"):
            x_l_onset, x_r_onset, x_l_avg, x_r_avg = self.onset_slicer(x_l, x_r)

        ITD_map = _compute_itd_map(
            self.CORR, self.itd_normaliser, x_l_onset, x_r_onset
        )  # (B, 1, F, ITD_bins)
        ILD_map = _compute_ild_map(
            self.ild_module, self.ild_normaliser, x_l_avg, x_r_avg
        )  # (B, 1, F, t_window//stride)

        with record_function("ITD_Encoder"):
            ITD_feature = self.ITD_Encoder(ITD_map)  # (B, 256)
        with record_function("ILD_Encoder"):
            ILD_feature = self.ILD_Encoder(ILD_map)  # (B, 256)

        with record_function("Classifier"):
            return self.Classifier(torch.cat([ITD_feature, ILD_feature], dim=-1))


class DuckyItdMultiScaleModel(nn.Module):
    """ITD only, onset-window + whole-signal encoders (multiscale). FC: [512, 256].
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.onset_slicer = OnsetSlicer(sr=sr, avg_window=0.005, slice_window=0.050)
        self.CORR = Correlagram(
            sr_res=sr, phy_itd_range=config["DUCKY"]["CORR"]["phy_ITD_range"]
        )
        self.itd_normaliser = ITDNormaliser()

        self.ITD_onset_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)
        self.ITD_whole_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)  # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)  # (B, F, T)

        x_l_onset, x_r_onset, _, _ = self.onset_slicer(x_l, x_r)

        ITD_onset = _compute_itd_map(
            self.CORR, self.itd_normaliser, x_l_onset, x_r_onset
        )  # (B, 1, F, ITD_bins)
        ITD_whole = _compute_itd_map(
            self.CORR, self.itd_normaliser, x_l, x_r
        )  # (B, 1, F, ITD_bins)

        feat = torch.cat(
            [
                self.ITD_onset_Encoder(ITD_onset),
                self.ITD_whole_Encoder(ITD_whole),
            ],
            dim=-1,
        )  # (B, 512)

        with record_function("Classifier"):
            return self.Classifier(feat)


class DuckyIldMultiScaleModel(nn.Module):
    """ILD only, onset-window + whole-signal encoders (multiscale). FC: [512, 256].
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.onset_slicer = OnsetSlicer(sr=sr, avg_window=0.005, slice_window=0.050)
        self.ild_module = ILD(eps=1e-8)
        self.avg_ild = AvgILD(avg=int(0.005 * sr), eps=1e-8)
        self.ild_normaliser = ILDNormaliser()

        self.ILD_onset_Encoder = (
            _make_ild_onset_encoder()
        )  # (B, 1, F, t_window//stride) → (B, 256)
        self.ILD_whole_Encoder = (
            _make_ild_whole_encoder()
        )  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)  # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)  # (B, F, T)

        _, _, x_l_avg_onset, x_r_avg_onset = self.onset_slicer(x_l, x_r)

        ILD_onset = _compute_ild_map(
            self.ild_module, self.ild_normaliser, x_l_avg_onset, x_r_avg_onset
        )  # (B, 1, F, t_window//stride)
        ILD_whole = _compute_ild_map(
            self.avg_ild, self.ild_normaliser, x_l, x_r
        )  # (B, 1, F, T//stride)

        feat = torch.cat(
            [
                self.ILD_onset_Encoder(ILD_onset),
                self.ILD_whole_Encoder(ILD_whole),
            ],
            dim=-1,
        )  # (B, 512)

        with record_function("Classifier"):
            return self.Classifier(feat)


class DuckyMultiScaleModel(nn.Module):
    """Experiment 4: ITD + ILD, each with onset-window and whole-signal encoders.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        sr = config["DUCKY"]["SR"]
        num_classes = config["NUM_CLASSES"]

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.onset_slicer = OnsetSlicer(sr=sr, avg_window=0.005, slice_window=0.050)
        self.CORR = Correlagram(
            sr_res=sr, phy_itd_range=config["DUCKY"]["CORR"]["phy_ITD_range"]
        )
        self.itd_normaliser = ITDNormaliser()
        self.ild_module = ILD(eps=1e-8)
        self.avg_ild = AvgILD(avg=int(0.005 * sr), eps=1e-8)
        self.ild_normaliser = ILDNormaliser()

        self.ITD_onset_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)
        self.ITD_whole_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)
        self.ILD_onset_Encoder = (
            _make_ild_onset_encoder()
        )  # (B, 1, F, t_window//stride) → (B, 256)
        self.ILD_whole_Encoder = (
            _make_ild_whole_encoder()
        )  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, :, 0].unsqueeze(1)  # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)  # (B, F, T)

        x_l_onset, x_r_onset, x_l_avg_onset, x_r_avg_onset = self.onset_slicer(x_l, x_r)

        ITD_onset = _compute_itd_map(
            self.CORR, self.itd_normaliser, x_l_onset, x_r_onset
        )  # (B, 1, F, ITD_bins)
        ITD_whole = _compute_itd_map(
            self.CORR, self.itd_normaliser, x_l, x_r
        )  # (B, 1, F, ITD_bins)
        ILD_onset = _compute_ild_map(
            self.ild_module, self.ild_normaliser, x_l_avg_onset, x_r_avg_onset
        )  # (B, 1, F, t_window//stride)
        ILD_whole = _compute_ild_map(
            self.avg_ild, self.ild_normaliser, x_l, x_r
        )  # (B, 1, F, T//stride)

        feat = torch.cat(
            [
                self.ITD_onset_Encoder(ITD_onset),
                self.ITD_whole_Encoder(ITD_whole),
                self.ILD_onset_Encoder(ILD_onset),
                self.ILD_whole_Encoder(ILD_whole),
            ],
            dim=-1,
        )  # (B, 1024)

        with record_function("Classifier"):
            return self.Classifier(feat)
