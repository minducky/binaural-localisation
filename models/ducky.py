from torch.profiler import record_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.peripheral import (
    gammatone_filter_coefficients, ihc_coefficients, filt_tch,
    audspace_bw, calculate_betamul,
)


class BM(nn.Module):
    def __init__(self, fmin, fmax, bw, sr, order, filter_coeff_dir,
                 learnable_coefficients=True, fc_min_bound=50.0, fc_max_bound=20000.0,
                 ourbeta_min=10.0, ourbeta_max=2500.0):
        super().__init__()
        self.sr = sr
        self.order = order
        self.learnable_coefficients = learnable_coefficients
        self.fc_min_bound = fc_min_bound
        self.fc_max_bound = fc_max_bound
        self.ourbeta_min = ourbeta_min
        self.ourbeta_max = ourbeta_max

        if learnable_coefficients:
            fc_np, _ = audspace_bw(fmin, fmax, bw)
            betamul = calculate_betamul(order)
            ERB = 24.7 + fc_np / 9.265
            ourbeta_np = betamul * ERB

            def _inv_sig(x):
                return np.log(x / (1.0 - x))

            fc_norm = (fc_np - self.fc_min_bound) / (self.fc_max_bound - self.fc_min_bound)
            ourbeta_norm = (ourbeta_np - ourbeta_min) / (ourbeta_max - ourbeta_min)

            self.raw_fc = nn.Parameter(
                torch.tensor(_inv_sig(fc_norm), dtype=torch.float64))
            self.raw_ourbeta = nn.Parameter(
                torch.tensor(_inv_sig(ourbeta_norm), dtype=torch.float64))
        else:
            self.fc, b, a = gammatone_filter_coefficients(
                fmin, fmax, bw, sr, order=order, filter_coeff_dir=filter_coeff_dir)
            self.register_buffer('b', b)
            self.register_buffer('a', a)

    def _compute_coefficients(self):
        fc = torch.sigmoid(self.raw_fc) * (self.fc_max_bound - self.fc_min_bound) + self.fc_min_bound
        ourbeta = torch.sigmoid(self.raw_ourbeta) * (self.ourbeta_max - self.ourbeta_min) + self.ourbeta_min

        phi = 2 * torch.pi * ourbeta / self.sr
        theta = 2 * torch.pi * fc / self.sr

        atilde = torch.exp(torch.complex(-phi, -theta))  # (F,) complex128

        a = torch.stack([
            torch.ones_like(atilde),
            -4 * atilde,
            6 * atilde ** 2,
            -4 * atilde ** 3,
            atilde ** 4,
        ], dim=-1)  # (F, 5)

        b_real = (1 - torch.exp(-phi)) ** self.order    # (F,)
        b = b_real.to(a.dtype).unsqueeze(-1)            # (F, 1)

        return a, b

    def get_learned_params(self):
        """Return current fc and ourbeta values as numpy arrays. Only valid when learnable_coefficients=True."""
        with torch.no_grad():
            fc = (torch.sigmoid(self.raw_fc) * (self.fc_max_bound - self.fc_min_bound) + self.fc_min_bound).cpu().numpy()
            ourbeta = (torch.sigmoid(self.raw_ourbeta) * (self.ourbeta_max - self.ourbeta_min) + self.ourbeta_min).cpu().numpy()
        return {'fc': fc, 'ourbeta': ourbeta}

    def forward(self, x):
        if self.learnable_coefficients:
            a, b = self._compute_coefficients()
            return filt_tch(x, a, b)
        return filt_tch(x, self.a, self.b)


class IHC(nn.Module):
    def __init__(self, fcut, sr, order, filter_coeff_dir):
        super().__init__()
        b, a = ihc_coefficients(fcut=fcut, sr=sr, order=order, filter_coeff_dir=filter_coeff_dir)
        self.register_buffer('b', b)
        self.register_buffer('a', a)

    def forward(self, x):
        x = F.relu(x)
        x = torch.sqrt(x)
        return filt_tch(x, self.a, self.b)


class PeripheralFrontend(nn.Module):
    """Shared BM + IHC pipeline. Processes left and right channels together."""
    def __init__(self, config):
        super().__init__()
        bm_config = config['DUCKY']['BM']
        ihc_config = config['DUCKY']['IHC']
        sr = config['DUCKY']['SR']
        filter_coeff_dir = config['FILTER_COEFF_DIR']
        layers = config['DUCKY']['LAYERS']

        self.module_list = nn.ModuleList()
        self.module_names = []

        if 'BM' in layers:
            self.module_list.append(BM(
                fmin=bm_config['fmin'], fmax=bm_config['fmax'], bw=bm_config['bw'],
                sr=sr, order=bm_config['order'], filter_coeff_dir=filter_coeff_dir,
                learnable_coefficients=bm_config['learnable_coefficients'],
                fc_min_bound=bm_config['fc_min_bound'],
                fc_max_bound=bm_config['fc_max_bound'],
                ourbeta_min=bm_config['ourbeta_min'],
                ourbeta_max=bm_config['ourbeta_max'],
            ))
            self.module_names.append('BM')
        if 'IHC' in layers:
            self.module_list.append(IHC(
                fcut=ihc_config['fcut'], sr=sr, order=ihc_config['order'],
                filter_coeff_dir=filter_coeff_dir,
            ))
            self.module_names.append('IHC')

    def forward(self, x_l, x_r):
        for name, layer in zip(self.module_names, self.module_list):
            with record_function(name):
                x_l = layer(x_l)
                x_r = layer(x_r)
        return x_l, x_r


class Correlagram(nn.Module):
    def __init__(self, sr, phy_ITD_range, eps=1e-8):
        super().__init__()
        self.phy_ITD_range_in_samples = [int(i * sr) for i in phy_ITD_range]
        self.eps = eps

    def forward(self, peri_l, peri_r):
        # peri_l/r: (B, F, T)
        B, Freq, T = peri_l.shape

        ITD_bins = self.phy_ITD_range_in_samples[1] - self.phy_ITD_range_in_samples[0] + 1

        gamma_l_pad = F.pad(peri_l, (-self.phy_ITD_range_in_samples[0], self.phy_ITD_range_in_samples[1]))
        T_pad = gamma_l_pad.shape[-1]
        nfft = 2 ** (T_pad + T - 1).bit_length()

        L = torch.fft.rfft(gamma_l_pad, n=nfft)
        R = torch.fft.rfft(peri_r, n=nfft)

        corr = torch.fft.irfft(L * R.conj(), n=nfft)[..., :ITD_bins]   # (B, F, ITD_bins)

        norm_l = peri_l.pow(2).sum(dim=-1, keepdim=True).sqrt()
        norm_r = peri_r.pow(2).sum(dim=-1, keepdim=True).sqrt()
        corr = corr / (norm_l * norm_r + self.eps)

        return corr.unsqueeze(1)   # (B, 1, F, ITD_bins)


class ILD(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, peri_l, peri_r):
        # peri_l/r: (B, F, T)
        ild_db = 20.0 * (torch.log10(peri_l.clamp(min=self.eps))
                         - torch.log10(peri_r.clamp(min=self.eps)))
        return ((ild_db / 20.0).clamp(-1.0, 1.0)).unsqueeze(1)   # (B, 1, F, T)


def detect_onset_and_slice(x_l, x_r, stride, t_window):
    """
    Detects sound onset and slices both original and avg-pooled signals to the onset window.

    Args:
        x_l, x_r: (B, F, T)
        stride:   avg-pool window/stride in samples (e.g. int(0.005 * sr))
        t_window: slice length in samples (e.g. int(0.05 * sr))
    Returns:
        x_l_slice, x_r_slice:         (B, F, t_window)
        x_l_avg_slice, x_r_avg_slice: (B, F, t_window // stride)
    """
    B, Freq, T = x_l.shape

    # avg_pool2d requires 4D input
    x_l_avg = F.avg_pool2d(x_l.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)
    x_r_avg = F.avg_pool2d(x_r.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)
    # (B, F, T_avg)

    x_l_diff = torch.diff(x_l_avg, dim=-1)                      # (B, F, T_avg-1)
    x_r_diff = torch.diff(x_r_avg, dim=-1)
    x_mean = x_l_diff.mean(dim=-2) * x_r_diff.mean(dim=-2)      # (B, T_avg-1)
    skip = 1
    onset_frame = torch.argmax(x_mean[:, skip:], dim=-1) + skip  # (B,)

    t_window_avg = t_window // stride

    onset_sample = torch.clamp(onset_frame * stride, 0, T - t_window)
    onset_frame  = torch.clamp(onset_frame,           0, x_l_avg.shape[-1] - t_window_avg)

    offsets = torch.arange(t_window, device=x_l.device)
    indices = (onset_sample[:, None] + offsets[None, :])[:, None, :].expand(B, Freq, t_window)
    x_l_slice = torch.gather(x_l, -1, indices)
    x_r_slice = torch.gather(x_r, -1, indices)

    offsets_avg = torch.arange(t_window_avg, device=x_l.device)
    indices_avg = (onset_frame[:, None] + offsets_avg[None, :])[:, None, :].expand(B, Freq, t_window_avg)
    x_l_avg_slice = torch.gather(x_l_avg, -1, indices_avg)
    x_r_avg_slice = torch.gather(x_r_avg, -1, indices_avg)

    return x_l_slice, x_r_slice, x_l_avg_slice, x_r_avg_slice


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------

def _make_itd_encoder():
    """Conv encoder for ITD map (B, 1, F, ITD_bins) → (B, 256)."""
    return nn.Sequential(
        # (B, 1, 71, 133)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 32, 35, 66)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 64, 17, 33)
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 128, 8, 16)
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 256, 4, 8)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 256, 2, 4)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        # (B, 256, 1, 1)
        nn.Flatten(),
        # (B, 256)
    )


def _make_ild_onset_encoder():
    """Conv encoder for onset ILD map (B, 1, F, t_window//stride) → (B, 256)."""
    return nn.Sequential(
        # (B, 1, 71, 10)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d((2, 1), (2, 1)),
        # (B, 32, 35, 10)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d((2, 1), (2, 1)),
        # (B, 64, 17, 10)
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(), nn.AvgPool2d((2, 1), (2, 1)),
        # (B, 128, 8, 10)
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 256, 4, 5)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 256, 2, 2)
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2, 2),
        # (B, 256, 1, 1)
        nn.Flatten(),
        # (B, 256)
    )


def _make_ild_whole_encoder():
    """Conv encoder for whole-signal ILD map (B, 1, F, T//stride) → (B, 256).
    AdaptiveAvgPool2d handles variable time length.
    """
    return nn.Sequential(
        # (B, 1, F, T//stride)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(), nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d((2, 2), (2, 2)),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d((2, 2), (2, 2)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # (B, 256)
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DuckyItdModel(nn.Module):
    """Experiments 1 & 2: ITD-only model with optional onset detection.
    Config: DUCKY.ONSET = False (exp 1) or True (exp 2).
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr
        self.use_onset = config['DUCKY']['ONSET']

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(sr=sr, phy_ITD_range=config['DUCKY']['CORR']['phy_ITD_range'])

        self.ITD_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)   # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)   # (B, F, T)

        if self.use_onset:
            stride   = int(0.005 * self.sr)
            t_window = int(0.050 * self.sr)
            x_l, x_r, _, _ = detect_onset_and_slice(x_l, x_r, stride, t_window)

        with record_function("CORR"):
            ITD = self.CORR(x_l, x_r)   # (B, 1, F, ITD_bins)

        with record_function("Classifier"):
            return self.Classifier(self.ITD_Encoder(ITD))


class DuckyIldOnsetModel(nn.Module):
    """ILD-only model using onset window.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.ILD = ILD()
        self.ILD_Encoder = _make_ild_onset_encoder()  # (B, 1, F, t_window//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)

        stride   = int(0.005 * self.sr)
        t_window = int(0.050 * self.sr)
        _, _, x_l_avg_onset, x_r_avg_onset = detect_onset_and_slice(x_l, x_r, stride, t_window)

        ILD_map = self.ILD(x_l_avg_onset, x_r_avg_onset)  # (B, 1, F, t_window//stride)

        with record_function("Classifier"):
            return self.Classifier(self.ILD_Encoder(ILD_map))


class DuckyIldWholeModel(nn.Module):
    """ILD-only model using whole signal.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.ILD = ILD()
        self.ILD_Encoder = _make_ild_whole_encoder()  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)

        stride = int(0.005 * self.sr)
        x_l_avg = F.avg_pool2d(x_l.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)
        x_r_avg = F.avg_pool2d(x_r.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)

        ILD_map = self.ILD(x_l_avg, x_r_avg)  # (B, 1, F, T//stride)

        with record_function("Classifier"):
            return self.Classifier(self.ILD_Encoder(ILD_map))


class DuckyItdIldWholeModel(nn.Module):
    """ITD + ILD, both computed on whole signal.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(sr=sr, phy_ITD_range=config['DUCKY']['CORR']['phy_ITD_range'])
        self.ILD = ILD()

        self.ITD_Encoder = _make_itd_encoder()        # (B, 1, F, ITD_bins) → (B, 256)
        self.ILD_Encoder = _make_ild_whole_encoder()  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)

        stride = int(0.005 * self.sr)
        x_l_avg = F.avg_pool2d(x_l.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)
        x_r_avg = F.avg_pool2d(x_r.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)

        ITD_map = self.CORR(x_l, x_r)             # (B, 1, F, ITD_bins)
        ILD_map = self.ILD(x_l_avg, x_r_avg)      # (B, 1, F, T//stride)

        feat = torch.cat([
            self.ITD_Encoder(ITD_map),
            self.ILD_Encoder(ILD_map),
        ], dim=-1)  # (B, 512)

        with record_function("Classifier"):
            return self.Classifier(feat)


class DuckyModel(nn.Module):
    """Experiment 3: ITD + ILD, both computed on the onset window.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(sr=sr, phy_ITD_range=config['DUCKY']['CORR']['phy_ITD_range'])
        self.ILD = ILD()

        self.ITD_Encoder = _make_itd_encoder()         # (B, 1, F, ITD_bins) → (B, 256)
        self.ILD_Encoder = _make_ild_onset_encoder()   # (B, 1, F, t_window//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)   # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)   # (B, F, T)

        stride   = int(0.005 * self.sr)
        t_window = int(0.050 * self.sr)

        with record_function("CORR"):
            x_l_onset, x_r_onset, x_l_avg, x_r_avg = detect_onset_and_slice(
                x_l, x_r, stride, t_window
            )

        ITD_map = self.CORR(x_l_onset, x_r_onset)   # (B, 1, F, ITD_bins)
        ILD_map = self.ILD(x_l_avg, x_r_avg)         # (B, 1, F, t_window//stride)

        with record_function("ITD_Encoder"):
            ITD_feature = self.ITD_Encoder(ITD_map)   # (B, 256)
        with record_function("ILD_Encoder"):
            ILD_feature = self.ILD_Encoder(ILD_map)   # (B, 256)

        with record_function("Classifier"):
            return self.Classifier(torch.cat([ITD_feature, ILD_feature], dim=-1))


class DuckyItdMultiScaleModel(nn.Module):
    """ITD only, onset-window + whole-signal encoders (multiscale). FC: [512, 256].
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(sr=sr, phy_ITD_range=config['DUCKY']['CORR']['phy_ITD_range'])

        self.ITD_onset_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)
        self.ITD_whole_Encoder = _make_itd_encoder()  # (B, 1, F, ITD_bins) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)   # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)   # (B, F, T)

        stride   = int(0.005 * self.sr)
        t_window = int(0.050 * self.sr)

        x_l_onset, x_r_onset, _, _ = detect_onset_and_slice(x_l, x_r, stride, t_window)

        ITD_onset = self.CORR(x_l_onset, x_r_onset)  # (B, 1, F, ITD_bins)
        ITD_whole = self.CORR(x_l, x_r)              # (B, 1, F, ITD_bins)

        feat = torch.cat([
            self.ITD_onset_Encoder(ITD_onset),
            self.ITD_whole_Encoder(ITD_whole),
        ], dim=-1)   # (B, 512)

        with record_function("Classifier"):
            return self.Classifier(feat)


class DuckyIldMultiScaleModel(nn.Module):
    """ILD only, onset-window + whole-signal encoders (multiscale). FC: [512, 256].
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.ILD = ILD()

        self.ILD_onset_Encoder = _make_ild_onset_encoder()  # (B, 1, F, t_window//stride) → (B, 256)
        self.ILD_whole_Encoder = _make_ild_whole_encoder()  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)   # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)   # (B, F, T)

        stride   = int(0.005 * self.sr)
        t_window = int(0.050 * self.sr)

        _, _, x_l_avg_onset, x_r_avg_onset = detect_onset_and_slice(x_l, x_r, stride, t_window)
        x_l_avg_whole = F.avg_pool2d(x_l.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)
        x_r_avg_whole = F.avg_pool2d(x_r.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)

        ILD_onset = self.ILD(x_l_avg_onset, x_r_avg_onset)  # (B, 1, F, t_window//stride)
        ILD_whole = self.ILD(x_l_avg_whole, x_r_avg_whole)  # (B, 1, F, T//stride)

        feat = torch.cat([
            self.ILD_onset_Encoder(ILD_onset),
            self.ILD_whole_Encoder(ILD_whole),
        ], dim=-1)   # (B, 512)

        with record_function("Classifier"):
            return self.Classifier(feat)


class DuckyMultiScaleModel(nn.Module):
    """Experiment 4: ITD + ILD, each with onset-window and whole-signal encoders.
    Input: (B, T, 2) stereo audio → Output: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        sr = config['DUCKY']['SR']
        num_classes = config['NUM_CLASSES']

        self.sr = sr

        self.peripheral = PeripheralFrontend(config)
        self.CORR = Correlagram(sr=sr, phy_ITD_range=config['DUCKY']['CORR']['phy_ITD_range'])
        self.ILD = ILD()

        self.ITD_onset_Encoder = _make_itd_encoder()        # (B, 1, F, ITD_bins) → (B, 256)
        self.ITD_whole_Encoder = _make_itd_encoder()        # (B, 1, F, ITD_bins) → (B, 256)
        self.ILD_onset_Encoder = _make_ild_onset_encoder()  # (B, 1, F, t_window//stride) → (B, 256)
        self.ILD_whole_Encoder = _make_ild_whole_encoder()  # (B, 1, F, T//stride) → (B, 256)

        self.Classifier = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_l = x[:, :, 0].unsqueeze(1)   # (B, 1, T)
        x_r = x[:, :, 1].unsqueeze(1)

        x_l, x_r = self.peripheral(x_l, x_r)   # (B, F, T)

        stride   = int(0.005 * self.sr)
        t_window = int(0.050 * self.sr)

        x_l_onset, x_r_onset, x_l_avg_onset, x_r_avg_onset = detect_onset_and_slice(
            x_l, x_r, stride, t_window
        )
        x_l_avg_whole = F.avg_pool2d(x_l.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)
        x_r_avg_whole = F.avg_pool2d(x_r.unsqueeze(1), kernel_size=(1, stride), stride=(1, stride)).squeeze(1)

        ITD_onset = self.CORR(x_l_onset, x_r_onset)         # (B, 1, F, ITD_bins)
        ITD_whole = self.CORR(x_l, x_r)                     # (B, 1, F, ITD_bins)
        ILD_onset = self.ILD(x_l_avg_onset, x_r_avg_onset)  # (B, 1, F, t_window//stride)
        ILD_whole = self.ILD(x_l_avg_whole, x_r_avg_whole)  # (B, 1, F, T//stride)

        feat = torch.cat([
            self.ITD_onset_Encoder(ITD_onset),
            self.ITD_whole_Encoder(ITD_whole),
            self.ILD_onset_Encoder(ILD_onset),
            self.ILD_whole_Encoder(ILD_whole),
        ], dim=-1)   # (B, 1024)

        with record_function("Classifier"):
            return self.Classifier(feat)
