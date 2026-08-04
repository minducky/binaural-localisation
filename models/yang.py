import os
import numpy as np
from scipy.special import factorial
from scipy.signal import filtfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TAT
from scipy.signal import butter

def print_gpu_mem(tag=""):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{tag}] Allocated: {alloc:.2f}GB / Reserved: {reserved:.2f}GB")

def fft_filt(x, a, b, analytic):
    # a, b shape should be [n_filters (n_channel), order]
    batch, channel, time = x.shape

    nfft = 2 ** (time.bit_length() + 1)
    freqs = torch.fft.fftfreq(nfft, device=x.device, dtype=torch.float64)
    z_inv = torch.exp(-2j * torch.pi * freqs)  # z^-1 = e^(-jw) = e^(-2j*pi*freq)
    a_powers = z_inv.unsqueeze(0) ** torch.arange(a.shape[1], device=x.device).unsqueeze(-1)  # (a.shape[1], nfft) = (1, nfft) ** (a.shape[1], 1)
    # print(a.dtype, a_powers.dtype)
    H_den = a @ a_powers  # (freq_bin, nfft) = (freq_bin, a.shape[1]) @ (a.shape[1], nfft)
    b_powers = z_inv.unsqueeze(0) ** torch.arange(b.shape[1], device=x.device).unsqueeze(-1)
    H_num = b @ b_powers  # (freq_bin)
    H = H_num / H_den  # (freq_bin, nfft)

    X = torch.fft.fft(x, nfft)
    Y = X * H.unsqueeze(0)
    output = torch.fft.ifft(Y, n=nfft, dim=-1)[..., :time].real.float()

    if analytic:
        output *= 2
    return output


class YangOM(nn.Module):
    '''Yang Outer Middle Ear'''
    def __init__(self,  filter_coeff_dir, sr=44100, trainable=False):
        super().__init__()
        self.sr = torch.tensor(sr)
        self.trainable = trainable
        a_fpath = os.path.join(filter_coeff_dir, f"yang_a_om_{sr}.pth")
        b_fpath = os.path.join(filter_coeff_dir, f"yang_b_om_{sr}_yang.pth")
        if os.path.exists(a_fpath) and os.path.exists(b_fpath):
            print(f"Loading OM filter...")
            a = torch.load(a_fpath)
            b = torch.load(b_fpath)
        else:
            print(f"Making OM filter...")
            a, b = self.om_filter_coeff(sr=sr, freq_range=[1000, 4000], order=1)
            torch.save(a, a_fpath)
            torch.save(b, b_fpath)

        self.a = nn.Parameter(a, requires_grad=trainable)
        self.b = nn.Parameter(b, requires_grad=trainable)
        self.freq_bin = self.a.shape[0]

    def om_filter_coeff(self, sr=44100, freq_range=[1000, 4000], order=1):
        b, a = butter(order, freq_range, btype='bandpass', fs=sr, output='ba')
        b = torch.tensor(b, dtype=torch.complex128).unsqueeze(0)
        a = torch.tensor(a, dtype=torch.complex128).unsqueeze(0)

        return a, b

    def forward(self, x):
        return fft_filt(x, self.a, self.b, analytic=False)


class YangBM(nn.Module):
    '''Yang Basilar Membrane Layer'''
    def __init__(self, filter_coeff_dir, sr=44100, trainable=False):
        super().__init__()
        a_fpath = os.path.join(filter_coeff_dir, f'yang_a_bm_{sr}.pth')
        b_fpath = os.path.join(filter_coeff_dir, f'yang_b_bm_{sr}.pth')
        if os.path.exists(a_fpath) and os.path.exists(b_fpath):
            print(f'Loading BM filter...')
            a = torch.load(a_fpath)
            b = torch.load(b_fpath)
        else:
            print(f'Making BM filter...')
            a, b = self.bm_filter_coeff(sr=sr, freq_range=[125, 16000], num_channel=31, order=4)
            torch.save(a, a_fpath)
            torch.save(b, b_fpath)

        self.a = nn.Parameter(a, requires_grad=trainable)
        self.b = nn.Parameter(b, requires_grad=trainable)
        self.freq_bin = self.a.shape[0]

    def _audspace_bw(self, fmin, fmax, num_channel, bw=torch.tensor(1.0, dtype=torch.float64)):
        audlimits = self._freq_to_aud(torch.tensor([fmin, fmax], dtype=torch.float64))
        audrange = audlimits[1] - audlimits[0]
        # n = int(torch.floor(audrange / bw))
        # remainder = audrange - n * bw
        # audpoints = audlimits[0] + torch.arange(n + 1) * bw + remainder / 2
        audpoints = torch.linspace(audlimits[0], audlimits[1], num_channel, dtype=torch.float64)
        n = num_channel
        print(f'audpoints : {audpoints}')
        y = self._aud_to_freq(audpoints)
        print(f'center freqs : {y}')
        return y, n

    def _freq_to_aud(self, freq):
        return 9.2645 * torch.log(1 + freq * 0.00437)

    def _aud_to_freq(self, aud):
        return 1 / 0.00437 * (torch.exp(aud / 9.2645) - 1)

    def _calculate_betamul(self, n):
        betamul = (factorial(n - 1) ** 2) / (
                np.pi * factorial(2 * n - 2) * 2 ** (-(2 * n - 2))
        )
        return torch.tensor(betamul)

    def bm_filter_coeff(self, sr=44100, freq_range=[125, 16000], num_channel=31, order=4):
        fc, n_filters = self._audspace_bw(freq_range[0], freq_range[1], num_channel)
        n = order
        betamul = self._calculate_betamul(n)

        ERB = 24.7 + fc / 9.265
        ourbeta = betamul * ERB

        b = torch.zeros((n_filters, 1), dtype=torch.complex128)
        a = torch.zeros((n_filters, n + 1), dtype=torch.complex128)

        for i in range(n_filters):
            theta = 2 * np.pi * fc[i].item() / sr
            phi = 2 * np.pi * ourbeta[i].item() / sr
            atilde = np.exp(-phi - 1j * theta)
            poles = atilde * np.ones(n)
            a[i, :] = torch.tensor(np.poly(poles), dtype=torch.complex128)
            btmp = 1 - np.exp(-phi)
            b[i] = btmp ** n

        return a, b

    def forward(self, x):
        return fft_filt(x, self.a, self.b, analytic=True)


class YangIHC(nn.Module):
    '''Yang Inner Hair Cell Layer'''
    def __init__(self, filter_coeff_dir, orig_sr=44100, target_sr=4000, trainable=False):
        super().__init__()
        a_fpath = os.path.join(filter_coeff_dir, f'yang_a_ihc_{orig_sr}.pth')
        b_fpath = os.path.join(filter_coeff_dir, f'yang_b_ihc_{orig_sr}.pth')
        if os.path.exists(a_fpath) and os.path.exists(b_fpath):
            print(f'Loading IHC filter...')
            a = torch.load(a_fpath)
            b = torch.load(b_fpath)
        else:
            print(f'Making IHC filter...')
            a, b = self.ihc_filter_coeff(sr=orig_sr)
            torch.save(a, a_fpath)
            torch.save(b, b_fpath)

        self.a = nn.Parameter(a, requires_grad=trainable)
        self.b = nn.Parameter(b, requires_grad=trainable)
        self.resampler = TAT.Resample(orig_sr, target_sr)

    def ihc_filter_coeff(self, sr=44100):
        b, a = butter(5, 1000, btype='low', fs=sr, output='ba')
        b = torch.tensor(b, dtype=torch.complex128).unsqueeze(0)
        a = torch.tensor(a, dtype=torch.complex128).unsqueeze(0)
        return a, b

    def _downsample(self, x):
        return self.resampler(x)

    def _power_law(self, x):
        return torch.pow(torch.nn.functional.relu(x), 0.3)

    def forward(self, x):
        x = torch.nn.functional.relu(x)
        x = fft_filt(x, self.a, self.b, analytic=False)
        x = self._downsample(x)
        x = self._power_law(x)

        return x


class YangEI(nn.Module):
    def __init__(self, sr=4000, tau_range=5e-3, tau_taps_num=24,
                 alpha_range=10.0, alpha_taps_num=12, temporal_window=10.0e-3,
                 trainable=False):
        super().__init__()
        self.sr = sr

        self.tau_taps = nn.Parameter(
            torch.linspace(-tau_range, tau_range, tau_taps_num, dtype=torch.float64),
            requires_grad=trainable
        )
        self.alpha_taps = nn.Parameter(
            torch.linspace(-alpha_range, alpha_range, alpha_taps_num, dtype=torch.float64),
            requires_grad=trainable
        )

        # Exponential smoothing window — matches ei_pattern_func.py
        M = int(6 * 0.03 * sr)
        n_win = torch.arange(-(M // 2), M // 2 + 1, dtype=torch.float64)
        window = torch.exp(-torch.abs(n_win) / (sr * 0.03))
        window = window / window.sum()
        self.register_buffer('window', window)  # (M,)

    def _fft_smooth(self, x):
        """FFT convolution along last dim, equivalent to fftconvolve(mode='same')."""
        n = x.shape[-1]
        M = self.window.shape[0]
        N = 2 ** (n + M - 1).bit_length()
        W = torch.fft.rfft(self.window, N)
        y = torch.fft.irfft(torch.fft.rfft(x, N, dim=-1) * W, N, dim=-1)
        return y[..., M // 2 : M // 2 + n]

    def forward(self, coch_l, coch_r):
        # coch_l, coch_r: (batch, ch, n_samples)
        batch, n_ch, n_samples = coch_l.shape

        # shift = |tau_sec| * sr / 2  — matches numpy where shift = |tau_samples| / 2
        shifts = (self.tau_taps.abs() * self.sr / 2).round().long()  # (n_tau,)
        max_shift = int(shifts.max().item())

        l_pad = F.pad(coch_l.double(), (max_shift, max_shift))
        r_pad = F.pad(coch_r.double(), (max_shift, max_shift))

        # Start index in padded tensor for each tau:
        #   tau > 0 → left delayed (max_shift - shift), right not (max_shift)
        #   tau < 0 → right delayed, left not
        l_starts = torch.where(self.tau_taps > 0,
                               max_shift - shifts,
                               torch.full_like(shifts, max_shift))  # (n_tau,)
        r_starts = torch.where(self.tau_taps < 0,
                               max_shift - shifts,
                               torch.full_like(shifts, max_shift))  # (n_tau,)

        # All shifted versions at once via unfold: (batch, ch, 2*max_shift+1, n_samples)
        unf_l = l_pad.unfold(2, n_samples, 1)
        unf_r = r_pad.unfold(2, n_samples, 1)

        left_delayed  = unf_l[:, :, l_starts, :]   # (batch, ch, n_tau, n_samples)
        right_delayed = unf_r[:, :, r_starts, :]

        # Expand (aL - bR)^2 = a^2*L^2 - 2*L*R + b^2*R^2  (a*b = 10^(α/40-α/40) = 1)
        # → smooth only 3 terms then combine with alpha, instead of smoothing n_alpha copies
        to_smooth = torch.stack([left_delayed.pow(2),
                                 left_delayed * right_delayed,
                                 right_delayed.pow(2)], dim=3)  # (batch, ch, n_tau, 3, n_samples)
        smoothed = self._fft_smooth(to_smooth)

        sLL = smoothed[:, :, :, 0, :]  # (batch, ch, n_tau, n_samples)
        sLR = smoothed[:, :, :, 1, :]
        sRR = smoothed[:, :, :, 2, :]

        a2 = 10 ** (self.alpha_taps / 20.0)   # (n_alpha,)
        b2 = 10 ** (-self.alpha_taps / 20.0)  # (n_alpha,)

        # p_tau: 10^(-|tau_samples| / (fs*0.005)) = 10^(-|tau_sec| / 0.005)
        p_taus = 10 ** (-self.tau_taps.abs() / 0.005)  # (n_tau,)

        # Loop over alpha to avoid (batch, ch, n_tau, n_samples, n_alpha) peak memory
        ei_parts = []
        for i in range(len(a2)):
            ei_i = sLL * a2[i] + sRR * b2[i] - 2.0 * sLR  # (batch, ch, n_tau, n_samples)
            ei_i = ei_i.clamp(min=0).sqrt()
            ei_i = ei_i * p_taus.view(1, 1, -1, 1)
            # Temporal integration per alpha: RMS over time
            ei_i = torch.sqrt(torch.mean(ei_i ** 2, dim=-1))  # (batch, ch, n_tau)
            ei_parts.append(ei_i)

        ei = torch.stack(ei_parts, dim=-1).float()  # (batch, ch, n_tau, n_alpha)
        return ei

# ----------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------- #
class YangModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        sr = config['YANG']['SR']
        sr_res = config['YANG']['SR_RES']
        input_dim = config['YANG']['INPUT_DIM']    # tau_taps * alpha_taps * freq_bins
        hidden_dim_1 = config['YANG']['HIDDEN_DIM_1']
        hidden_dim_2 = config['YANG']['HIDDEN_DIM_2']
        output_dim = config['NUM_CLASSES']            # 72 * 7 / 19 * 5
        filter_coeff_dir = config['FILTER_COEFF_DIR']
        os.makedirs(filter_coeff_dir, exist_ok=True)
        self.outermiddle = YangOM(filter_coeff_dir=filter_coeff_dir, sr=sr, trainable=False)
        self.basilar = YangBM(filter_coeff_dir=filter_coeff_dir, trainable=False)
        self.ihc = YangIHC(filter_coeff_dir=filter_coeff_dir, orig_sr=sr, target_sr=sr_res, trainable=False)
        self.ei = YangEI(sr=sr_res, trainable=False)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):               # (B, 44100, 2)
        x_l = x[:, :, 0].unsqueeze(1)   # (B, 1, 44100)
        x_r = x[:, :, 1].unsqueeze(1)   # (B, 1, 44100)

        coch_l = self.ihc(self.basilar(self.outermiddle(x_l)))
        coch_r = self.ihc(self.basilar(self.outermiddle(x_r)))
        ei_out = self.ei(coch_l, coch_r)
        ei_out = ei_out.to(torch.float32)

        output = ei_out.flatten(start_dim=1)
        output = self.dropout(self.relu(self.bn1(self.fc1(output))))
        output = self.dropout(self.relu(self.bn2(self.fc2(output))))
        output = self.fc3(output)

        return output


class YangMLP(nn.Module):
    """MLP-only model that takes precomputed EI features as input.

    Same MLP architecture as YangModel (8928 -> 75 * 14 -> 25 * 14 -> 512)
    but without the cochlear pipeline, for use with precomputed EI features.
    """

    def __init__(self, config):
        super().__init__()
        input_dim = config["YANGMLP"]["YANG_INPUT_DIM"]      # 31 * 24 * 12 = 8928
        hidden_dim_1 = config["YANGMLP"]["HIDDEN_DIM_1"]     # 75 * 14 = 1050
        hidden_dim_2 = config["YANGMLP"]["HIDDEN_DIM_2"]     # 25 * 14 = 350
        output_dim = config["NUM_CLASSES"]

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x: (batch, 31, 24, 12) precomputed EI features
        output = x.flatten(start_dim=1)  # (batch, 8928)
        output = self.dropout(self.relu(self.bn1(self.fc1(output))))
        output = self.dropout(self.relu(self.bn2(self.fc2(output))))
        output = self.fc3(output)
        return output


