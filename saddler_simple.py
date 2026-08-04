# todo : 1. resample first from sr_input to sr_cochlea  -> implemented (FIRResample)
# todo : 2. fir gammatone filterbank (util_signal.fir_gammatone_filterbank)   -> implemented
# todo : 3. half-wave rectifictiaon (relu)                                    -> implemented (F.relu in forward)
# todo : 4. resample from sr_cochlea to sr_output                             -> implemented (FIRResample)
# todo : 5. power compression (no)                                            -> skip
# todo : 6. custom slice (no)                                                 -> skip
# todo : 7. util_cochlea.sigmoid_rate_level_function                          -> implemented
# todo : 8. add_noise (no)                                                    -> skip
# todo : 9. spike generator binomial -> Just SpikeGeneratorBinomial           -> implemented
# todo : 10. No correlogram                                                   -> skip

"""
PyTorch port of the Saddler et al. peripheral auditory model pipeline.

Pipeline (Simple_Peripheral_Saddler):
    [batch, time]  @ sr_input
        -> FIRResample             (FIR anti-aliasing, sr_input → sr_cochlea)  [batch, time']
        -> FIRGammatoneFilterbank  (causal conv1d, learnable weights)           [batch, n_cf, time']
        -> Half-wave rectification (ReLU)                                        [batch, n_cf, time']
        -> FIRResample             (FIR anti-aliasing, sr_cochlea → sr_output)  [batch, n_cf, time'']
        -> SigmoidRateLevelFunction (ANF model, learnable params)               [batch, n_cf, time'', n_ch]
        -> SpikeGeneratorBinomial                                                [batch, n_cf, time'', n_ch]

Reference implementations:
    util_signal.fir_gammatone_filterbank  (TF, Slaney 1998 Auditory Toolbox)
    util_cochlea.sigmoid_rate_level_function
    util_cochlea.SpikeGeneratorBinomial
"""

import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ERB / frequency helpers  (torch versions, used at __init__ time only)
# ============================================================================

def freq2erb_tch(freq):
    """Hz -> ERB-number (Cams).  Glasberg & Moore (1990) eq. 4.  freq: scalar tensor"""
    return 21.4 * torch.log10(0.00437 * freq + 1.0)


def erb2freq_tch(erb):
    """ERB-number (Cams) -> Hz.  Glasberg & Moore (1990) eq. 4.  erb: scalar tensor"""
    return (1.0 / 0.00437) * (10.0 ** (erb / 21.4) - 1.0)


def erbspace_tch(freq_min, freq_max, num):
    """Tensor of `num` frequencies linearly spaced on the ERB-number scale (Hz)."""
    erb_min = freq2erb_tch(torch.tensor(freq_min, dtype=torch.float64))
    erb_max = freq2erb_tch(torch.tensor(freq_max, dtype=torch.float64))
    erbs = torch.linspace(erb_min.item(), erb_max.item(), num, dtype=torch.float64)
    return erb2freq_tch(erbs)   # (num,) tensor


# ============================================================================
# Gammatone filterbank helpers  (torch versions, used at __init__ time only)
# ============================================================================

def get_gammatone_filter_coefs_tch(sr, fc, EarQ=9.2644, minBW=24.7, order=1):
    """
    4th-order gammatone IIR filter coefficients (torch).
    Returns list of 4 dicts, each with 'b' (3, n_cf) and 'a' (3, n_cf) tensors.
    """
    if not isinstance(fc, torch.Tensor):
        fc = torch.tensor(fc, dtype=torch.float64)
    print(f'fc dtype : {fc.dtype}')
    T   = 1.0 / sr
    T   = torch.tensor(T, dtype=torch.float64)
    ERB = ((fc / EarQ) ** order + minBW ** order) ** (1.0 / order)
    print(f'ERB dtype : {ERB.dtype}')
    B   = 1.019 * 2.0 * torch.pi * ERB

    A0  = T * torch.ones_like(fc)
    A2  = torch.zeros_like(fc)
    B0  = torch.ones_like(fc)
    B1  = -2.0 * torch.cos(2.0 * fc * torch.pi * T) / torch.exp(B * T)
    B2  = torch.exp(-2.0 * B * T)

    tmp0 = 2.0 * T * torch.cos(2.0 * fc * torch.pi * T) / torch.exp(B * T)
    tmp1 = T * torch.sin(2.0 * fc * torch.pi * T) / torch.exp(B * T)
    A11  = -(tmp0 + 2.0 * math.sqrt(3.0 + 2.0 ** 1.5) * tmp1) / 2.0
    A12  = -(tmp0 - 2.0 * math.sqrt(3.0 + 2.0 ** 1.5) * tmp1) / 2.0
    A13  = -(tmp0 + 2.0 * math.sqrt(3.0 - 2.0 ** 1.5) * tmp1) / 2.0
    A14  = -(tmp0 - 2.0 * math.sqrt(3.0 - 2.0 ** 1.5) * tmp1) / 2.0

    fc_c = fc.to(torch.complex128)
    B_c  = B.to(torch.complex128)
    tmp2 = torch.exp(4j * fc_c * torch.pi * T)
    tmp3 = 2.0 * torch.exp(-B_c * T + 2j * fc_c * torch.pi * T) * T
    tmp4 = torch.cos(2.0 * fc_c * torch.pi * T)
    tmp5 = torch.sin(2.0 * fc_c * torch.pi * T)
    sq1  = math.sqrt(3.0 - 2.0 ** (3.0 / 2.0))
    sq2  = math.sqrt(3.0 + 2.0 ** (3.0 / 2.0))
    gain = torch.abs(
        (-2.0 * tmp2 * T + tmp3 * (tmp4 - sq1 * tmp5)) *
        (-2.0 * tmp2 * T + tmp3 * (tmp4 + sq1 * tmp5)) *
        (-2.0 * tmp2 * T + tmp3 * (tmp4 - sq2 * tmp5)) *
        (-2.0 * tmp2 * T + tmp3 * (tmp4 + sq2 * tmp5)) /
        (-2.0 / torch.exp(2.0 * B_c * T) - 2.0 * tmp2 + 2.0 * (1.0 + tmp2) / torch.exp(B_c * T)) ** 4
    ).double()

    filter_coefs = [
        {'b': torch.stack([A0 / gain, A11 / gain,  A2]), 'a': torch.stack([B0, B1, B2])},
        {'b': torch.stack([A0,        A12,         A2]), 'a': torch.stack([B0, B1, B2])},
        {'b': torch.stack([A0,        A13,         A2]), 'a': torch.stack([B0, B1, B2])},
        {'b': torch.stack([A0,        A14,         A2]), 'a': torch.stack([B0, B1, B2])},
    ]
    return filter_coefs


def _filt(signal, a_coeff, b_coeff):
    """
    Frequency-domain IIR filter applied to real signal.

    Args
        signal  : (batch, n_cf, time)  float64 tensor
        a_coeff : (n_cf, n_taps)       float64 tensor (denominator)
        b_coeff : (n_cf, n_taps)       float64 tensor (numerator)

    Returns
        output  : (batch, n_cf, time)  float64 tensor
    """
    batch, channel, time = signal.shape
    nfft = 2 ** time.bit_length()

    freqs = torch.fft.rfftfreq(nfft, device=signal.device).double()
    z_inv = torch.exp(-2j * torch.pi * freqs)                           # (nfft//2+1,) complex128

    # (n_cf, n_taps) cast to complex128 for matmul with complex128 a_powers
    a_c = a_coeff.to(torch.complex128)
    b_c = b_coeff.to(torch.complex128)

    n_freqs = z_inv.shape[0]
    # z^{-k} for k = 0..n_taps-1, shape: (n_taps, n_freqs)
    k_a = torch.arange(a_c.shape[1], device=signal.device).unsqueeze(-1).double()
    k_b = torch.arange(b_c.shape[1], device=signal.device).unsqueeze(-1).double()
    a_powers = z_inv.unsqueeze(0) ** k_a    # (n_taps, n_freqs) complex128
    b_powers = z_inv.unsqueeze(0) ** k_b    # (n_taps, n_freqs) complex128

    H_den = a_c @ a_powers    # (n_cf, n_freqs) complex128
    H_num = b_c @ b_powers    # (n_cf, n_freqs) complex128
    H     = H_num / H_den     # (n_cf, n_freqs) complex128

    X      = torch.fft.rfft(signal.double(), nfft)         # (batch, n_cf, n_freqs) complex128
    Y      = X * H.unsqueeze(0)                             # (batch, n_cf, n_freqs)
    output = torch.fft.irfft(Y, n=nfft, dim=-1)[..., :time]  # (batch, n_cf, time) float64
    return output


def get_gammatone_impulse_responses_tch(sr, fir_dur, fc, **kwargs):
    """
    FIR approximation of gammatone filterbank via impulse responses (pure torch).

    Returns
    -------
    impulse_responses (torch.Tensor): shape (n_cf, n_samples)  float32
    """
    if not isinstance(fc, torch.Tensor):
        fc = torch.tensor(fc, dtype=torch.float64)
    n_cf      = len(fc)
    n_samples = int(fir_dur * sr)

    # Impulse: (1, n_cf, T)
    impulse = torch.zeros(1, n_cf, n_samples, dtype=torch.float64)
    impulse[0, :, 0] = 1.0

    filter_coefs = get_gammatone_filter_coefs_tch(sr, fc, **kwargs)
    for stage in filter_coefs:
        a = stage['a'].T.contiguous()   # (n_cf, 3)
        b = stage['b'].T.contiguous()   # (n_cf, 3)
        impulse = _filt(impulse, a, b)  # (1, n_cf, T)

    return impulse.squeeze(0).float()   # (n_cf, T)


# ============================================================================
# Gammatone filterbank helpers  (numpy/scipy, for reference tests only)
# ============================================================================

def get_gammatone_filter_coefs(sr, cfs, EarQ=9.2644, minBW=24.7, order=1):
    """numpy version — used in equivalence tests only."""
    cfs = np.asarray(cfs)
    T   = 1 / sr
    ERB = ((cfs / EarQ) ** order + minBW ** order) ** (1 / order)
    B   = 1.019 * 2 * np.pi * ERB

    A0  = T * np.ones_like(cfs)
    A2  = 0 * np.ones_like(cfs)
    B0  = 1 * np.ones_like(cfs)
    B1  = -2 * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    B2  = np.exp(-2 * B * T)

    tmp0 = 2 * T * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    tmp1 = T * np.sin(2 * cfs * np.pi * T) / np.exp(B * T)
    A11  = -(tmp0 + 2 * np.sqrt(3 + 2 ** 1.5) * tmp1) / 2
    A12  = -(tmp0 - 2 * np.sqrt(3 + 2 ** 1.5) * tmp1) / 2
    A13  = -(tmp0 + 2 * np.sqrt(3 - 2 ** 1.5) * tmp1) / 2
    A14  = -(tmp0 - 2 * np.sqrt(3 - 2 ** 1.5) * tmp1) / 2

    tmp2 = np.exp(4 * 1j * cfs * np.pi * T)
    tmp3 = 2 * np.exp(-(B * T) + 2 * 1j * cfs * np.pi * T) * T
    tmp4 = np.cos(2 * cfs * np.pi * T)
    tmp5 = np.sin(2 * cfs * np.pi * T)
    gain = np.abs(
        (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 - 2 ** (3/2)) * tmp5)) *
        (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 - 2 ** (3/2)) * tmp5)) *
        (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 + 2 ** (3/2)) * tmp5)) *
        (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 + 2 ** (3/2)) * tmp5)) /
        (-2 / np.exp(2 * B * T) - 2 * tmp2 + 2 * (1 + tmp2) / np.exp(B * T)) ** 4
    )
    return [
        {'b': np.array([A0, A11, A2]) / gain, 'a': np.array([B0, B1, B2])},
        {'b': np.array([A0, A12, A2]),          'a': np.array([B0, B1, B2])},
        {'b': np.array([A0, A13, A2]),          'a': np.array([B0, B1, B2])},
        {'b': np.array([A0, A14, A2]),          'a': np.array([B0, B1, B2])},
    ]


def scipy_gammatone_filterbank(x, filter_coefs):
    """numpy/scipy IIR reference — used in equivalence tests only."""
    if x.ndim == 1:
        x_subbands = x[np.newaxis, np.newaxis, :]   # (1, 1, T)
        squeeze = True
    elif x.ndim == 2:
        x_subbands = x[:, np.newaxis, :]             # (B, 1, T)
        squeeze = False
    else:
        raise ValueError("Expected input shape [time] or [batch, time]")

    n_cf = filter_coefs[0]['b'].shape[-1]
    x_subbands = np.tile(x_subbands, [1, n_cf, 1])   # (B, n_cf, T)

    for fc in filter_coefs:
        for i in range(n_cf):
            x_subbands[:, i, :] = scipy.signal.lfilter(
                fc['b'][:, i], fc['a'][:, i], x_subbands[:, i, :], axis=-1)

    if squeeze:
        x_subbands = x_subbands[0]   # (n_cf, T)
    return x_subbands


# ============================================================================
# FIR Gammatone filterbank nn.Module  (learnable filter weights)
# ============================================================================

class FIRGammatoneFilterbank(nn.Module):
    """
    FIR Gammatone filterbank as a causal Conv1d layer with learnable weights.
    Port of util_signal.fir_gammatone_filterbank().

    Weights are initialised from gammatone impulse responses (Slaney 1998) and
    stored as nn.Parameter so they can be fine-tuned end-to-end.

    Input shape:  [batch, time]
    Output shape: [batch, n_cf, time]
    """

    def __init__(self,
                 sr,
                 fir_dur,
                 fc=None,
                 min_cf=125.0,
                 max_cf=8e3,
                 num_cf=50,
                 kwargs_filter_coefs={}):
        super().__init__()

        if fc is None:
            fc = erbspace_tch(min_cf, max_cf, num_cf)   # (num_cf,) tensor

        self.num_cf = len(fc)
        self.fc     = fc

        # Compute impulse responses (torch, one-time only)
        fir = get_gammatone_impulse_responses_tch(sr, fir_dur, fc, **kwargs_filter_coefs)
        # fir: (n_cf, T) float32 tensor
        self.kernel_size = fir.shape[-1]

        # Conv1d weight: (n_cf, 1, kernel_size)
        # Flip so Conv1d cross-correlation matches scipy lfilter convolution.
        weight_init = fir.unsqueeze(1).flip(-1).contiguous()   # (n_cf, 1, T) float32
        self.weight = nn.Parameter(weight_init)

    def forward(self, x):
        """
        Args
            x: [batch, time]  or  [time]

        Returns
            y: [batch, n_cf, time]
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)                        # (1, T)
        x = x.unsqueeze(1)                            # (B, 1, T)
        x = F.pad(x, (self.kernel_size - 1, 0))       # causal left-padding
        return F.conv1d(x, self.weight, padding=0)    # (B, n_cf, T)


# ============================================================================
# SigmoidRateLevelFunction nn.Module  (learnable ANF parameters)
# ============================================================================

class SigmoidRateLevelFunction(nn.Module):
    """
    Generalised sigmoid auditory-nerve rate-level function with learnable parameters.
    Port of util_cochlea.sigmoid_rate_level_function().

    rate_spont, rate_max, threshold, dynamic_range are nn.Parameter and can be
    jointly optimised with the rest of the network.

    Input shape:  [batch, n_cf, time]  or  [batch, n_cf, time, n_ch]
    Output shape: same  (units: spikes/s)
    """

    def __init__(self,
                 n_channels=1,
                 rate_spont=70.0,
                 rate_max=250.0,
                 threshold=0.0,
                 dynamic_range=25.0,
                 dynamic_range_interval=0.95,
                 envelope_mode=False):
        super().__init__()

        def _init(v):
            if isinstance(v, (list, tuple)):
                return torch.tensor(v, dtype=torch.float64)
            return torch.full((n_channels,), float(v), dtype=torch.float64)

        self.rate_spont    = nn.Parameter(_init(rate_spont))
        self.rate_max      = nn.Parameter(_init(rate_max))
        self.threshold     = nn.Parameter(_init(threshold))
        self.dynamic_range = nn.Parameter(_init(dynamic_range))

        self.dynamic_range_interval = dynamic_range_interval
        self.envelope_mode          = envelope_mode

    def forward(self, tensor_subbands):
        # Derive sigmoid slope k and inflection x0 from learnable parameters
        y_thr   = (1.0 - self.dynamic_range_interval) / 2.0
        log_val = torch.log(
            torch.tensor(1.0 / y_thr - 1.0,
                         dtype=tensor_subbands.dtype,
                         device=tensor_subbands.device)
        )
        k  = log_val / (self.dynamic_range / 2.0)    # (n_channels,)
        x0 = self.threshold + log_val / k             # (n_channels,)

        n_ch = self.rate_spont.shape[0]

        if tensor_subbands.ndim == 3:
            if n_ch == 1:
                rs  = self.rate_spont.reshape(1, 1, 1)
                rm  = self.rate_max.reshape(1, 1, 1)
                k_  = k.reshape(1, 1, 1)
                x0_ = x0.reshape(1, 1, 1)
            else:
                tensor_subbands = tensor_subbands.unsqueeze(-1)   # [B, n_cf, T, n_ch]
                rs  = self.rate_spont.reshape(1, 1, 1, -1)
                rm  = self.rate_max.reshape(1, 1, 1, -1)
                k_  = k.reshape(1, 1, 1, -1)
                x0_ = x0.reshape(1, 1, 1, -1)
        else:    # ndim == 4
            rs  = self.rate_spont.reshape(1, 1, 1, -1)
            rm  = self.rate_max.reshape(1, 1, 1, -1)
            k_  = k.reshape(1, 1, 1, -1)
            x0_ = x0.reshape(1, 1, 1, -1)

        if self.envelope_mode:
            env = _hilbert_envelope(tensor_subbands, axis=2)
            tfs = torch.where(env > 0,
                              tensor_subbands / env,
                              torch.zeros_like(tensor_subbands))
            pa  = env
        else:
            pa = tensor_subbands

        log10 = torch.log(torch.tensor(10.0, dtype=pa.dtype, device=pa.device))
        x_db  = 20.0 * torch.log(pa.clamp(min=1e-12) / 2e-5) / log10
        y     = torch.sigmoid(k_ * (x_db - x0_))

        if self.envelope_mode:
            y = y * tfs

        return rs + (rm - rs) * y


def _hilbert_envelope(x, axis=-1):
    """FFT-based analytic signal magnitude. Pure torch."""
    n = x.shape[axis]
    # X = torch.fft.fft(x, n=n, dim=axis)
    # h = torch.zeros(n, dtype=x.dtype, device=x.device)

    X = torch.fft.fft(x.to(torch.complex64), n=n, dim=axis)
    h = torch.zeros(n, dtype=x.dtype, device=x.device)
    if n % 2 == 0:
        h[0] = torch.ones(1, dtype=X.dtype); h[1:n // 2] = 2 * torch.ones(n // 2 - 1, dtype=X.dtype); h[n // 2] = torch.ones(1, dtype=X.dtype)
    # else:
    #     h[0] = 1; h[1:(n + 1) // 2] = 2
    shape = [1] * len(x.shape)
    shape[axis] = n
    analytic = torch.fft.ifft(X * h.reshape(shape), dim=axis)
    return torch.abs(analytic).to(x.dtype)


# ============================================================================
# SpikeGeneratorBinomial  (pure torch forward pass)
# ============================================================================

class SpikeGeneratorBinomial_Tch(nn.Module):
    """
    Stochastic binomial spike generator.
    Port of util_cochlea.SpikeGeneratorBinomial.

    Input shape:  [batch, n_cf, time]  or  [batch, n_cf, time, n_ch]
    Output shape: same  (spike counts, same dtype as input)
    """

    def __init__(self,
                 sr,
                 n_per_channel=1,
                 mode='exact',
                 p_dtype=torch.float32,
                 p_noise_stddev=None,
                 channels_to_manipulate=None,
                 channelwise_manipulation='shuffle',
                 seed=0):
        super().__init__()
        self.sr = sr
        n_list = [n_per_channel] if isinstance(n_per_channel, int) else list(n_per_channel)
        self.register_buffer(
            'n_per_channel',
            torch.tensor(n_list, dtype=torch.float32).reshape(1, 1, 1, -1)
        )
        self.mode = mode.lower()
        assert self.mode in ['approx', 'exact']
        self.p_dtype                  = p_dtype
        self.p_noise_stddev           = p_noise_stddev if p_noise_stddev is not None else 0.0
        self.channels_to_manipulate   = channels_to_manipulate or []
        self.channelwise_manipulation = channelwise_manipulation
        torch.manual_seed(seed)

    def forward(self, inputs):
        tensor_spike_probs = inputs.to(self.p_dtype) / self.sr

        if tensor_spike_probs.ndim < 4:
            tensor_spike_probs = tensor_spike_probs.unsqueeze(-1)

        if self.p_noise_stddev > 0:
            noise = torch.normal(
                mean=torch.zeros_like(tensor_spike_probs),
                std=torch.full_like(tensor_spike_probs, self.p_noise_stddev),
            )
            tensor_spike_probs = F.relu(tensor_spike_probs + noise)

        if self.channels_to_manipulate:
            manip = self.channelwise_manipulation.lower()
            if manip == 'shuffle':
                channels = list(tensor_spike_probs.unbind(dim=-1))
                for ch in self.channels_to_manipulate:
                    tmp  = channels[ch]
                    flat = tmp.reshape(tmp.shape[0], -1)
                    perm = torch.stack([
                        torch.randperm(flat.shape[1], device=flat.device)
                        for _ in range(flat.shape[0])
                    ])
                    channels[ch] = flat.gather(1, perm).reshape(tmp.shape)
                tensor_spike_probs = torch.stack(channels, dim=-1)
            elif manip == 'silence':
                n_ch = tensor_spike_probs.shape[-1]
                mask = torch.ones(n_ch,
                                  dtype=tensor_spike_probs.dtype,
                                  device=tensor_spike_probs.device)
                for ch in self.channels_to_manipulate:
                    mask[ch] = 0.0
                tensor_spike_probs = tensor_spike_probs * mask.reshape(1, 1, 1, -1)
            else:
                raise ValueError(
                    f"channelwise_manipulation={self.channelwise_manipulation!r} not recognized"
                )

        if self.mode == 'exact':
            dist = torch.distributions.Binomial(
                total_count=self.n_per_channel,
                probs=tensor_spike_probs.clamp(0.0, 1.0),
            )
            tensor_spike_counts = dist.sample()
        else:
            p    = tensor_spike_probs
            mean = self.n_per_channel * p
            std  = torch.sqrt(self.n_per_channel * p * (1.0 - p))
            tensor_spike_counts = F.relu(torch.round(torch.normal(mean=mean, std=std)))

        return tensor_spike_counts.to(inputs.dtype)


# ============================================================================
# FIR Lowpass filter helper + FIRResample nn.Module
# Port of util_signal_ducky.fir_lowpass_filter / tf_fir_resample
# ============================================================================

def fir_lowpass_filter(sr_input, sr_output, numtaps=None, fir_dur=None,
                       cutoff=None, order=None, ihc_filter=False,
                       window=('kaiser', 5.0), verbose=False):
    """
    Build a FIR lowpass filter for anti-aliasing during resampling.
    Pure numpy/scipy — port of util_signal_ducky.fir_lowpass_filter.

    Specify exactly one of [numtaps, fir_dur].
    """
    none_args = [_ for _ in [numtaps, fir_dur] if _ is None]
    assert len(none_args) == 1, "Specify exactly one of [numtaps, fir_dur]"
    if sr_output is None:
        sr_output = sr_input
    gcd     = np.gcd(int(sr_input), int(sr_output))
    down    = int(sr_input)  // gcd
    up      = int(sr_output) // gcd
    sr_filt = sr_input * up
    if ihc_filter:
        assert cutoff is not None, "cutoff required for ihc_filter"
        assert order  is not None, "order required for ihc_filter"
    if cutoff is None:
        cutoff = sr_output / 2
    if fir_dur is None:
        fir_dur = int(2 * (numtaps // 2)) / sr_filt
    else:
        numtaps = int(2 * (fir_dur * sr_filt // 2)) + 1   # ensure odd
    assert cutoff <= sr_output / 2, "cutoff may not exceed Nyquist"
    if ihc_filter:
        # FIR approximation of IHC lowpass filter (bez2018model)
        impulse = np.zeros(numtaps)
        filt    = np.zeros_like(impulse)
        impulse[0] = 1
        ihc  = np.zeros(order + 1)
        ihcl = np.zeros(order + 1)
        c1LP = (sr_filt - 2 * np.pi * cutoff) / (sr_filt + 2 * np.pi * cutoff)
        c2LP = (np.pi * cutoff)                / (sr_filt + 2 * np.pi * cutoff)
        for n in range(len(impulse)):
            ihc[0] = impulse[n]
            for i in range(order):
                ihc[i + 1] = c1LP * ihcl[i + 1] + c2LP * (ihc[i] + ihcl[i])
            ihcl = ihc.copy()
            filt[n] = ihc[order]
        filt = filt * scipy.signal.windows.hann(len(filt))
        filt = filt / filt.sum()
    else:
        filt = scipy.signal.firwin(
            numtaps=numtaps, cutoff=cutoff, width=None,
            window=tuple(window), pass_zero=True, scale=True, fs=sr_filt)
    if verbose:
        print(f"[fir_lowpass_filter] sr_filt={sr_filt} Hz, numtaps={numtaps}, "
              f"fir_dur={fir_dur:.6f} s, cutoff={cutoff} Hz")
    return filt, sr_filt


class FIRResample(nn.Module):
    """
    FIR-based resampling: upsample (zero-insertion) → lowpass filter → downsample.
    Port of util_signal_ducky.tf_fir_resample.

    Input shape:  [batch, time]  or  [batch, channels, time]
    Output shape: same ndim, resampled along time axis

    The lowpass filter kernel is a fixed (non-learnable) buffer.
    """

    def __init__(self, sr_input, sr_output, kwargs_fir_lowpass_filter={}, verbose=False):
        super().__init__()
        gcd       = np.gcd(int(sr_input), int(sr_output))
        self.up   = int(sr_output) // gcd
        self.down = int(sr_input)  // gcd

        kw = dict(kwargs_fir_lowpass_filter)
        if kw.get('cutoff', None) is None:
            kw['cutoff'] = sr_output / 2

        filt, _ = fir_lowpass_filter(sr_input, sr_output, **kw, verbose=verbose)
        filt = filt * self.up   # rescale to offset attenuation from upsampling

        self.ihc_filter = kw.get('ihc_filter', False)
        if self.ihc_filter:
            filt = filt[::-1].copy()   # time-reverse for IHC causal filter

        self.filter_len = len(filt)
        self.register_buffer('filt', torch.tensor(filt, dtype=torch.float32))

    def forward(self, x):
        """
        Args
            x: [batch, time]  or  [batch, channels, time]
        Returns
            resampled tensor with same number of dimensions
        """
        # Identity shortcut
        if self.up == 1 and self.down == 1:
            return x

        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)   # (B, 1, T)
            squeeze = True

        B, C, T = x.shape

        # --- Step 1: Upsample via zero-insertion ---
        if self.up > 1:
            x_up = x.new_zeros(B, C, T * self.up)
            x_up[..., ::self.up] = x
            x = x_up   # (B, C, T*up)

        # --- Step 2: Build depthwise conv kernel (C, 1, filter_len) ---
        weight = self.filt.view(1, 1, -1).repeat(C, 1, 1)

        # --- Step 3: Lowpass filter + downsample (strided depthwise conv1d) ---
        T_up = x.shape[-1]
        if self.ihc_filter:
            # VALID conv after manual left-pad (matching TF behaviour)
            x = F.pad(x, (self.filter_len - 1, 0))
            x = F.conv1d(x, weight, stride=self.down, padding=0, groups=C)
        else:
            # SAME padding — replicate TF's SAME semantics for strided conv
            out_len   = math.ceil(T_up / self.down)
            total_pad = max((out_len - 1) * self.down + self.filter_len - T_up, 0)
            pad_left  = total_pad // 2
            pad_right = total_pad - pad_left
            x = F.pad(x, (pad_left, pad_right))
            x = F.conv1d(x, weight, stride=self.down, padding=0, groups=C)

        if squeeze:
            x = x.squeeze(1)
        return x


# ============================================================================
# Full pipeline
# ============================================================================

class Simple_Peripheral_Saddler(nn.Module):
    """
    Simple peripheral auditory model following Saddler et al.

    Pipeline:
        [batch, time]
        -> FIRGammatoneFilterbank    [batch, n_cf, time]        (todo 2, learnable weights)
        -> Half-wave rectification   [batch, n_cf, time]        (todo 3)
        -> SigmoidRateLevelFunction  [batch, n_cf, time, n_ch]  (todo 7, learnable params)
        -> SpikeGeneratorBinomial    [batch, n_cf, time, n_ch]  (todo 9)
    """

    def __init__(self,
                 sr_input=20e3,
                 sr_cochlea=20e3,
                 sr_output=20e3,
                 fir_dur=0.05,
                 min_cf=125.0,
                 max_cf=8e3,
                 num_cf=50,
                 kwargs_fir_resample_cochlea=None,
                 kwargs_fir_resample_output=None,
                 kwargs_sigmoid=None,
                 kwargs_spike_gen=None):
        super().__init__()

        # todo 1: resample sr_input → sr_cochlea
        self.resample_to_cochlea = FIRResample(
            sr_input, sr_cochlea, kwargs_fir_resample_cochlea or {}
        )
        # todo 2: gammatone filterbank at sr_cochlea
        self.filterbank = FIRGammatoneFilterbank(
            sr=sr_cochlea, fir_dur=fir_dur, min_cf=min_cf, max_cf=max_cf, num_cf=num_cf,
        )
        # todo 4: resample sr_cochlea → sr_output (applied per subband)
        self.resample_to_output = FIRResample(
            sr_cochlea, sr_output, kwargs_fir_resample_output or {}
        )
        self.sigmoid_fn = SigmoidRateLevelFunction(**(kwargs_sigmoid or {}))
        self.spike_gen  = SpikeGeneratorBinomial_Tch(sr=sr_output, **(kwargs_spike_gen or {}))

    def forward(self, x):
        """
        Args
            x: [batch, time]  @ sr_input

        Returns
            [batch, n_cf, time'', n_ch]  (spike counts)  @ sr_output
        """
        x = self.resample_to_cochlea(x)   # todo 1: [B, T']        @ sr_cochlea
        x = self.filterbank(x)            # todo 2: [B, n_cf, T']
        x = F.relu(x)                     # todo 3: half-wave rectification
        x = self.resample_to_output(x)    # todo 4: [B, n_cf, T''] @ sr_output
        x = self.sigmoid_fn(x)            # todo 7: [B, n_cf, T'', n_ch]
        x = self.spike_gen(x)             # todo 9: [B, n_cf, T'', n_ch]
        return x


# ============================================================================
# Numerical equivalence / shape tests
# ============================================================================

def run_filterbank_equivalence_test(sr=20e3, fir_dur=0.05, num_cf=10, t_samples=1000):
    """
    Compare FIRGammatoneFilterbank (torch) against scipy IIR reference.
    FIR is an approximation of IIR, so a small non-zero error is expected.
    """
    print("=" * 60)
    print("Equivalence: FIRGammatoneFilterbank vs scipy IIR reference")
    print("=" * 60)

    np.random.seed(42)
    fc   = erbspace_tch(125.0, 8e3, num_cf)               # (n_cf,) tensor
    x_np = np.random.randn(t_samples).astype(np.float32)

    # scipy IIR reference (numpy path)
    ref = scipy_gammatone_filterbank(x_np, get_gammatone_filter_coefs(sr, fc.numpy()))

    # PyTorch FIR
    fb  = FIRGammatoneFilterbank(sr=sr, fir_dur=fir_dur, fc=fc)
    x_t = torch.from_numpy(x_np)
    with torch.no_grad():
        out = fb(x_t).numpy()[0]    # (n_cf, T)

    abs_err = np.abs(out - ref)
    print(f"  FIR kernel size : {fb.kernel_size}")
    print(f"  Output shape    : {out.shape}")
    print(f"  Max  |FIR - IIR|: {abs_err.max():.4e}")
    print(f"  Mean |FIR - IIR|: {abs_err.mean():.4e}")
    print(f"  (FIR approximation of IIR; expect small non-zero error)")
    print()


def run_sigmoid_equivalence_test():
    """
    Compare SigmoidRateLevelFunction against a torch reference.
    Should pass with max error < 1e-3.
    """
    print("=" * 60)
    print("Equivalence: SigmoidRateLevelFunction vs torch reference")
    print("=" * 60)

    torch.manual_seed(42)
    x_t = torch.rand(2, 50, 100, dtype=torch.float32) * 2e-4 + 1e-6

    def _ref(x, rate_spont=70., rate_max=250., threshold=0., dynamic_range=25.):
        y_thr   = (1 - 0.95) / 2
        log_val = torch.log(torch.tensor(1.0 / y_thr - 1.0))
        k  = log_val / (dynamic_range / 2)
        x0 = threshold + log_val / k
        xdb = 20.0 * torch.log(x / 20e-6) / torch.log(torch.tensor(10.0))
        y   = 1.0 / (1.0 + torch.exp(-k * (xdb - x0)))
        return rate_spont + (rate_max - rate_spont) * y

    ref = _ref(x_t).float()

    module = SigmoidRateLevelFunction(
        n_channels=1, rate_spont=70., rate_max=250., threshold=0., dynamic_range=25.
    )
    with torch.no_grad():
        out = module(x_t)

    abs_err = (out - ref).abs()
    print(f"  Max absolute error : {abs_err.max():.3e}")
    print(f"  Mean absolute error: {abs_err.mean():.3e}")
    print(f"  {'PASS' if abs_err.max() < 1e-3 else 'FAIL'} (threshold 1e-3)")
    print()


def run_pipeline_shape_test():
    """End-to-end shape test for Simple_Peripheral_Saddler."""
    print("=" * 60)
    print("Shape test: Simple_Peripheral_Saddler forward pass")
    print("=" * 60)

    sr_input   = 44100
    sr_cochlea = 20000
    sr_output  = 10000
    B = 2
    T = int(sr_input * 0.5)    # 0.5 s @ sr_input

    model = Simple_Peripheral_Saddler(
        sr_input=sr_input,
        sr_cochlea=sr_cochlea,
        sr_output=sr_output,
        fir_dur=0.05,
        num_cf=50,
        kwargs_fir_resample_cochlea={'fir_dur': 0.05},
        kwargs_fir_resample_output={'fir_dur': 0.05},
        kwargs_sigmoid={
            'n_channels':    1,
            'rate_spont':    70.,
            'rate_max':      250.,
            'threshold':     0.,
            'dynamic_range': 25.,
        },
        kwargs_spike_gen={'n_per_channel': 1, 'mode': 'exact'},
    )

    x = torch.randn(B, T)
    with torch.no_grad():
        out = model(x)

    print(f"  Input  shape: {list(x.shape)}")
    print(f"  Output shape: {list(out.shape)}")
    print(f"  Unique spike values: {out.unique().tolist()}")
    print()


if __name__ == "__main__":
    run_filterbank_equivalence_test()
    run_sigmoid_equivalence_test()
    run_pipeline_shape_test()
