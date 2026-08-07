"""Saddler/Phaselocknet peripheral auditory model (cochlea -> auditory nerve).

Faithful port of the ``peripheral_model.py`` module from
``Phaselocknet_torch`` (the user's own PyTorch port of the Saddler et al.
sound-localization model). All stages are non-learnable, fixed DSP
(``register_buffer``, no ``nn.Parameter``): a gammatone filterbank, IHC
half-wave rectification + lowpass/downsample, a sigmoid auditory-nerve
rate-level function, and a stochastic binomial spike generator, followed by
a random crop that stays stochastic even at eval time (kept intentionally,
matching the original model rather than being made deterministic).
"""

import collections

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torchvision

# %% ERB / gammatone filter design helpers
# One-off setup calculations run at __init__ time to build fixed FIR filter
# taps, not per-forward-pass operations - plain functions, not nn.Module.


def freq2erb(freq: np.ndarray) -> np.ndarray:
    """Converts frequency in Hz to ERB-number (same as ``freqtoerb.m`` in the AMT)."""
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb: np.ndarray) -> np.ndarray:
    """Converts ERB-number to frequency in Hz (same as ``erbtofreq.m`` in the AMT)."""
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(freq_min: float, freq_max: float, num: int) -> np.ndarray:
    """Builds frequencies evenly spaced on an ERB-number scale (AMT ``erbspace.m``).

    Args:
        freq_min: Minimum frequency in Hz.
        freq_max: Maximum frequency in Hz.
        num: Number of frequencies (length of the returned array).

    Returns:
        ERB-spaced frequencies in Hz, lowest to highest.
    """
    return erb2freq(np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num))


def get_gammatone_filter_coefs(
    sr: float,
    cfs: np.ndarray,
    EarQ: float = 9.2644,
    minBW: float = 24.7,
    order: int = 1,
) -> list:
    """Computes 4th-order gammatone IIR filter coefficients.

    Based on ``MakeERBFilters.m`` / ``ERBFilterBank.m`` from Malcolm Slaney's
    Auditory Toolbox (1998).

    Args:
        sr: Sample rate in Hz.
        cfs: Center frequencies in Hz, shape (n_cf,).
        EarQ: Asymptotic filter quality at large frequencies.
        minBW: Minimum bandwidth for low-frequency channels.
        order: Filter order used in the bandwidth formula.

    Returns:
        List of 4 dicts (one per filter cascade stage), each with ``"b"``
        (numerator, shape (3, n_cf)) and ``"a"`` (denominator, shape (3, n_cf)).
    """
    T = 1 / sr
    ERB = ((cfs / EarQ) ** order + minBW**order) ** (1 / order)
    B = 1.019 * 2 * np.pi * ERB
    A0 = T * np.ones_like(cfs)
    A2 = 0 * np.ones_like(cfs)
    B0 = 1 * np.ones_like(cfs)
    B1 = -2 * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    tmp0 = 2 * T * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    tmp1 = T * np.sin(2 * cfs * np.pi * T) / np.exp(B * T)
    A11 = -(tmp0 + 2 * np.sqrt(3 + 2**1.5) * tmp1) / 2
    A12 = -(tmp0 - 2 * np.sqrt(3 + 2**1.5) * tmp1) / 2
    A13 = -(tmp0 + 2 * np.sqrt(3 - 2**1.5) * tmp1) / 2
    A14 = -(tmp0 - 2 * np.sqrt(3 - 2**1.5) * tmp1) / 2

    tmp2 = np.exp(4 * 1j * cfs * np.pi * T)
    tmp3 = 2 * np.exp(-(B * T) + 2 * 1j * cfs * np.pi * T) * T
    tmp4 = np.cos(2 * cfs * np.pi * T)
    tmp5 = np.sin(2 * cfs * np.pi * T)
    gain = np.abs(
        (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 - 2 ** (3 / 2)) * tmp5))
        * (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 - 2 ** (3 / 2)) * tmp5))
        * (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 + 2 ** (3 / 2)) * tmp5))
        * (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 + 2 ** (3 / 2)) * tmp5))
        / (-2 / np.exp(2 * B * T) - 2 * tmp2 + 2 * (1 + tmp2) / np.exp(B * T)) ** 4
    )

    filter_coefs = [
        {"b": np.array([A0, A11, A2]) / gain, "a": np.array([B0, B1, B2])},
        {"b": np.array([A0, A12, A2]), "a": np.array([B0, B1, B2])},
        {"b": np.array([A0, A13, A2]), "a": np.array([B0, B1, B2])},
        {"b": np.array([A0, A14, A2]), "a": np.array([B0, B1, B2])},
    ]
    return filter_coefs


def scipy_gammatone_filterbank(x: np.ndarray, filter_coefs: list) -> np.ndarray:
    """Filters a waveform through a gammatone filterbank via ``scipy.signal.lfilter``.

    Args:
        x: Waveform, shape (time,) or (batch, time).
        filter_coefs: Filter coefficients from `get_gammatone_filter_coefs`.

    Returns:
        Subband signals, shape (n_cf, time) or (batch, n_cf, time).
    """
    if len(x.shape) == 1:
        x_subbands = x[np.newaxis, np.newaxis, :]
    elif len(x.shape) == 2:
        x_subbands = x[:, np.newaxis, :]
    else:
        raise ValueError("Expected input shape [time] or [batch, time]")
    n_subbands = filter_coefs[0]["b"].shape[-1]
    x_subbands = np.tile(x_subbands, [1, n_subbands, 1])
    for fc in filter_coefs:
        for itr_subbands in range(n_subbands):
            x_subbands[:, itr_subbands, :] = scipy.signal.lfilter(
                fc["b"][:, itr_subbands],
                fc["a"][:, itr_subbands],
                x_subbands[:, itr_subbands, :],
                axis=-1,
            )
    if len(x.shape) == 1:
        x_subbands = x_subbands[0]
    return x_subbands


def get_gammatone_impulse_responses(
    sr: float,
    fir_dur: float,
    cfs: np.ndarray,
    EarQ: float = 9.2644,
    minBW: float = 24.7,
    order: int = 1,
) -> np.ndarray:
    """Computes FIR impulse responses approximating a gammatone filterbank.

    Returns:
        Impulse responses, shape (n_cf, sr * fir_dur).
    """
    impulse = np.zeros(int(fir_dur * sr))
    impulse[0] = 1
    filter_coefs = get_gammatone_filter_coefs(
        sr, cfs, EarQ=EarQ, minBW=minBW, order=order
    )
    impulse_responses = scipy_gammatone_filterbank(impulse, filter_coefs)
    return impulse_responses


def ihc_lowpass_filter_fir(
    sr: float, fir_dur: float, cutoff: float = 3e3, order: int = 7
) -> np.ndarray:
    """Computes the FIR impulse response of the IHC lowpass filter.

    Ported from ``bez2018model/model_IHC_BEZ2018.c``. This is the filter
    whose cutoff frequency names the ``simplified_IHCxxxx`` model family
    (e.g. 3000 Hz for ``simplified_IHC3000``) and sets the phase-locking
    fidelity limit.

    Returns:
        FIR taps, shape (n_taps,).
    """
    n_taps = int(sr * fir_dur)
    if n_taps % 2 == 0:
        n_taps = n_taps + 1
    impulse = np.zeros(n_taps)
    impulse[0] = 1
    fir = np.zeros(n_taps)
    ihc = np.zeros(order + 1)
    ihcl = np.zeros(order + 1)
    c1LP = (sr - 2 * np.pi * cutoff) / (sr + 2 * np.pi * cutoff)
    c2LP = (np.pi * cutoff) / (sr + 2 * np.pi * cutoff)
    for n in range(n_taps):
        ihc[0] = impulse[n]
        for i in range(order):
            ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
        ihcl = ihc
        fir[n] = ihc[order]
    fir = fir * scipy.signal.windows.hann(n_taps)
    fir = fir / fir.sum()
    return fir


# %% Peripheral model stages (nn.Module, all non-learnable buffers)


class FIRFilterbank(nn.Module):
    """Causal FIR filterbank implemented as a fixed-weight ``conv1d``."""

    def __init__(self, fir, dtype: torch.dtype = torch.float32, **kwargs_conv1d):
        """Initializes the filterbank.

        Args:
            fir: Filter coefficients, shape (n_taps,) or (n_filters, n_taps).
            dtype: Dtype to cast `fir` to if it isn't already a tensor.
            kwargs_conv1d: Extra kwargs forwarded to `torch.nn.functional.conv1d`
                (must not include `groups`, which this class uses for batching).
        """
        super().__init__()
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(
                "fir must be list, np.ndarray or torch.Tensor, got "
                f"{fir.__class__.__name__}"
            )
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(
                "fir must be one- or two-dimensional with shape (n_taps,) or "
                f"(n_filters, n_taps), got shape {fir.shape}"
            )
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x: torch.Tensor, batching: bool = False) -> torch.Tensor:
        """Filters the input signal.

        Args:
            x: Input signal.
            batching: If True, `x` has shape (..., n_filters, time) and each
                channel is filtered with its own filter (`self.fir` must be
                2D). If False, the same filter(s) are applied to all of `x`.

        Returns:
            Filtered signal.
        """
        y = x
        if batching:
            assert y.shape[-2] == self.fir.shape[0]
        else:
            y = y.unsqueeze(-2)
        unflatten_shape = y.shape[:-2]
        y = torch.flatten(y, start_dim=0, end_dim=-2 - 1)
        y = nn.functional.conv1d(
            input=nn.functional.pad(y, (self.fir.shape[-1] - 1, 0)),
            weight=self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            **self.kwargs_conv1d,
            groups=y.shape[-2] if batching else 1,
        )
        y = torch.unflatten(y, 0, unflatten_shape)
        if self.fir.ndim == 1:
            y = y.squeeze(-2)
        return y


class GammatoneFilterbank(nn.Module):
    """Cochlear filterbank: fixed-weight FIR approximation of a gammatone bank."""

    def __init__(
        self,
        sr: float = 20e3,
        fir_dur: float = 0.05,
        cfs: np.ndarray = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """Initializes the filterbank.

        Args:
            sr: Sample rate in Hz.
            fir_dur: FIR filter duration in seconds.
            cfs: Center frequencies in Hz, shape (n_cf,). Defaults to
                `erbspace(80, 8000, 50)` if not given.
            dtype: Dtype for the filter weights.
            kwargs: Extra kwargs forwarded to `get_gammatone_impulse_responses`.
        """
        super().__init__()
        if cfs is None:
            cfs = erbspace(8e1, 8e3, 50)
        fir = get_gammatone_impulse_responses(sr=sr, fir_dur=fir_dur, cfs=cfs, **kwargs)
        self.fb = FIRFilterbank(fir, dtype=dtype)

    def forward(self, x: torch.Tensor, batching: bool = False) -> torch.Tensor:
        """Filters `x` through the gammatone filterbank. See `FIRFilterbank.forward`."""
        return self.fb(x, batching=batching)


class IHCTransduction(nn.Module):
    """Inner-hair-cell transduction: optional compression + half-wave rectification."""

    def __init__(
        self,
        compression_power: float = None,
        compression_dbspl_min: float = None,
        compression_dbspl_max: float = None,
        rectify: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the transduction stage.

        Args:
            compression_power: If set, applies broken-stick power compression
                between `compression_dbspl_min` and `compression_dbspl_max`.
            compression_dbspl_min: Lower dB SPL bound of the compression region.
            compression_dbspl_max: Upper dB SPL bound of the compression region.
            rectify: If True, applies half-wave rectification.
            dtype: Dtype for buffers.
        """
        super().__init__()
        if compression_power is not None:
            self.register_buffer(
                "compression_power",
                torch.tensor(compression_power, dtype=dtype),
            )
        else:
            self.compression_power = None
        if compression_dbspl_min is not None:
            self.compression_pa_min = torch.tensor(
                20e-6 * np.power(10, compression_dbspl_min / 20),
                dtype=dtype,
            )
        else:
            self.compression_pa_min = torch.tensor(-np.inf, dtype=dtype)
        if compression_dbspl_max is not None:
            self.compression_pa_max = torch.tensor(
                20e-6 * np.power(10, compression_dbspl_max / 20),
                dtype=dtype,
            )
        else:
            self.compression_pa_max = torch.tensor(np.inf, dtype=dtype)
        self.rectify = rectify

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies compression (if configured) and half-wave rectification."""
        if self.compression_power is not None:
            if self.compression_power.ndim > 0:
                if not self.compression_power.ndim == x.ndim:
                    shape = [1 for _ in range(x.ndim)]
                    shape[-2] = x.shape[-2]
                    self.compression_power = self.compression_power.view(*shape)
            abs_x = torch.abs(x)
            idx_compression = torch.logical_and(
                abs_x >= self.compression_pa_min,
                abs_x < self.compression_pa_max,
            )
            idx_amplification = abs_x < self.compression_pa_min
            x = torch.sign(x) * torch.where(
                idx_compression,
                abs_x**self.compression_power,
                torch.where(
                    idx_amplification,
                    abs_x * (self.compression_pa_min ** (self.compression_power - 1)),
                    abs_x,
                ),
            )
        if self.rectify:
            x = nn.functional.relu(x, inplace=False)
        return x


class IHCLowpassFilter(FIRFilterbank):
    """IHC lowpass filter: sets the phase-locking limit and downsamples."""

    def __init__(
        self,
        sr_input: float = 20e3,
        sr_output: float = 10e3,
        fir_dur: float = 0.05,
        cutoff: float = 3e3,
        order: int = 7,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the lowpass filter.

        Args:
            sr_input: Sample rate of the input signal in Hz.
            sr_output: Sample rate after downsampling in Hz. Must evenly
                divide `sr_input`.
            fir_dur: FIR filter duration in seconds.
            cutoff: Lowpass cutoff frequency in Hz.
            order: Filter order (see `ihc_lowpass_filter_fir`).
            dtype: Dtype for the filter weights.
        """
        fir = ihc_lowpass_filter_fir(
            sr=sr_input, fir_dur=fir_dur, cutoff=cutoff, order=order
        )
        stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(stride, sr_input / sr_output), msg
        super().__init__(fir, dtype=dtype, stride=stride)


class Hilbert(nn.Module):
    """Analytic-signal Hilbert transform (torch analogue of `scipy.signal.hilbert`)."""

    def __init__(self, dim: int = -1):
        """Initializes the transform.

        Args:
            dim: Dimension along which to compute the Hilbert transform.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the analytic signal of `x` along `self.dim`."""
        n = x.shape[self.dim]
        X = torch.fft.fft(x, n=n, dim=self.dim, norm=None)
        h = torch.zeros(n, dtype=X.dtype).to(X.device)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        ind = [np.newaxis] * x.ndim
        ind[self.dim] = slice(None)
        return torch.fft.ifft(X * h[tuple(ind)], n=n, dim=self.dim, norm=None)


class HilbertEnvelope(nn.Module):
    """Hilbert-transform envelope extraction."""

    def __init__(self, **args):
        """Initializes the envelope extractor (kwargs forwarded to `Hilbert`)."""
        super().__init__()
        self.hilbert = Hilbert(**args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the magnitude of the analytic signal of `x`."""
        return torch.abs(self.hilbert(x))


class SigmoidRateLevelFunction(nn.Module):
    """Sigmoid auditory-nerve rate-level function (one or more spont-rate classes)."""

    def __init__(
        self,
        rate_spont: list = None,
        rate_max: list = None,
        threshold: list = None,
        dynamic_range: list = None,
        dynamic_range_interval: float = 0.95,
        compression_power: float = None,
        compression_power_default: float = 0.3,
        envelope_mode: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the rate-level function.

        Args:
            rate_spont: Spontaneous firing rate per ANF spont-rate class (sp/s).
            rate_max: Maximum firing rate per ANF spont-rate class (sp/s).
            threshold: Threshold (dB SPL) per ANF spont-rate class.
            dynamic_range: Dynamic range (dB) per ANF spont-rate class.
            dynamic_range_interval: Fraction of the sigmoid's range spanned
                by `dynamic_range` (e.g. 0.95 -> [2.5%, 97.5%] of max rate).
            compression_power: If set, applies power compression before the
                sigmoid (with `threshold`/`dynamic_range` correction).
            compression_power_default: Reference compression power used to
                rescale `threshold`/`dynamic_range` when `compression_power`
                is set.
            envelope_mode: If True, the sigmoid is applied to the Hilbert
                envelope and the result is recombined with the fine
                structure; if False, it's applied directly to the subbands.
            dtype: Dtype for buffers.
        """
        super().__init__()
        if rate_spont is None:
            rate_spont = [0.0, 0.0, 0.0]
        if rate_max is None:
            rate_max = [250.0, 250.0, 250.0]
        if threshold is None:
            threshold = [0.0, 12.0, 28.0]
        if dynamic_range is None:
            dynamic_range = [20.0, 40.0, 80.0]
        if compression_power is not None:
            self.register_buffer(
                "compression_power",
                torch.tensor(compression_power, dtype=dtype),
            )
            if compression_power_default is not None:
                shift = 20 * np.log10(20e-6 ** (compression_power_default - 1))
                threshold = np.array(threshold) * compression_power_default + shift
                dynamic_range = np.array(dynamic_range) * compression_power_default
        else:
            self.compression_power = None
        assert np.all(
            np.array(rate_max) > np.array(rate_spont)
        ), "rate_max must be greater than rate_spont"
        argument_lengths = [
            len(rate_spont),
            len(rate_max),
            len(threshold),
            len(dynamic_range),
        ]
        channel_specific_size = [1, max(argument_lengths), 1, 1]
        rate_spont = self._resize(rate_spont, channel_specific_size)
        rate_max = self._resize(rate_max, channel_specific_size)
        threshold = self._resize(threshold, channel_specific_size)
        dynamic_range = self._resize(dynamic_range, channel_specific_size)
        y_threshold = (1 - dynamic_range_interval) / 2
        k = np.log((1 / y_threshold) - 1) / (dynamic_range / 2)
        x0 = threshold - (np.log((1 / y_threshold) - 1) / (-k))
        self.register_buffer("rate_spont", torch.tensor(rate_spont, dtype=dtype))
        self.register_buffer("rate_max", torch.tensor(rate_max, dtype=dtype))
        self.register_buffer("threshold", torch.tensor(threshold, dtype=dtype))
        self.register_buffer("dynamic_range", torch.tensor(dynamic_range, dtype=dtype))
        self.register_buffer(
            "dynamic_range_interval", torch.tensor(dynamic_range_interval, dtype=dtype)
        )
        self.register_buffer("y_threshold", torch.tensor(y_threshold, dtype=dtype))
        self.register_buffer("k", torch.tensor(k, dtype=dtype))
        self.register_buffer("x0", torch.tensor(x0, dtype=dtype))
        self.envelope_mode = envelope_mode
        if self.envelope_mode:
            self.envelope_function = HilbertEnvelope(dim=-1)

    @staticmethod
    def _resize(x: list, shape: list) -> np.ndarray:
        """Broadcasts a per-channel parameter list to `shape`."""
        x = np.array(x).reshape([-1])
        if len(x) == 1:
            x = np.full(shape, x[0])
        else:
            x = np.reshape(x, shape)
        return x

    def forward(self, tensor_subbands: torch.Tensor) -> torch.Tensor:
        """Maps subband sound pressure to per-spont-class firing rates.

        Args:
            tensor_subbands: Subband signal, shape (..., n_cf, time); gains
                a leading spont-rate-class axis of length `len(rate_spont)`.

        Returns:
            Firing rates in spikes/s, shape (batch, n_spont, n_cf, time).
        """
        while tensor_subbands.ndim < 4:
            tensor_subbands = tensor_subbands.unsqueeze(-3)
        if self.envelope_mode:
            tensor_env = self.envelope_function(tensor_subbands)
            tensor_tfs = torch.divide(tensor_subbands, tensor_env)
            tensor_tfs = torch.where(
                torch.isfinite(tensor_tfs), tensor_tfs, tensor_subbands
            )
            tensor_pa = tensor_env
        else:
            tensor_pa = tensor_subbands
        if self.compression_power is not None:
            tensor_pa = tensor_pa ** self.compression_power.view(1, 1, -1, 1)
        x = 20.0 * torch.log(tensor_pa / 20e-6) / np.log(10)
        y = 1.0 / (1.0 + torch.exp(-self.k * (x - self.x0)))
        if self.envelope_mode:
            y = y * tensor_tfs
        tensor_rates = self.rate_spont + (self.rate_max - self.rate_spont) * y
        return tensor_rates


class BinomialSpikeGenerator(nn.Module):
    """Stochastic spike-count sampler (normal approximation to the binomial)."""

    def __init__(
        self,
        sr: float = 10000,
        mode: str = "approx",
        n_per_channel: list = None,
        n_per_step: int = 48,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the spike generator.

        Args:
            sr: Sample rate of `tensor_rates` in Hz (converts rates to
                per-sample spike probabilities).
            mode: `"approx"` (normal approximation, differentiable via
                `rsample`), `"exact"` (Bernoulli-sum binomial sampling), or
                `"additive"` (additive Gaussian noise, for backprop).
            n_per_channel: Number of auditory-nerve fibers per spont-rate
                class.
            n_per_step: Batch size used internally by `mode="exact"`.
            dtype: Dtype for buffers.
        """
        super().__init__()
        if n_per_channel is None:
            n_per_channel = [384, 160, 96]
        self.sr = sr
        self.mode = mode
        self.n_per_step = n_per_step
        self.register_buffer(
            "n_per_channel", torch.tensor(n_per_channel, dtype=dtype).view([-1])
        )

    def forward(self, tensor_rates: torch.Tensor) -> torch.Tensor:
        """Samples spike counts from firing rates.

        Args:
            tensor_rates: Firing rates in spikes/s, shape
                (batch, channel, freq, time).

        Returns:
            Spike counts, same shape as `tensor_rates`.
        """
        assert (
            tensor_rates.ndim == 4
        ), "Requires input shape [batch, channel, freq, time]"
        tensor_probs = tensor_rates / self.sr
        if self.mode == "approx":
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            sample = torch.distributions.normal.Normal(
                loc=n * p,
                scale=torch.sqrt(n * p * (1 - p)),
                validate_args=False,
            ).rsample()
            tensor_spike_counts = torch.round(nn.functional.relu(sample))
        elif self.mode == "exact":
            n = self.n_per_channel
            p = tensor_probs
            assert (n.ndim == 1) and (n.shape[0] == p.shape[1])
            tensor_spike_counts = torch.zeros_like(p)
            for channel in range(p.shape[1]):
                total = int(n[channel])
                count = 0
                while count < total:
                    n_sample_per_step = min(self.n_per_step, total - count)
                    sample = (
                        torch.rand(
                            size=(n_sample_per_step, *p[:, channel, :, :].shape),
                            device=self.n_per_channel.device,
                        )
                        < p[None, :, channel, :, :]
                    )
                    tensor_spike_counts[:, channel, :, :] += sample.sum(dim=0)
                    count += n_sample_per_step
        elif self.mode == "additive":
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            noise = torch.randn_like(p) / n
            tensor_spike_counts = nn.functional.relu((p + noise) * n)
        else:
            raise NotImplementedError(f"mode=`{self.mode}` is not implemented")
        return tensor_spike_counts


class RandomSlice(nn.Module):
    """Random crop of the nervegram, kept stochastic even at eval time.

    Matches the original Saddler et al. model, which does not make this
    deterministic for evaluation - see `torchvision.transforms.RandomCrop`.
    """

    def __init__(self, size: list = None, buffer: list = None, **kwargs):
        """Initializes the random slice.

        Args:
            size: Crop size `[n_freq, n_time]`.
            buffer: Fixed trim `[buffer_freq, buffer_time]` applied before
                the random crop (e.g. to discard filter edge artifacts).
            kwargs: Extra kwargs forwarded to `torchvision.transforms.RandomCrop`.
        """
        super().__init__()
        if size is None:
            size = [50, 20000]
        if buffer is None:
            buffer = [0, 0]
        self.size = size
        self.pre_crop_slice = []
        for b in buffer:
            if b is None:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, int) and b > 0:
                self.pre_crop_slice.append(slice(b, -b))
            elif isinstance(b, int) and b == 0:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, (tuple, list)):
                self.pre_crop_slice.append(slice(*b))
        self.crop = torchvision.transforms.RandomCrop(size=self.size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trims by `self.buffer` then applies a random crop of `self.size`."""
        return self.crop(x[..., *self.pre_crop_slice])


class PeripheralModel(nn.Module):
    """Full peripheral pipeline: cochlea -> IHC -> auditory-nerve rate/spikes.

    Processes binaural input `(batch, time, ear=2)` by running each ear
    independently through the same (weight-tied, though there are no
    learnable weights) stages, then concatenating along the channel axis -
    giving the downstream CNN `2 * n_spont_classes` channels to learn
    ITD/ILD-sensitive filters from directly, rather than hand-computed
    cross-correlation cues.
    """

    def __init__(
        self,
        sr_input: float = None,
        sr_output: float = None,
        config_cochlear_filterbank: dict = None,
        config_ihc_transduction: dict = None,
        config_ihc_lowpass_filter: dict = None,
        config_anf_rate_level: dict = None,
        config_anf_spike_generator: dict = None,
        config_random_slice: dict = None,
    ):
        """Constructs the peripheral model from config dictionaries.

        Args:
            sr_input: Input audio sample rate in Hz.
            sr_output: Sample rate after the IHC lowpass/downsample stage.
                Defaults to `sr_input` (no downsampling) if not given.
            config_cochlear_filterbank: Kwargs for `GammatoneFilterbank`
                (via `erbspace`), or empty/None to skip (identity).
            config_ihc_transduction: Kwargs for `IHCTransduction`.
            config_ihc_lowpass_filter: Kwargs for `IHCLowpassFilter`.
            config_anf_rate_level: Kwargs for `SigmoidRateLevelFunction`.
            config_anf_spike_generator: Kwargs for `BinomialSpikeGenerator`.
            config_random_slice: Kwargs for `RandomSlice`, or empty/None to
                skip (identity).
        """
        super().__init__()
        config_cochlear_filterbank = config_cochlear_filterbank or {}
        config_ihc_transduction = config_ihc_transduction or {}
        config_ihc_lowpass_filter = config_ihc_lowpass_filter or {}
        config_anf_rate_level = config_anf_rate_level or {}
        config_anf_spike_generator = config_anf_spike_generator or {}
        config_random_slice = config_random_slice or {}

        self.sr_input = sr_input
        self.sr_output = sr_input if sr_output is None else sr_output
        body = collections.OrderedDict()
        if config_cochlear_filterbank:
            msg = "Cochlear filterbank mode must be `fir_gammatone`"
            assert "fir_gammatone" in config_cochlear_filterbank["mode"], msg
            if config_cochlear_filterbank.get("cfs", False):
                cfs = np.array(config_cochlear_filterbank["cfs"])
            else:
                cfs = erbspace(
                    config_cochlear_filterbank["min_cf"],
                    config_cochlear_filterbank["max_cf"],
                    config_cochlear_filterbank["num_cf"],
                )
            body["cochlear_filterbank"] = GammatoneFilterbank(
                sr=sr_input,
                fir_dur=config_cochlear_filterbank.get("fir_dur", 0.05),
                cfs=cfs,
                **config_cochlear_filterbank.get("kwargs_filter_coefs", {}),
            )
        else:
            body["cochlear_filterbank"] = nn.Identity()
        if config_ihc_transduction:
            body["ihc_transduction"] = IHCTransduction(**config_ihc_transduction)
        if config_ihc_lowpass_filter:
            body["ihc_lowpass_filter"] = IHCLowpassFilter(
                sr_input=self.sr_input,
                sr_output=self.sr_output,
                **config_ihc_lowpass_filter,
            )
        if config_anf_rate_level:
            body["anf_rate_level"] = SigmoidRateLevelFunction(**config_anf_rate_level)
        if config_anf_spike_generator:
            body["anf_spike_generator"] = BinomialSpikeGenerator(
                **config_anf_spike_generator
            )
        self.body = nn.Sequential(body)
        if config_random_slice:
            self.head = RandomSlice(**config_random_slice)
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the peripheral pipeline.

        Args:
            x: Audio, shape (batch, time) monaural or (batch, time, 2) binaural.

        Returns:
            Nervegram, shape (batch, 2 * n_spont, n_cf, time) for binaural
            input (concatenated along the channel axis), or
            (batch, n_spont, n_cf, time) for monaural input.
        """
        if x.shape[-1] == 2:
            assert x.ndim in [3, 5], "expected binaural audio or nervegram input"
            y0 = self.body(x[..., 0])
            y1 = self.body(x[..., 1])
            if y0.ndim == 4:
                y = torch.concat([y0, y1], axis=1)
            else:
                y = torch.stack([y0, y1], axis=-1)
        else:
            y = self.body(x)
        y = self.head(y)
        return y
