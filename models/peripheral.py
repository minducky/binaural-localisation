import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as TAT
from scipy.signal import filtfilt, lfilter, butter
from scipy.special import factorial

class Peripheral:
    def __init__(self, outermiddle=True, basilar='', ihc='', adaptation=''):
        self.outermiddle = outermiddle
        self.basilar = basilar
        self.ihc = ihc
        self.adaptation = adaptation


# %% 1. Outer Middle Ear [1000, 4000]Hz Bandpass IIR Filter
def outer_middle_ear(x, sr):
    """
    Outer and middle ear filtering

    Args:
        x: (batch, channel, time)
        sr: sampling rate
    Returns:
        (batch, channel, time)
    """
    batch_size, channel, length = x.shape

    pi_64 = torch.tensor(np.pi, dtype=torch.float64)

    q = 2 - torch.cos(2*pi_64*4000/sr) - torch.sqrt((torch.cos(2*pi_64*4000/sr)-2)**2 - 1)
    r = 2 - torch.cos(2*pi_64*1000/sr) - torch.sqrt((torch.cos(2*pi_64*1000/sr)-2)**2 - 1)

    y = torch.zeros(batch_size, channel, length+2, device=x.device, dtype=x.dtype)
    x_padded = torch.zeros(batch_size, channel, length+2, device=x.device, dtype=x.dtype)
    x_padded[:, :, 2:] = x

    for n in range(2, length+2):
        y[:, :, n] = ((1-q)*r*x_padded[:, :, n] -
                   (1-q)*r*x_padded[:, :, n-1] +
                   (q+r)*y[:, :, n-1] -
                   q*r*y[:, :, n-2])

    return y[:, :, 2:]


# %% 2-1. Gammatone Filtering Lyon
def audspace_bw(fmin, fmax, bw=1.0):
    """ Auditory scale points specified by bandwidth """

    # Convert frequency limits to ERB scale
    audlimits = freq_to_aud(np.array([fmin, fmax]))
    audrange = audlimits[1] - audlimits[0]

    # Calculate number of points (excluding final point)
    n = int(np.floor(audrange / bw))

    # Center the points between fmin and fmax
    remainder = audrange - n * bw
    audpoints = audlimits[0] + np.arange(n + 1) * bw + remainder / 2

    # Add final point
    n = n + 1
    y = aud_to_freq(audpoints)

    return y, n

def freq_to_aud(freq):
    """ Convert frequency (Hz) to ERB scale / ERB scale formula from Moore & Glasberg (1983) """
    return 9.2645 * np.log(1 + freq * 0.00437)

def aud_to_freq(aud):
    """ Convert ERB scale to frequency (Hz) """
    return 1 / 0.00437 * (np.exp(aud / 9.2645) - 1)

def calculate_betamul(n):
    """ Calculate betamul for Gammatone filter """
    betamul = (factorial(n - 1) ** 2) / (
        np.pi * factorial(2 * n - 2) * 2 ** (-(2 * n - 2))
    )
    return betamul

def gammatone_filter_coefficients(fmin, fmax, bw, sr, order=4, filter_coeff_dir=None):
    """
    Generate complex Gammatone filter coefficients

    Parameters:
    -----------
    fc : np.ndarray
        Center frequencies (Hz)
    sr : float
        Sampling frequency (Hz)
    order : int
        Filter order (default: 4)

    Returns:
    --------
    b : np.ndarray
        Numerator coefficients, shape (n_filters,)
    a : np.ndarray
        Denominator coefficients, shape (n_filters, order+1)
    """
    fc, n_filters = audspace_bw(fmin, fmax, bw)

    b_save_fpath = os.path.join(filter_coeff_dir, f'b_BM_fmin_{fmin}_fmax_{fmax}_bw_{bw}_sr_{sr}.pt')
    a_save_fpath = os.path.join(filter_coeff_dir, f'a_BM_fmin_{fmin}_fmax_{fmax}_bw_{bw}_sr_{sr}.pt')

    if os.path.isfile(b_save_fpath) and os.path.isfile(a_save_fpath):
        b = torch.load(b_save_fpath)
        a = torch.load(a_save_fpath)
        return fc, b, a

    # Calculate betamul
    betamul = calculate_betamul(order)

    # ERB bandwidth
    ERB = 24.7 + fc / 9.265
    ourbeta = betamul * ERB

    b = np.zeros(n_filters, dtype=complex)
    a = np.zeros((n_filters, order + 1), dtype=complex)

    for i in range(n_filters):
        # Convert to radians
        theta = 2 * np.pi * fc[i] / sr
        phi = 2 * np.pi * ourbeta[i] / sr

        # Compute pole position
        atilde = np.exp(-phi - 1j * theta)

        # Repeat pole n times and expand polynomial
        poles = atilde * np.ones(order)
        a[i, :] = np.poly(poles)

        # Numerator coefficient
        btmp = 1 - np.exp(-phi)
        b[i] = btmp**order

    b = torch.tensor(b).unsqueeze(-1)
    a = torch.tensor(a)
    torch.save(b, b_save_fpath)
    torch.save(a, a_save_fpath)
    return fc, b, a

def basilar_membrane(x, a, b, sr=48000, resample_freq=4800):
    """
    Basilar membrane filtering with frequency decomposition

    Args:
        x: (batch, channel, time)
        a: filter denominator coefficients (freq_bin, a_order)
        b: filter numerator coefficients (freq_bin, b_order)
        sr: original sampling rate
        resample_freq: target sampling rate after filtering
    Returns:
        (batch, freq_bin, time_resampled)
    """
    batch, channel, time = x.shape
    freq_bin = a.shape[0]

    nfft = 2 ** (time.bit_length())
    freqs = torch.fft.fftfreq(nfft, device=x.device, dtype=torch.float64)
    z_inv = torch.exp(-2j * torch.pi * freqs)

    a_powers = z_inv.unsqueeze(0) ** torch.arange(a.shape[1], device=x.device).unsqueeze(-1)
    H_den = a @ a_powers
    b_powers = z_inv.unsqueeze(0) ** torch.arange(b.shape[1], device=x.device).unsqueeze(-1)
    H_num = b @ b_powers
    H = H_num / H_den

    X = torch.fft.fft(x, nfft)
    Y = X * H.unsqueeze(0)
    output = 2 * torch.fft.ifft(Y, n=nfft, dim=-1)[..., :time].real.float()
    # todo : should we do
    if sr != resample_freq:
        resampler = TAT.Resample(orig_freq=sr, new_freq=resample_freq)
        output = resampler(output)

    return output

# %% 2-2. Gamma Chirp

# %% 2-3. Transmission line

# %% 2-4. Dual-Resonance Non Linear (DRNL)

# %% Inner Hair Cell Stage - Half-wave rectifier
def ihc_coefficients(fcut, sr, order, filter_coeff_dir):
    b_save_fpath = os.path.join(filter_coeff_dir, f'b_IHC_fcut_{fcut}sr_{sr}.pt')
    a_save_fpath = os.path.join(filter_coeff_dir, f'a_IHC_fcut_{fcut}sr_{sr}.pt')

    if os.path.isfile(b_save_fpath) and os.path.isfile(a_save_fpath):
        b = torch.load(b_save_fpath)
        a = torch.load(a_save_fpath)
        return b, a

    b, a = butter(order, 3000, btype='low', fs=sr, output='ba')
    b = torch.tensor(b, dtype=torch.complex128).unsqueeze(0)
    a = torch.tensor(a, dtype=torch.complex128).unsqueeze(0)

    print(f'b.shape : {b.shape}')
    print(f'a.shape : {a.shape}')

    torch.save(b, b_save_fpath)
    torch.save(a, a_save_fpath)
    return b, a

    # Calculate betamul
    betamul = calculate_betamul(order)

def filt_tch(x, a, b):
    batch, channel, time = x.shape
    device = x.device

    nfft = 2 ** (time.bit_length())
    freqs = torch.fft.fftfreq(nfft, device=x.device, dtype=torch.float64).to(device)
    z_inv = torch.exp(-2j * torch.pi * freqs)

    a_powers = z_inv.unsqueeze(0) ** torch.arange(a.shape[1], device=x.device).unsqueeze(-1)
    H_den = a @ a_powers
    b_powers = z_inv.unsqueeze(0) ** torch.arange(b.shape[1], device=x.device).unsqueeze(-1)
    H_num = b @ b_powers
    H = H_num / H_den

    X = torch.fft.fft(x, nfft)
    Y = X * H.unsqueeze(0)
    output = 2 * torch.fft.ifft(Y, n=nfft, dim=-1)[..., :time].real.float()

    return output


# %% Inner Hair Cell Stage - Detailed Implementation (lowpass 2000Hz 5 times) : lowpass 700Hz
def inner_hair_cell(x, sr, filter_coeff_dir, n_iterations=5):
    """
    Inner hair cell processing with rectification and filtering

    Args:
        x: (batch, channel, time)
        a: filter denominator coefficients (channel, a_order)
        b: filter numerator coefficients (channel, b_order)
        n_iterations: number of filtering iterations
    Returns:
        (batch, channel, time)
    """
    b, a = butter(1, 2000, btype='low', fs=sr, output='ba')
    b = torch.tensor(b, dtype=torch.complex128).unsqueeze(0)
    a = torch.tensor(a, dtype=torch.complex128).unsqueeze(0)
    print(b.shape)
    print(a.shape)
    b_save_fpath = os.path.join(filter_coeff_dir, f'b_sr_{sr}.pt')
    a_save_fpath = os.path.join(filter_coeff_dir, f'a_sr_{sr}.pt')
    if not os.path.isfile(b_save_fpath) and not os.path.isfile(a_save_fpath):
        torch.save(b, b_save_fpath)
        torch.save(a, a_save_fpath)

    def _filt(signal, a_coeff, b_coeff):
        batch, channel, time = signal.shape

        nfft = 2 ** (time.bit_length())

        freqs = torch.fft.rfftfreq(nfft, device=signal.device, dtype=torch.float64)
        z_inv = torch.exp(-2j * torch.pi * freqs)
        a_powers = z_inv.unsqueeze(0) ** torch.arange(a_coeff.shape[1], device=signal.device).unsqueeze(-1)
        H_den = a_coeff @ a_powers
        b_powers = z_inv.unsqueeze(0) ** torch.arange(b_coeff.shape[1], device=signal.device).unsqueeze(-1)
        H_num = b_coeff @ b_powers

        H = H_num / H_den

        X = torch.fft.rfft(signal, nfft)
        Y = X * H.unsqueeze(0)
        output = torch.fft.irfft(Y, n=nfft, dim=-1)[..., :time]

        return output

    rectified = torch.clamp(x, min=0)
    filtered = rectified
    for _ in range(n_iterations):
        filtered = _filt(filtered, a, b)

    return filtered


# %% Adaptation
def adaptation(x, sr=4800, limit=1, tau=None, minlvl=1e-5):
    """
    Adaptation mechanism for neural response

    Args:
        x: (batch, n_channels, n_samples)
        sr: sampling rate
        limit: overshoot limitation parameter
        tau: time constants in seconds (default: linspace from 5ms to 500ms, 5 values)
        minlvl: minimum level threshold
    Returns:
        (batch, n_channels, n_samples)
    """
    if tau is None:
        tau = torch.linspace(5, 500, 5, dtype=torch.float64) / 1000

    minlvl = torch.tensor(minlvl, dtype=torch.float64)

    def _adaptation_loop(signal_per_ch):
        batch, n_samples = signal_per_ch.shape
        n_loops = len(tau)

        b0 = 1 / (tau * sr)
        a1 = torch.exp(-b0)
        b0 = 1 - a1

        corr = minlvl ** (1 / 2 ** n_loops)
        mult = 100 / (1 - corr)

        signal = torch.maximum(signal_per_ch, minlvl)

        state_init = minlvl ** (1 / 2 ** torch.arange(1, n_loops + 1))
        state = state_init.unsqueeze(0).expand(batch, -1).clone()

        output = torch.zeros(batch, n_samples, dtype=signal.dtype, device=signal.device)

        if limit <= 1:
            for i in range(n_samples):
                tmp = signal[:, i]
                for j in range(n_loops):
                    tmp = tmp / state[:, j]
                    state[:, j] = a1[j] * state[:, j] + b0[j] * tmp
                output[:, i] = tmp
        else:
            maxvalue = (1 - state**2) * limit - 1
            factor = maxvalue * 2
            expfac = -2 / maxvalue
            offset = maxvalue - 1

            for i in range(n_samples):
                tmp = signal[:, i]
                for j in range(n_loops):
                    tmp = tmp / state[:, j]

                    if tmp > 1:
                        tmp = (
                            factor[j] / (1 + torch.exp(expfac[j] * (tmp - 1))) - offset[j]
                        )

                    state[:, j] = a1[j] * state[:, j] + b0[j] * tmp
                output[:, i] = tmp

        output = (output - corr) * mult
        return output

    batch, n_channels, n_samples = x.shape
    output = torch.zeros_like(x)
    for ch in range(n_channels):
        output[:, ch, :] = _adaptation_loop(x[:, ch, :])

    return output