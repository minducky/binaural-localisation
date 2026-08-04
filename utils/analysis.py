import librosa
import librosa.feature
import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks


# %% Temporal Analysis
def cal_gcc_phat(y1, y2, sr, max_time=None):
    n = y1.shape[0] + y2.shape[0]

    Y1 = np.fft.rfft(y1, n)  # n : odd, shape : (n+1)/2
    Y2 = np.fft.rfft(y2, n)  # n : even, shape : (n/2)+1

    R = Y1 * np.conj(Y2)
    R /= np.abs(R)

    cc = np.fft.ifft(R, n=n)
    max_shift = int(n / 2)
    if max_time:
        max_shift = np.minimum(int(sr * max_time), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    estimated_delay = shift / sr  # (-) : y2 is more delayed, (+) : y1 is more delayed
    return estimated_delay


# %% Temporal Transform
def resample_audio(y, sr_orig, sr_target):
    y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)
    return y_resampled


# %% Frequency Analysis
def cal_fft(sig, sr, db):
    n_fft = len(sig)
    fft = np.fft.fft(sig, n=n_fft)
    p_ref = 20e-6
    magnitude = 20 * np.log10(np.abs(2 * fft[:n_fft // 2]) / n_fft / p_ref) if db else np.abs(fft[:n_fft // 2]) / n_fft
    freqs = np.fft.fftfreq(n_fft, 1 / sr)[:n_fft // 2]
    return freqs, magnitude

def cal_psd(sig, sr, db, n_fft):
    freqs, psd = welch(sig, sr, nperseg=n_fft)
    p_ref = 20e-6
    psd = 10 * np.log10(psd / (p_ref ** 2)) if db else psd
    return freqs, psd


# %% Frequency Transform (Filter)
def highpass(y, sr, f_cut, order=8):
    nyquist = sr / 2
    normalized_cutoff = f_cut / nyquist
    b, a = butter(order, normalized_cutoff, btype='high')
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def lowpass(y, sr, f_cut, order=8):
    nyquist = sr / 2
    normalized_cutoff = f_cut / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def bandpass(y, sr, f_low, f_high, order=8):
    nyquist = sr / 2
    normalized_freqs = [f_low / nyquist, f_high / nyquist]
    b, a = butter(order, normalized_freqs, btype='band')
    y_filtered = filtfilt(b, a, y)
    return y_filtered


# %% Time-Frequency Analysis
def cal_stft(sig, sr, db, n_fft, win_length, hop_length, window):
    stft = librosa.stft(sig, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    t_bin = np.linspace(0, len(sig) / sr, stft_db.shape[1])
    f_bin = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    amplitude = stft_db if db else stft
    return t_bin, f_bin, amplitude

def cal_melspec(sig, sr, db, n_fft, win_length, hop_length, window, power, n_mels, fmin, fmax):
    melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                             window=window,
                                             power=power, n_mels=n_mels, fmin=fmin, fmax=fmax)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    t_bin = np.linspace(0, len(sig) / sr, melspec_db.shape[1])
    f_bin = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    amplitude = melspec_db if db else melspec
    return t_bin, f_bin, amplitude


# %% Peak Detection
def cal_peaks(sig, prominence=50):
    peaks, _ = find_peaks(sig, prominence=prominence)