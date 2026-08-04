import os

import librosa
import numpy as np
# import sounddevice as sd
# import soundfile as sf
from scipy.signal import chirp

from utils.plotter import plot_wave


# %% Read audio
def read_audio(data_fpath):
    y, sr = librosa.load(data_fpath, sr=None, mono=True)
    print(f'-------- Audio Information --------')
    print(f'audio filename : {os.path.basename(data_fpath)}')
    print(f'audio length : {len(y)/sr} seconds')
    print(f'audio sampling rate : {sr} Hz')
    print(f'audio shape : {len(y)}\n')
    return y, sr

def read_stereo_audio(data_fpath):
    y, sr = librosa.load(data_fpath, sr=None, mono=False)
    print(f'-------- Audio Information --------')
    print(f'audio filename : {os.path.basename(data_fpath)}')
    print(f'audio length : {len(y[0])/sr} seconds')
    print(f'audio sampling rate : {sr} Hz')
    print(f'audio shape : {y[0].shape} / {y[1].shape} \n')
    return y[0], y[1], sr


# %% Listen audio
def listen_audio(y, sr):
    print(f'-------- Playing Audio --------')
    sd.play(y, samplerate=sr)
    sd.wait()
    print(f'Playing Finished\n')


# %% Generate audio
def generate_gauss_noise(duration, sr, rms, plot=True):
    noise_gauss = rms * np.random.normal(size=int(duration*sr))
    if plot:
        plot_wave(noise_gauss, sr, name='Gaussian Noise', title='Gaussian Noise', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)
    return noise_gauss

def generate_sweep(duration, sr, f0=20, f1=20000, method='logarithmic', plot=True):
    t = np.linspace(0, duration, int(duration*sr))
    sweep_signal = chirp(t, f0, duration, f1, method=method)
    if plot:
        plot_wave(sweep_signal, sr, name='Sweep', title=f'Sweep ({f0}-{f1} Hz, {method})', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)
    return sweep_signal


# %% Save audio
def save_audio(y, sr, save_path):
    sf.write(save_path, y, sr)
    print(f'--------Saved to {save_path} ---------\n')

def save_stereo_audio(signal_l, signal_r, sr, save_path):
    stereo = np.column_stack([signal_l, signal_r])
    sf.write(save_path, stereo, sr)
    print(f'-------- Saved to {save_path} ---------\n')


# %% Main function
if __name__ == '__main__':
    duration = 10
    sr = 48000

    # sweep = generate_sweep(duration, sr)
    # listen_audio(sweep, sr)

    rms = 0.1
    noise_gauss = generate_gauss_noise(duration, sr, rms)
    listen_audio(noise_gauss, sr)
