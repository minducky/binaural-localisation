import plotly.graph_objects as go

from utils.analysis import *


# %% Plot function (1d, 2d)
def plot_1d(x, y, name, title=None, title_font='Arial', title_fontsize=24, title_fontcolor='black',
                  xaxis_title=None, xaxis_font='Arial', xaxis_fontsize=16, xaxis_fontcolor='black',
                  yaxis_title=None, yaxis_font='Arial', yaxis_fontsize=16, yaxis_fontcolor='black',
                  width=None, height=None, download=False, download_fpath=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name,
            line=dict(color='black', width=1),
        )
    )

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',  # <b> for bold
            font=dict(family=title_font, size=title_fontsize, color=title_fontcolor),
            x=0.5,  # Center the title
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text=f'<b>{xaxis_title}</b>',
                font=dict(family=xaxis_font, size=xaxis_fontsize, color=xaxis_fontcolor),
            ),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(
                text=f'<b>{yaxis_title}</b>',
                font=dict(family=yaxis_font, size=yaxis_fontsize, color=yaxis_fontcolor),
            ),
            showgrid=True,
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='white',
        width=width,
        height=height
    )

    # fig.show()
    if download:
        fig.write_image(download_fpath)
        fig = None


def plot_2d(x, y, z, name, title=None, title_font='Arial', title_fontsize=24, title_fontcolor='black',
                     xaxis_title=None, xaxis_font='Arial', xaxis_fontsize=16, xaxis_fontcolor='black',
                     yaxis_title=None, yaxis_font='Arial', yaxis_type='log', yaxis_fontsize=16, yaxis_fontcolor='black',
                     width=None, height=None, download=False, download_fpath=None):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale='Cividis',
            name=name
        )
    )
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(family=title_font, size=title_fontsize, color=title_fontcolor),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=f'<b>{xaxis_title}</b>', font=dict(family=xaxis_font, size=xaxis_fontsize, color=xaxis_fontcolor)),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text=f'<b>{yaxis_title}</b>', font=dict(family=yaxis_font, size=yaxis_fontsize, color=yaxis_fontcolor)),
            showgrid=True,
            type=yaxis_type,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        width=width,
        height=height
    )

    fig.show()
    if download:
        fig.write_image(download_fpath)
        fig = None


# %% Plot certain data (wave, fft, psd, stft, melspec)
def plot_wave(sig, sr, name, title, xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=400, download=False, download_fpath=None):
    x, y = np.linspace(0, len(sig)/sr, len(sig)), sig

    # Plot 1d Wave
    plot_1d(x, y, name=name, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, width=width, height=height, download=download, download_fpath=download_fpath)

def plot_fft(sig, sr, name, title, xaxis_title, yaxis_title, width, height, download=False, download_fpath=None,
             db=False):
    # Compute FFT using Numpy
    x, y = cal_fft(sig, sr, db)

    # Plot 1d FFT
    plot_1d(x, y, name=name, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, width=width, height=height, download=download, download_fpath=download_fpath)

def plot_psd(sig, sr, name, title, xaxis_title, yaxis_title, width, height, download=False, download_fpath=None,
             db=False, n_fft=None):
    # Compute PSD using Scipy
    x, y = cal_psd(sig, sr, db, n_fft)

    # Plot 1d PSD
    plot_1d(x, y, name=name, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, width=width, height=height, download=download, download_fpath=download_fpath)

def plot_stft(sig, sr, name, title, xaxis_title, yaxis_title, yaxis_type, width, height, download=False, download_fpath=None,
              db=False, n_fft=2048, win_length=2048, hop_length=512, window='hann'):
    # Compute STFT using Librosa
    x, y, z = cal_stft(sig, sr, db=db, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    z = np.abs(z)

    plot_2d(x=x, y=y, z=z, name=name, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, yaxis_type=yaxis_type, width=width, height=height, download=download, download_fpath=download_fpath)

def plot_melspec(sig, sr, name, title, xaxis_title, yaxis_title, yaxis_type, width, height, download=False, download_fpath=None,
                 db=False, n_fft=2048, win_length=2048, hop_length=512, window='hann', power=2.0, n_mels=128, fmin=50, fmax=8000):
    # Compute Mel Spectrogram using Librosa (always power=2.0)
    x, y, z = cal_melspec(sig, sr, db=db, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window, power=power, n_mels=n_mels, fmin=fmin, fmax=fmax)

    plot_2d(x=x, y=y, z=z, name=name, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, yaxis_type=yaxis_type, width=width, height=height, download=download, download_fpath=download_fpath)


# %% Main function
if __name__ == '__main__':
    sr = 16000
    freq = 500
    length = 0.5
    t = np.linspace(0, length, int(sr*length))
    sig = np.sin(2*np.pi*freq*t)

    plot_wave(sig=sig, sr=sr, name='Wave', title=f'sin {freq}Hz {length}sec Wave', xaxis_title='Time (sec)',
              yaxis_title='Amplitude', width=1200, height=500, download=True, download_fpath='test.html')

    plot_fft(sig=sig, sr=sr, name='FFT', title=f'sin {freq}Hz {length}sec FFT', xaxis_title='Frequency (Hz)',
              yaxis_title='Half Amplitude (dB)', width=1200, height=500, download=True, download_fpath='test.html', db=True)

    plot_psd(sig=sig, sr=sr, name='PSD', title=f'sin {freq}Hz {length}sec PSD', xaxis_title='Frequency (Hz)',
              yaxis_title=r'PSD (dB/Hz)', width=1200, height=500, download=True, download_fpath='test.html', db=True, n_fft=512)

    plot_stft(sig=sig, sr=sr, name='STFT', title=f'sin {freq}Hz {length}sec STFT', xaxis_title='Time (sec)', yaxis_title='Frequency (Hz)',
              width=800, height=800, download=False, download_fpath=None, n_fft=1024, win_length=1024, hop_length=256, window='hann')

    plot_melspec(sig=sig, sr=sr, name='Mel', title=f'sin {freq}Hz {length}sec Mel', xaxis_title='Time (sec)', yaxis_title='Frequency (Hz)',
              width=800, height=800, download=False, download_fpath=None, n_fft=1024, win_length=1024, hop_length=256, window='hann')