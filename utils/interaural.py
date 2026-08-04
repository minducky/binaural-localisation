import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import correlate

from utils.analysis import *
from utils.read_listen_save import *

# %% Interaural Features
def cal_ild(left_sig, right_sig, sr, n_fft=2048, win_length=2048, hop_length=512, window='hann'):
    """
    Calculate Interaural Level Difference (ILD) spectrogram
    ILD = 20 * log10(|L| / |R|) in dB
    """
    t, f, L = cal_stft(left_sig, sr, db=False, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    _, _, R = cal_stft(right_sig, sr, db=False, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)

    # Magnitude
    L_mag = np.abs(L)
    R_mag = np.abs(R)

    # ILD in dB (with small epsilon to avoid log(0))
    eps = 1e-10
    ild = 20 * np.log10((L_mag + eps) / (R_mag + eps))

    return f, t, ild

def cal_ipd(left_sig, right_sig, sr, n_fft=2048, win_length=2048, hop_length=512, window='hann'):
    """
    Calculate Interaural Phase Difference (IPD) spectrogram
    IPD = angle(L) - angle(R) in radians
    """
    t, f, L = cal_stft(left_sig, sr, db=False, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    _, _, R = cal_stft(right_sig, sr, db=False, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)

    # Phase difference
    ipd = np.angle(L) - np.angle(R)

    # Wrap to [-π, π]
    ipd = np.angle(np.exp(1j * ipd))

    return f, t, ipd

def cal_itd(left_sig, right_sig, sr, n_fft=2048, win_length=2048, hop_length=512, window='hann'):
    """
    Calculate Interaural Time Difference (ITD) spectrogram
    ITD derived from IPD: ITD = IPD / (2π * f)
    """
    f, t, ipd = cal_ipd(left_sig, right_sig, sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)

    itd = np.zeros_like(ipd)
    itd[1:, :] = ipd[1:, :] / (2 * np.pi * f[1:, np.newaxis])

    return f, t, itd

def cal_iacc(left_sig, right_sig, sr, win_length=2048, hop_length=512):
    """
    Calculate Interaural Cross-Correlation Coefficient (IACC)
    IACC = max(normalized cross-correlation) over time windows
    """

    n_frames = (len(left_sig) - win_length) // hop_length + 1
    iacc = np.zeros(n_frames)
    t = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + win_length

        l_win = left_sig[start:end]
        r_win = right_sig[start:end]

        # Normalized cross-correlation
        corr = correlate(l_win, r_win, mode='full', method='fft')

        # Normalize
        norm = np.sqrt(np.sum(l_win**2) * np.sum(r_win**2))
        if norm > 0:
            corr = corr / norm
            iacc[i] = np.max(np.abs(corr))
        else:
            iacc[i] = 0

        t[i] = (start + win_length // 2) / sr

    return t, iacc


# %% Plot interaural features
def plot_interaural_features(left_sig, right_sig, sr, n_fft=2048, win_length=2048, hop_length=512, window='hann',
                              f_max=8000, width=1400, height=1200):
    """
    Plot all interaural features in a single figure
    """
    # Calculate all features
    f_ild, t_ild, ild = cal_ild(left_sig, right_sig, sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    f_ipd, t_ipd, ipd = cal_ipd(left_sig, right_sig, sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    f_itd, t_itd, itd = cal_itd(left_sig, right_sig, sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    t_iacc, iacc = cal_iacc(left_sig, right_sig, sr, win_length=win_length, hop_length=hop_length)

    # Limit frequency range
    f_idx = f_ild <= f_max
    f_ild = f_ild[f_idx]
    f_ipd = f_ipd[f_idx]
    f_itd = f_itd[f_idx]
    ild = ild[f_idx, :]
    ipd = ipd[f_idx, :]
    itd = itd[f_idx, :]

    # Convert ITD to milliseconds
    itd_ms = itd * 1000

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            '<b>ILD (Interaural Level Difference)</b>',
            '<b>IPD (Interaural Phase Difference)</b>',
            '<b>ITD (Interaural Time Difference)</b>',
            '<b>IACC (Interaural Cross-Correlation)</b>'
        ),
        vertical_spacing=0.08
    )

    # ILD spectrogram
    fig.add_trace(
        go.Heatmap(
            x=t_ild, y=f_ild, z=ild,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='dB', x=1.02, len=0.23, y=0.875),
            name='ILD'
        ),
        row=1, col=1
    )

    # IPD spectrogram
    fig.add_trace(
        go.Heatmap(
            x=t_ipd, y=f_ipd, z=ipd,
            colorscale='twilight',
            zmid=0,
            zmin=-np.pi,
            zmax=np.pi,
            colorbar=dict(title='rad', x=1.02, len=0.23, y=0.625),
            name='IPD'
        ),
        row=2, col=1
    )

    # ITD spectrogram
    fig.add_trace(
        go.Heatmap(
            x=t_itd, y=f_itd, z=itd_ms,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='ms', x=1.02, len=0.23, y=0.375),
            name='ITD'
        ),
        row=3, col=1
    )

    # IACC time series
    fig.add_trace(
        go.Scatter(
            x=t_iacc, y=iacc,
            mode='lines',
            line=dict(color='blue', width=2),
            name='IACC',
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ),
        row=4, col=1
    )

    # Update axes
    fig.update_xaxes(title_text='<b>Time (s)</b>', row=4, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text='<b>Frequency (Hz)</b>', row=1, col=1)
    fig.update_yaxes(title_text='<b>Frequency (Hz)</b>', row=2, col=1)
    fig.update_yaxes(title_text='<b>Frequency (Hz)</b>', row=3, col=1)
    fig.update_yaxes(title_text='<b>IACC</b>', range=[0, 1], row=4, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    # Update layout
    fig.update_layout(
        title='<b>Interaural Features Analysis</b>',
        title_x=0.5,
        plot_bgcolor='white',
        width=width,
        height=height,
        showlegend=False
    )

    fig.show()

    return fig


# %% Main function
if __name__ == "__main__":
    left_audio_fpath = 'data/HATS_L.wav'
    right_audio_fpath = 'data/HATS_R.wav'
    audio_l, sr_l = read_audio(left_audio_fpath)
    audio_r, sr_r = read_audio(right_audio_fpath)

    # Trim to reasonable length for visualization
    duration = 5  # seconds
    left = audio_l[:int(duration * sr_l)]
    right = audio_r[:int(duration * sr_r)]

    # Plot all features
    plot_interaural_features(left, right, sr_l)
