import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, filtfilt


def load_mono_normalized_audio(filepath):
    """Load a WAV file, convert stereo to mono, and normalize peak amplitude."""
    fs, audio = wavfile.read(filepath)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(float)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return fs, audio


def mix_down_to_iq(rx_audio, fs, carrier_freq):
    """Mix a real passband signal down to I and Q baseband arms."""
    t = np.arange(len(rx_audio)) / fs
    i_mixed = rx_audio * 2 * np.cos(2 * np.pi * carrier_freq * t)
    q_mixed = rx_audio * -2 * np.sin(2 * np.pi * carrier_freq * t)
    return i_mixed, q_mixed


def lowpass_filter(signal, fs, cutoff_hz, order=5):
    """Apply a zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def downconvert_to_baseband(rx_audio, fs, carrier_freq, cutoff_hz, filter_order=5):
    """Downconvert passband audio into filtered complex baseband samples."""
    i_mixed, q_mixed = mix_down_to_iq(rx_audio, fs, carrier_freq)
    i_baseband = lowpass_filter(i_mixed, fs, cutoff_hz, order=filter_order)
    q_baseband = lowpass_filter(q_mixed, fs, cutoff_hz, order=filter_order)
    return i_baseband + 1j * q_baseband


def matched_filter_sync(rx_baseband, reference_sequence):
    """Find a reference sequence using a matched filter."""
    matched_filter = reference_sequence[::-1].conj()
    corr = np.abs(fftconvolve(rx_baseband, matched_filter, mode="valid"))
    sync_start_idx = int(np.argmax(corr))
    return sync_start_idx, corr


def locate_pilot_start(sync_start_idx, sync_length, fs, gap_seconds=0.05):
    """Calculate where pilot symbols begin after the sync preamble and gap."""
    gap_samples = int(fs * gap_seconds)
    return sync_start_idx + sync_length + gap_samples


def slice_pilot_samples(rx_baseband, pilot_start_idx, num_pilot_symbols, samples_per_symbol):
    """Slice received baseband samples covering the known pilot symbols."""
    num_samples = (num_pilot_symbols +4)* samples_per_symbol
    return rx_baseband[pilot_start_idx:pilot_start_idx + num_samples]   