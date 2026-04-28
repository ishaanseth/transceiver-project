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


def estimate_baseband_bandwidth(symbol_rate, pulse_method="SINC", rolloff=0.0, margin=1.0):
    """Estimate the baseband bandwidth required for pulse-shaped symbols.

    For a rectangular/square pulse, the first null occurs at the symbol rate.
    For a sinc pulse, the ideal baseband bandwidth is approximately half the
    symbol rate. This helper provides an approximate lowpass cutoff that
    preserves the signal's occupied bandwidth.
    """
    pulse_method = pulse_method.upper()
    if pulse_method == "SINC":
        baseband_bw = 0.5 * symbol_rate
    elif pulse_method == "SQUARE":
        baseband_bw = symbol_rate
    else:
        baseband_bw = symbol_rate * (1 + rolloff)

    return baseband_bw * margin


def downconvert_to_baseband(rx_audio, fs, carrier_freq, cutoff_hz, filter_order=5):
    """Downconvert passband audio into filtered complex baseband samples.

    The cutoff_hz should be chosen to match the expected occupied bandwidth of
    the transmitted waveform after pulse shaping, not merely the symbol rate.
    """
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

def downsample_from_offset(rx_baseband, start, samples_per_symbol):
    """Downsample entire signal using detected timing offset."""
    
    L = samples_per_symbol
    
    # Downsample
    downsampled = rx_baseband[start::L]
    
    # Sparse version (like your ds_xt)
    ds_signal = np.zeros_like(rx_baseband, dtype=complex)
    ds_signal[start::L] = rx_baseband[start::L]
    
    return downsampled, ds_signal