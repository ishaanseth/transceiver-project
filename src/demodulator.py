import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, filtfilt, firwin


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
    # Use FIR filter with 101 taps for good frequency response
    numtaps = 101
    b = firwin(numtaps, normal_cutoff)
    return filtfilt(b, 1, signal)


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





def matched_filter_sync_new(rx_baseband, reference_sequence, pulse_shape, sps):
    """
    Matched filter sync that internally handles pulse shape strings.
    
    pulse_shape: "SQUARE", "SINC", or "RRC"
    sps: samples per symbol
    """

    # Ensure 1D
    rx_baseband = np.asarray(rx_baseband).flatten()
    reference_sequence = np.asarray(reference_sequence).flatten()

    # -----------------------------
    # Generate pulse shape filter
    # -----------------------------
    if pulse_shape == "SQUARE":
        h = np.ones(sps)

    elif pulse_shape == "SINC":
        SPAN = 4
        t = np.arange(-SPAN*sps, SPAN*sps + 1) / sps
        h = np.sinc(t)

    elif pulse_shape == "RRC":
        beta = 0.25
        SPAN = 6
        t = np.arange(-SPAN*sps, SPAN*sps + 1) / sps
        
        h = np.zeros_like(t)

        for i in range(len(t)):
            ti = t[i]

            if ti == 0.0:
                h[i] = 1.0 - beta + (4 * beta / np.pi)

            elif abs(ti) == 1 / (4 * beta):
                h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2/np.pi) * np.cos(np.pi / (4 * beta))
                )

            else:
                numerator = (
                    np.sin(np.pi * ti * (1 - beta)) +
                    4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                )
                denominator = (
                    np.pi * ti * (1 - (4 * beta * ti)**2)
                )
                h[i] = numerator / denominator

        # Normalize
        h = h / np.sqrt(np.sum(h**2))

    else:
        raise ValueError("Invalid pulse shape")

    # -----------------------------
    # Apply pulse shaping to reference
    # -----------------------------
    ref_shaped = fftconvolve(reference_sequence, h, mode="full")

    # -----------------------------
    # Matched filter
    # -----------------------------
    matched_filter = ref_shaped[::-1].conj()

    corr = np.abs(fftconvolve(rx_baseband, matched_filter, mode="valid"))

    sync_start_idx = int(np.argmax(corr))

    return sync_start_idx, corr







def locate_pilot_start(sync_start_idx, sync_length, fs, gap_seconds=0.05):
    """Calculate where pilot symbols begin after the sync preamble and gap."""
    gap_samples = int(fs * gap_seconds)
    return sync_start_idx + sync_length + gap_samples


def slice_pilot_samples(rx_baseband, pilot_start_idx, num_pilot_symbols, samples_per_symbol, span_symbols=4):
    """Slice received baseband samples covering the known pilot symbols."""
    num_samples = (num_pilot_symbols + span_symbols) * samples_per_symbol
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

def symbols_to_bits(symbols, M=4, METHOD="QAM"):
    
    if METHOD == "QAM":
        symbols = np.asarray(symbols)
        
        k = int(np.log2(M))
        k_half = k // 2
        L = int(np.sqrt(M))
        
        # Same normalization used in TX
        norm_factor = np.sqrt((2 / 3) * (M - 1))
        
        # Denormalize
        symbols = symbols * norm_factor
        
        bits_out = []
        
        for s in symbols:
            i_val = np.real(s)
            q_val = np.imag(s)
            
            # Find nearest level index
            i_idx = int(np.round((i_val + (L - 1)) / 2))
            q_idx = int(np.round((q_val + (L - 1)) / 2))
            
            # Clip (important for noise robustness)
            i_idx = np.clip(i_idx, 0, L - 1)
            q_idx = np.clip(q_idx, 0, L - 1)
            
            # Convert to bits
            i_bits = list(map(int, format(i_idx, f'0{k_half}b')))
            q_bits = list(map(int, format(q_idx, f'0{k_half}b')))
            
            bits_out.extend(i_bits + q_bits)
        
        return np.array(bits_out)
        
    elif METHOD == "BPSK":
        symbols = np.asarray(symbols)
        # BPSK: > 0 is 1, <= 0 is 0
        bits_out = (np.real(symbols) > 0).astype(int)
        return bits_out

