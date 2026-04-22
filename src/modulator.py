import numpy as np
from scipy.signal.windows import tukey


def map_bits_to_qam(bit_array, M=4):
    """Maps an array of bits to square M-QAM complex symbols."""
    bit_array = np.asarray(bit_array, dtype=int)
    k = int(np.log2(M))
    if 2**k != M or k % 2 != 0:
        raise ValueError("M must be a perfect square (4, 16, 64, 256...)")

    pad_len = (k - len(bit_array) % k) % k
    if pad_len > 0:
        bit_array = np.append(bit_array, np.zeros(pad_len, dtype=int))

    symbols = []
    k_half = k // 2
    norm_factor = np.sqrt((2 / 3) * (M - 1))

    for i in range(0, len(bit_array), k):
        chunk = bit_array[i:i + k]
        i_bits = chunk[:k_half]
        q_bits = chunk[k_half:]

        i_val = int("".join(map(str, i_bits)), 2)
        q_val = int("".join(map(str, q_bits)), 2)

        i_amp = 2 * i_val - (2**k_half - 1)
        q_amp = 2 * q_val - (2**k_half - 1)

        symbols.append(complex(i_amp, q_amp) / norm_factor)

    return np.array(symbols)


def pulse_shape_symbols(symbols, samples_per_symbol):
    """Apply square pulse shaping by repeating each symbol."""
    return np.repeat(symbols, samples_per_symbol)


def upconvert_to_passband(baseband_signal, fs, carrier_freq):
    """Modulate complex baseband samples onto a real passband carrier."""
    t = np.arange(len(baseband_signal)) / fs
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    return np.real(baseband_signal * carrier)


def apply_tukey_window(signal, alpha=0.01):
    """Apply a Tukey window to smooth the start and end of a signal."""
    return signal * tukey(len(signal), alpha=alpha)


def normalize_audio(signal):
    """Normalize audio to unit peak amplitude, leaving all-zero signals unchanged."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal / peak
    return signal


def to_int16_pcm(signal):
    """Convert normalized floating-point audio into 16-bit PCM samples."""
    return np.int16(signal * 32767)
