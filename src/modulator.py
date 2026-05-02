import numpy as np
from scipy.signal.windows import tukey


def map_bits_to_symbols(bit_array, M=4,METHOD="QAM"):
    if METHOD=="QAM":
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
    #else PAM


def pulse_shape_symbols(symbols, samples_per_symbol, METHOD="SINC"):
    
    if METHOD == "SQUARE":
        return np.repeat(symbols, samples_per_symbol)
    
    elif METHOD == "SINC":
        L = samples_per_symbol
        SPAN = 4  # ±4 symbols
        
        N = len(symbols)
        
        # Time axis
        t = np.arange(-SPAN*L, SPAN*L + 1) / L
        sinc_base = np.sinc(t)
        
        # Create buffer with padding
        baseband_signal = np.zeros((N + 2*SPAN) * L, dtype=complex)
        
        # Build signal
        for i in range(N):
            center = (i + SPAN) * L
            start = center - len(sinc_base)//2
            end   = start + len(sinc_base)
            
            baseband_signal[start:end] += symbols[i] * sinc_base
        
        return baseband_signal

    elif METHOD == "RRC":
        L = samples_per_symbol
        SPAN = 6       # filter span (in symbols)
        beta = 0.25    # roll-off factor
        
        N = len(symbols)
        
        # Time axis
        t = np.arange(-SPAN*L, SPAN*L + 1) / L
        
        # RRC pulse
        rrc = np.zeros_like(t)
        
        for i in range(len(t)):
            ti = t[i]
            
            if ti == 0.0:
                rrc[i] = 1.0 - beta + (4 * beta / np.pi)
            
            elif abs(ti) == 1 / (4 * beta):
                rrc[i] = (beta / np.sqrt(2)) * (
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
                
                rrc[i] = numerator / denominator
        
        # Normalize energy
        rrc = rrc / np.sqrt(np.sum(rrc**2))
        
        # Create buffer
        baseband_signal = np.zeros((N + 2*SPAN) * L, dtype=complex)
        
        # Convolve symbols with RRC pulse (same style as your sinc)
        for i in range(N):
            center = (i + SPAN) * L
            start = center - len(rrc)//2
            end   = start + len(rrc)
            
            baseband_signal[start:end] += symbols[i] * rrc
        
        return baseband_signal

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
