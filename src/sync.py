import numpy as np

def append_sync_sequence(data_symbols, sync_sequence):
    """
    Appends the time-domain sync sequence to the start of the transmission.
    """
    return np.concatenate((sync_sequence, data_symbols))


def generate_zadoff_chu_audio(N, u):
    """Generate the Zadoff-Chu sequence used as the audio sync preamble."""
    q = 0
    n_0 = np.arange(0, N, 1)
    n_1 = np.arange(1, N + 1, 1)

    if N % 2 == 1:
        return np.exp(-1j * u * np.pi / N * np.multiply(n_0, (n_1 + 2 * q)))
    return np.exp(-1j * u * np.pi / N * (n_0**2))


def make_silence(duration_seconds, fs, dtype=complex):
    """Create a zero-valued silence block for the requested duration."""
    return np.zeros(int(fs * duration_seconds), dtype=dtype)


def assemble_baseband_frame(
    sync_audio,
    payload_signal,
    fs,
    start_silence=5.0,
    middle_silence=0.05,
    end_silence=5.0,
):
    """Assemble the transmit frame: silence, sync, pilots, payload, silence."""
    silence_gap_start = make_silence(start_silence, fs, dtype=complex)
    silence_gap_middle = make_silence(middle_silence, fs, dtype=complex)
    silence_gap_end = make_silence(end_silence, fs, dtype=complex)

    return np.concatenate([
        silence_gap_start,
        sync_audio,
        silence_gap_middle,
        payload_signal,
        silence_gap_end,
    ])
