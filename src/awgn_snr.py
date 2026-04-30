import numpy as np

def add_awgn_snr(signal, snr_db):
    """
    Adds AWGN noise to a WAV signal safely without modifying external pipeline.

    Handles:
    - int16 / int32 / float inputs
    - stereo → mono (optional handling)
    - normalization internally
    - returns same dtype as input

    Parameters:
    signal : array-like (raw wav data)
    snr_db : desired SNR in dB

    Returns:
    noisy_signal (same dtype as input)
    """

    original_dtype = signal.dtype

    # Convert to float for processing
    signal = signal.astype(np.float64)

    # Handle stereo → keep shape but process per channel
    if signal.ndim == 2:
        # Process each channel independently
        noisy = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            s = signal[:, ch]

            # Normalize
            max_val = np.max(np.abs(s)) if np.max(np.abs(s)) != 0 else 1
            s_norm = s / max_val

            # Power
            power = np.mean(s_norm**2)

            # Noise variance
            snr_linear = 10**(snr_db / 10)
            noise_var = power / snr_linear

            # Noise
            noise = np.random.normal(0, np.sqrt(noise_var), s_norm.shape)

            # Add noise and de-normalize
            noisy[:, ch] = (s_norm + noise) * max_val

    else:
        # Mono
        max_val = np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else 1
        signal_norm = signal / max_val

        power = np.mean(signal_norm**2)
        snr_linear = 10**(snr_db / 10)
        noise_var = power / snr_linear

        noise = np.random.normal(0, np.sqrt(noise_var), signal_norm.shape)
        noisy = (signal_norm + noise) * max_val

    # Convert back to original dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        noisy = np.clip(noisy, info.min, info.max)
        noisy = noisy.astype(original_dtype)
    else:
        noisy = noisy.astype(original_dtype)

    return noisy

