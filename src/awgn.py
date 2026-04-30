import numpy as np

def add_awgn_snr(signal, snr_db):
    signal = np.array(signal)

    # Signal power
    signal_power = np.mean(signal**2)

    # Convert SNR to linear
    snr_linear = 10**(snr_db / 10)

    # Compute noise variance
    noise_variance = signal_power / snr_linear

    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_variance), signal.shape)

    return signal + noise