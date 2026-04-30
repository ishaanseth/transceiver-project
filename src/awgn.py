import numpy as np

def add_awgn(signal, variance):
    """
    Adds AWGN noise to a signal.

    Parameters:
    signal (array-like): Input signal
    variance (float): Noise variance (sigma^2)

    Returns:
    noisy_signal (numpy array): Signal with added AWGN
    """
    signal = np.array(signal)

    # Generate noise with mean 0 and given variance
    noise = np.random.normal(0, np.sqrt(variance), size=signal.shape)

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal
