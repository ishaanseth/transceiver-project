import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

def plot_hardware_response(tx_path, rx_path):
    """
    Plots the Power Spectral Density (PSD) of the transmitted and received
    audio files to estimate the end-to-end hardware frequency response.
    """
    try:
        fs_tx, tx_data = wavfile.read(tx_path)
        fs_rx, rx_data = wavfile.read(rx_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Convert to mono if stereo
    if len(tx_data.shape) > 1: tx_data = tx_data.mean(axis=1)
    if len(rx_data.shape) > 1: rx_data = rx_data.mean(axis=1)

    # Normalize audio to prevent arbitrary volume differences from skewing the graph
    tx_data = tx_data / np.max(np.abs(tx_data))
    rx_data = rx_data / np.max(np.abs(rx_data))

    # Welch's method computes an averaged periodogram (great for noise reduction)
    # nperseg defines frequency resolution. 4096 gives good smooth curves.
    f_tx, Pxx_tx = welch(tx_data, fs_tx, nperseg=4096)
    f_rx, Pxx_rx = welch(rx_data, fs_rx, nperseg=4096)

    # Convert Power to Decibels (dB)
    Pxx_tx_dB = 10 * np.log10(Pxx_tx + 1e-12)
    Pxx_rx_dB = 10 * np.log10(Pxx_rx + 1e-12)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(f_tx, Pxx_tx_dB, label='Transmitted (Laptop Speaker)', color='blue', alpha=0.7)
    plt.plot(f_rx, Pxx_rx_dB, label='Received (Phone Mic)', color='orange', linewidth=2)
    
    plt.title('End-to-End Frequency Response (PSD)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [dB/Hz]')
    plt.xlim(0, max(fs_tx, fs_rx) // 2)
    plt.ylim(min(np.min(Pxx_rx_dB), np.min(Pxx_tx_dB)) - 10, max(np.max(Pxx_tx_dB), np.max(Pxx_rx_dB)) + 10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
