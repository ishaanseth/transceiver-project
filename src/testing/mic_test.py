import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def analyze_recording(input_path):
    """Plots the spectrogram of a recorded audio file."""
    try:
        fs, data = wavfile.read(input_path)
        print(f"Loaded {input_path}. Sample rate: {fs} Hz")
    except FileNotFoundError:
        print(f"Error: Could not find '{input_path}'. Did you put it in the data folder?")
        return

    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Calculate spectrogram
    frequencies, times, Sxx = spectrogram(data, fs, nperseg=1024, noverlap=512)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.title(f'Spectrogram of {input_path}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, fs // 2) 
    plt.tight_layout()
    plt.show()