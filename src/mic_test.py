import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Load the recorded file
# Ensure the file is named correctly and is in the same folder
fs, data = wavfile.read("recorded_chirp.wav")

# If the recording is stereo (2 channels), convert to mono
if len(data.shape) > 1:
    data = data.mean(axis=1)

# Plot the Spectrogram
frequencies, times, Sxx = spectrogram(data, fs, nperseg=1024, noverlap=512)

plt.figure(figsize=(10, 6))
# Convert to Decibels for better visualization
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
plt.title('Spectrogram of Received Chirp')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time[sec]')
plt.colorbar(label='Intensity [dB]')
plt.ylim(0, 24000) # Limit view up to 24kHz
plt.tight_layout()
plt.show()