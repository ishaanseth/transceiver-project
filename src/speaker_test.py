import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp

# Parameters
fs = 48000        # 48 kHz Sample Rate
duration = 5.0    # 5 seconds
f0 = 20           # Start frequency (20 Hz)
f1 = 24000        # End frequency (24 kHz)

# Generate time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate the chirp signal (Logarithmic sweep is best for audio)
signal = chirp(t, f0=f0, f1=f1, t1=duration, method='logarithmic')

# Apply a slight fade-in and fade-out to prevent speaker popping
fade_len = int(fs * 0.05) # 50ms fade
window = np.ones(len(signal))
window[:fade_len] = np.linspace(0, 1, fade_len)
window[-fade_len:] = np.linspace(1, 0, fade_len)
signal = signal * window

# Convert to 16-bit PCM Audio
signal_16bit = np.int16(signal * 32767)

# Save to WAV
wavfile.write("chirp_test.wav", fs, signal_16bit)
print("chirp_test.wav generated successfully!")