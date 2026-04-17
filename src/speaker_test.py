import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp
import os

def generate_chirp_file(output_path="../data/chirp_test.wav", fs=48000, duration=5.0, f0=20, f1=24000):
    """Generates a logarithmic frequency sweep to test speaker capabilities."""
    print(f"Generating chirp from {f0}Hz to {f1}Hz over {duration} seconds...")
    
    # Generate time array
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Generate the chirp signal
    signal = chirp(t, f0=f0, f1=f1, t1=duration, method='logarithmic')
    
    # Fade in/out to prevent speaker "pop" sounds
    fade_len = int(fs * 0.05)
    window = np.ones(len(signal))
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)
    signal = signal * window
    
    # Convert to 16-bit PCM Audio
    signal_16bit = np.int16(signal * 32767)
    
    # Save to WAV inside the data folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, fs, signal_16bit)
    print(f"Success! Saved to {output_path}")