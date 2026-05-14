# %% [markdown]
# # Setup and Imports
# 

# %%
# Run this cell first in every notebook
import os
import sys

sys.path.append(os.path.abspath('../src'))

%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from demodulator import (
    downconvert_to_baseband,
    estimate_baseband_bandwidth,
    load_mono_normalized_audio,
    locate_pilot_start,
    matched_filter_sync,
    slice_pilot_samples,
    downsample_from_offset,
    symbols_to_bits
)
from modulator import map_bits_to_symbols
from plot_utils import plot_complex_parts, plot_correlation_response, plot_fft
from sync import generate_zadoff_chu_audio
import config
from hamming import hamming74_decode, hamming74_encode


# %% [markdown]
# # System Parameters
# 

# %%
# Match these exactly to TX.
FS = config.FS
F_CARRIER = config.F_CARRIER
SYMBOL_RATE = config.SYMBOL_RATE
SAMPLES_PER_SYMBOL = config.SAMPLES_PER_SYMBOL
ZC_ROOT = config.u_zadoff_chu
ZC_LENGTH = config.len_zadoff_chu
SYNC_GAP_SECONDS = config.SYNC_GAP_SECONDS

rx_path = '../data/untitled.wav'

""" rx_path = '../data/tx_single_carrier_test.wav' """

# %% [markdown]
# # Load and Downconvert Audio
# 

# %%
fs, rx_audio = load_mono_normalized_audio(rx_path)
if fs != FS:
    raise ValueError(f'Expected sample rate {FS}, got {fs}')

BASEBAND_CUTOFF_HZ = 6000
recommended_cutoff = estimate_baseband_bandwidth(
    SYMBOL_RATE,
    pulse_method=config.pulse_method,
    margin=1.1,
)

print(f'RX path: {rx_path}')
print(f'Sample rate: {fs} Hz')
print(f'Audio samples: {len(rx_audio)}')
print(f'Audio duration: {len(rx_audio) / fs:.6f} s')
print(f'Baseband LPF cutoff used: {BASEBAND_CUTOFF_HZ} Hz')
print(f'Estimated cutoff for {config.pulse_method} pulse with 10% margin: {recommended_cutoff:.1f} Hz')

rx_baseband = downconvert_to_baseband(
    rx_audio=rx_audio,
    fs=FS,
    carrier_freq=F_CARRIER,
    cutoff_hz=BASEBAND_CUTOFF_HZ,
    filter_order=5,
)

# %%
plot_fft(rx_audio,fs)

# %%
plot_fft(rx_baseband,fs)

# %%
plot_complex_parts(rx_audio)

# %%
reference_zc = generate_zadoff_chu_audio(ZC_LENGTH, ZC_ROOT)
zc_start_idx, corr = matched_filter_sync(rx_baseband, reference_zc)

pilot_start_idx = locate_pilot_start(
    sync_start_idx=zc_start_idx,
    sync_length=ZC_LENGTH,
    fs=FS,
    gap_seconds=SYNC_GAP_SECONDS,
)

print('-' * 50)
print(f'Zadoff-Chu peak found at sample index: {zc_start_idx}')
print(f'Zadoff-Chu peak time: {zc_start_idx / FS:.6f} s')
print(f'Expected ZC start for repaired TX: {int(5.0 * FS)}')
print(f'Correlation peak / median: {corr[zc_start_idx] / (np.median(corr) + 1e-12):.3e}')
print(f'Sync length samples: {ZC_LENGTH}')
print(f'Sync gap samples: {int(FS * SYNC_GAP_SECONDS)}')
print(f'Known pilots begin at sample index: {pilot_start_idx}')
print(f'Expected pilot waveform start for repaired TX: {int(5.0 * FS) + ZC_LENGTH}')
print('-' * 50)


# %%
plot_correlation_response(corr, zc_start_idx)
plt.show()


# %% [markdown]
# # Pilot Extraction
# 

# %%
pilot_bits = config.pilot_bits
pilot_symbols = map_bits_to_symbols(pilot_bits, config.M, METHOD=config.modulation_method)
KNOWN_PILOT_SYMBOLS = pilot_symbols

rx_pilots = slice_pilot_samples(
    rx_baseband=rx_baseband,
    pilot_start_idx=pilot_start_idx,
    num_pilot_symbols=len(KNOWN_PILOT_SYMBOLS),
    samples_per_symbol=SAMPLES_PER_SYMBOL,
    span_symbols=config.SPAN,
)

print(f'Pilot bits: {len(pilot_bits)}')
print(f'Pilot symbols: {len(KNOWN_PILOT_SYMBOLS)}')
print(f'Samples/symbol: {SAMPLES_PER_SYMBOL}')
print(f'SINC span symbols from config: {config.SPAN}')
print(f'SINC leading delay samples: {config.SPAN * SAMPLES_PER_SYMBOL}')
print(f'Pilot slice samples: {len(rx_pilots)}')
print(f'Expected pilot slice samples: {(len(KNOWN_PILOT_SYMBOLS) + config.SPAN) * SAMPLES_PER_SYMBOL}')


# %%
plot_complex_parts(pilot_symbols)

# %%
plot_complex_parts(rx_pilots, title_prefix='rx_pilots')
plt.show()

# %% [markdown]
# ### Synchronization 

# %%
L = SAMPLES_PER_SYMBOL
KNOWN = KNOWN_PILOT_SYMBOLS

# ================================
# REMOVE SINC DELAY REGION
# ================================
rxp = rx_pilots[config.SPAN * L:]

NUM_PILOTS = len(KNOWN)

# ================================
# FIND BEST OFFSET
# ================================
best_offset = 0
best_metric = -np.inf
offset_metrics = []

for offset in range(L):
    samples = rxp[offset::L][:NUM_PILOTS]
    
    metric = np.abs(np.sum(samples * np.conj(KNOWN)))
    offset_metrics.append(metric)
    
    if metric > best_metric:
        best_metric = metric
        best_offset = offset

print(f"Offset metrics for offsets 0..{L - 1}: {[float(m) for m in offset_metrics]}")
print(f"Best offset inside pilot slice: {best_offset}")
print(f"Best pilot timing metric: {best_metric:.6f}")
print(f"First pilot decision sample index: {pilot_start_idx + config.SPAN * L + best_offset}")
print(f"First message decision sample index: {pilot_start_idx + config.SPAN * L + best_offset + NUM_PILOTS * L}")

# ================================
# GET FINAL SAMPLE INDICES
# ================================
sample_indices = np.arange(best_offset, best_offset + NUM_PILOTS*L, L)

# ================================
# PLOT REAL PART
# ================================
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

plt.plot(np.real(rxp), label="Pilot Signal (Real)", alpha=0.7)

plt.scatter(sample_indices,
            np.real(rxp[sample_indices]),
            color='red',
            label="Sampled Points",
            zorder=5)

plt.title("Pilot Sampling Alignment (Real)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# ================================
# PLOT IMAG PART
# ================================
plt.figure(figsize=(20,5))

plt.plot(np.imag(rxp), label="Pilot Signal (Imag)", alpha=0.7)

plt.scatter(sample_indices,
            np.imag(rxp[sample_indices]),
            color='green',
            label="Sampled Points",
            zorder=5)

plt.title("Pilot Sampling Alignment (Imag)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# ================================
# VERIFY SYMBOLS
# ================================
rx_samples = rxp[best_offset::L][:NUM_PILOTS]
pilot_channel = np.vdot(KNOWN, rx_samples) / np.vdot(KNOWN, KNOWN)
rx_samples_equalized = rx_samples / pilot_channel
recovered_pilot_bits = symbols_to_bits(rx_samples_equalized, M=config.M, METHOD=config.modulation_method)
pilot_bit_errors = np.sum(recovered_pilot_bits[:len(pilot_bits)] != pilot_bits)

print("\nKnown pilot symbols:")
print(KNOWN)

print("\nRecovered pilot samples:")
print(rx_samples)

print("\nEstimated pilot channel scalar:")
print(pilot_channel)
print(f"abs={np.abs(pilot_channel):.6f}, phase_rad={np.angle(pilot_channel):.6f}")

print("\nEqualized recovered pilot samples:")
print(rx_samples_equalized)
print(f"Pilot bit errors after equalizing pilots: {pilot_bit_errors} / {len(pilot_bits)}")

# %%
start = pilot_start_idx + config.SPAN * L + best_offset

downsampled_symbols, ds_signal = downsample_from_offset(
    rx_baseband,
    start,
    L
)

NUM_PILOT_SYMBOLS = len(KNOWN_PILOT_SYMBOLS)
NUM_MESSAGE_SYMBOLS = config.total_symbols
message_symbol_start = NUM_PILOT_SYMBOLS
message_symbol_stop = message_symbol_start + int(NUM_MESSAGE_SYMBOLS * 7 / 4)

print(f'Downsample start index: {start}')
print(f'First downsampled symbol is pilot symbol 0')
print(f'Pilot symbols to skip before message: {NUM_PILOT_SYMBOLS}')
print(f'Pilot bits to skip before message: {len(pilot_bits)}')
print(f'Message symbol range in downsampled array: [{message_symbol_start}, {message_symbol_stop})')
print(f'First message sample index: {start + NUM_PILOT_SYMBOLS * L}')
print(f'Downsampled symbols available: {len(downsampled_symbols)}')

# %%
plt.figure(figsize=(20,5))

plt.plot(np.real(rx_baseband), label="Original Signal", alpha=0.5)

sample_indices = np.arange(start, len(rx_baseband), L)

plt.scatter(sample_indices,
            np.real(rx_baseband[sample_indices]),
            color='red',
            label="Sampled Points",
            zorder=5)

plt.title("Full Signal Sampling")
plt.grid(True)
plt.legend()
plt.show()

# %%
plot_complex_parts(rx_baseband[sample_indices][0:100])

# %%
pilot_symbol_count = len(KNOWN_PILOT_SYMBOLS)
message_symbols = downsampled_symbols[pilot_symbol_count : pilot_symbol_count + int(config.total_symbols * 7 / 4)]

# Apply the equalization scalar to the downsampled and message symbols
downsampled_symbols_equalized = downsampled_symbols / pilot_channel
message_symbols_equalized = message_symbols / pilot_channel

recovered_bits_with_pilots = symbols_to_bits(
    downsampled_symbols_equalized,  # <-- Use equalized symbols here
    M=config.M,
    METHOD=config.modulation_method
)

recovered_bits = symbols_to_bits(
    message_symbols_equalized,      # <-- Use equalized symbols here
    M=config.M,
    METHOD=config.modulation_method
)
print(len(recovered_bits))

recovered_bits = hamming74_decode(recovered_bits)
print(len(recovered_bits))

print(f'Demodulated symbols including pilots: {len(downsampled_symbols)}')
print(f'Message symbols used for BER: {len(message_symbols)}')
print(f'Skipped pilot symbols before BER: {pilot_symbol_count}')
print(f'Skipped pilot bits before BER: {len(pilot_bits)}')

# %%
input_path = "../data/message.txt"
with open(input_path, "r") as f:
        data_binary = f.read().strip()


# Calculate Bit Error Rate (BER)
recovered_bit_string_with_pilots = ''.join(map(str, recovered_bits_with_pilots))
recovered_bit_string = ''.join(map(str, recovered_bits))

# This is the old alignment, kept only as a diagnostic. It should be bad for the repaired frame.
wrong_min_length = min(len(recovered_bit_string_with_pilots), len(data_binary))
wrong_bit_errors = sum(
    r != d
    for r, d in zip(recovered_bit_string_with_pilots[:wrong_min_length], data_binary[:wrong_min_length])
)
wrong_ber = wrong_bit_errors / wrong_min_length if wrong_min_length > 0 else 0

# Determine the length to compare (use the shorter length or match as needed)
min_length = min(len(recovered_bit_string), len(data_binary))

# Compare bits
bit_errors = sum(r != d for r, d in zip(recovered_bit_string[:min_length], data_binary[:min_length]))

# Calculate BER
ber = bit_errors / min_length if min_length > 0 else 0

print(f"Recovered bits: {recovered_bit_string[:min_length]}")
print(f"Original bits:  {data_binary[:min_length]}")
print(f"\nTotal bits compared: {min_length}")
print(f"Bit errors: {bit_errors}")
print(f"Bit Error Rate (BER): {ber:.6f} ({ber*100:.2f}%)")

# %%
plt.figure(figsize=(6,6))

no_of_symbols_to_plot = 500
plot_symbols = message_symbols_equalized[: -1]
plt.scatter(np.real(plot_symbols),
            np.imag(plot_symbols),
            s=10)

plt.axhline(0)
plt.axvline(0)

plt.title("Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axis('equal')

plt.show()


