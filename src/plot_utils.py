from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_PLOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "plots"


def save_important_plot(filename, fig=None, plots_dir=DEFAULT_PLOTS_DIR, dpi=300, **savefig_kwargs):
    """Save a selected plot to data/plots for report-ready figures.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output file name. If no extension is provided, ".png" is used.
    fig : matplotlib.figure.Figure, optional
        Figure to save. Defaults to the current active matplotlib figure.
    plots_dir : str or pathlib.Path, optional
        Destination folder. Defaults to the project's data/plots folder.
    dpi : int, optional
        Resolution for raster outputs.
    **savefig_kwargs
        Extra keyword arguments passed to matplotlib's savefig.
    """
    fig = fig or plt.gcf()
    plots_dir = Path(plots_dir)
    output_path = Path(filename)

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")

    if not output_path.is_absolute():
        output_path = plots_dir / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    savefig_options = {"bbox_inches": "tight", "dpi": dpi}
    savefig_options.update(savefig_kwargs)
    fig.savefig(output_path, **savefig_options)

    return output_path


def plot_complex_parts(signal, title_prefix="Signal"):
    """Plot the real and imaginary parts of a complex-valued signal."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(np.real(signal), color="blue", marker="o", linestyle="-", linewidth=1.5)
    axes[0].set_title(f"Real Part of {title_prefix}")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.imag(signal), color="red", marker="o", linestyle="-", linewidth=1.5)
    axes[1].set_title(f"Imaginary Part of {title_prefix}")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_waveform(data, sample_rate=None, title="Waveform"):
    """Plot an audio waveform against samples or seconds."""
    if sample_rate is None:
        x = np.arange(len(data))
        xlabel = "Samples"
    else:
        x = np.arange(len(data)) / sample_rate
        xlabel = "Time (seconds)"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


def plot_fft(audio_data, fs, title="FFT"):
    """Plot the positive-frequency magnitude spectrum of an audio signal."""
    fft_result = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), 1 / fs)

    positive_freqs = freqs[:len(freqs) // 2]
    magnitude = np.abs(fft_result[:len(fft_result) // 2]) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(positive_freqs, magnitude)
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    ax.set_xlim([0, fs / 2])
    plt.tight_layout()
    return fig, ax


def plot_correlation_response(corr, peak_idx, title="Zadoff-Chu Matched Filter Response", window=5000):
    """Plot a correlation response centered around the detected peak."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(corr, color="blue")
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Correlation Magnitude")
    ax.axvline(peak_idx, color="red", linestyle="--", label=f"Sync Peak ({peak_idx})")
    ax.set_xlim(max(0, peak_idx - window), min(len(corr), peak_idx + window))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax
