from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import welch


DEFAULT_BANDS = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 28.0),
    (30.0, 45.0),
)


def sliding_window(data: np.ndarray, window_length: int, overlap: float) -> np.ndarray:
    """Split a time x channel signal into overlapping windows."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(window_length * (1 - overlap)))
    windows = []
    start = 0
    end = window_length
    # Match the original experimental preprocessing, which omits the exact
    # right endpoint and yields 16/23/36/76 for 4/3/2/1 s windows in 20 s.
    while end < data.shape[0]:
        windows.append(data[start:end])
        start += step
        end = start + window_length
    if not windows:
        raise ValueError("window_length is longer than the input signal")
    return np.stack(windows, axis=0)


def relative_bandpower(
    segment: np.ndarray,
    sampling_rate: int,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    """Compute channel x band relative PSD features with Welch's method."""
    if segment.ndim != 2:
        raise ValueError("segment must have shape (time, channels)")

    freqs, psd = welch(
        segment,
        fs=sampling_rate,
        axis=0,
        nperseg=min(segment.shape[0], sampling_rate * 2),
    )
    total_mask = (freqs >= bands[0][0]) & (freqs <= bands[-1][1])
    total_power = np.trapz(psd[total_mask], freqs[total_mask], axis=0)
    total_power = np.maximum(total_power, 1.0e-12)

    features = []
    for low, high in bands:
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.trapz(psd[band_mask], freqs[band_mask], axis=0)
        features.append(band_power / total_power)
    return np.stack(features, axis=-1)


def make_multiscale_rpsd(
    raw_eeg: np.ndarray,
    sampling_rate: int,
    first_window_seconds: float = 20.0,
    first_hop_seconds: float = 4.0,
    sub_window_seconds: Sequence[float] = (4.0, 3.0, 2.0, 1.0),
    sub_overlap: float = 0.75,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> list[np.ndarray]:
    """Create MSGM multi-scale rPSD tensors from a raw EEG trial.

    Args:
        raw_eeg: Array with shape (time, channels).
        sampling_rate: EEG sampling rate in Hz.
        first_window_seconds: First-level segment length.
        first_hop_seconds: First-level hop size.
        sub_window_seconds: Second-level window sizes, one per temporal scale.
        sub_overlap: Overlap ratio for second-level windows.
        bands: Frequency bands for rPSD extraction.

    Returns:
        A list of arrays, one per scale. Each array has shape
        (first_segments, scale_sequence_length, channels, frequency_bands).
    """
    if raw_eeg.ndim != 2:
        raise ValueError("raw_eeg must have shape (time, channels)")

    first_window = int(round(first_window_seconds * sampling_rate))
    first_hop = int(round(first_hop_seconds * sampling_rate))
    first_overlap = 1.0 - first_hop / first_window
    first_segments = sliding_window(raw_eeg, first_window, first_overlap)

    outputs = []
    for seconds in sub_window_seconds:
        sub_window = int(round(seconds * sampling_rate))
        scale_trials = []
        for segment in first_segments:
            sub_segments = sliding_window(segment, sub_window, sub_overlap)
            scale_trials.append(
                np.stack(
                    [relative_bandpower(sub, sampling_rate, bands=bands) for sub in sub_segments],
                    axis=0,
                )
            )
        outputs.append(np.stack(scale_trials, axis=0))
    return outputs
