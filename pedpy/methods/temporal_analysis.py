"""Module containing functions to compute temporal analysis methods.

For example: Short-Time Fourier Transform (STFT) and Welch's method for spectral analysis.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import stft, welch


def compute_stft(
    signal_series: pd.Series,
    frame_rate: int,
    segments_length: Optional[int] = None,
    overlap_length: Optional[int] = None,
    zeros_padded: Optional[int] = None,
    window: str = "hann",
) -> pd.DataFrame:
    r"""Computes the Short-Time Fourier Transform (STFT) of a signal.

    This function calculates the time-frequency representation of a signal
    using the Short-Time Fourier Transform (STFT). The STFT provides
    information about the frequency content of the signal over time by
    computing the Fourier Transform within a sliding window.

    The output consists of the magnitude and phase of the STFT, allowing
    for both amplitude and phase analysis.

    .. math::
        STFT\{x[n]\}(m, k) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j 2 \pi k n / N}

    where :math:`x[n]` is the discrete-time signal, :math:`w[n]` is the
    window function, :math:`m` is the time index, :math:`k` is the
    frequency index, and :math:`N` is the number of FFT points.

    Args:
        signal_series (pd.Series): A pandas Series containing data values measured at a constant time interval.
        frame_rate (int): The frame rate of the signal data. The frame rate
            has to remain constant throughout the whole dataset.
        segments_length (int, optional): Length of each segment for the STFT window.
            Defaults to 5 times `frame_rate`.
        overlap_length (int, optional): Number of overlapping points between
            segments. Defaults to None (half of `segments_length` is used).
        zeros_padded (int, optional): Number of FFT points. Defaults to None
            (5 times `segments_length`).
        window (str, optional): The window function to apply before
            computing the STFT. Defaults to `'hann'`. Other options are
            `'hamming'`, `'bartlett'`, `'blackman'`, `'boxcar'`, `'triang'`, etc.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - `"Frequency"`: The frequency bins of the STFT.
            - `"Time"`: The time bins corresponding to the STFT computation.
            - `"Magnitude"`: The absolute magnitude of the STFT at each
              time-frequency point.
            - `"Phase"`: The phase of the STFT at each time-frequency point.
    """
    if segments_length is None:
        segments_length = frame_rate * 5

    if overlap_length is None:
        overlap_length = segments_length // 2

    if zeros_padded is None:
        zeros_padded = 5 * segments_length

    f, t, zxx = stft(
        signal_series.values,
        fs=frame_rate,
        nperseg=segments_length,
        noverlap=overlap_length,
        nfft=zeros_padded,
        window=window,
    )

    return pd.DataFrame(
        {
            "Frequency": np.repeat(f, len(t)),
            "Time": np.tile(t, len(f)),
            "Magnitude": np.abs(zxx).flatten(),
            "Phase": np.angle(zxx).flatten(),
        }
    )


def compute_welch_spectral_distribution(
    signal_series: pd.Series,
    frame_rate: int,
    segments_length: Optional[int] = None,
    overlap_length: Optional[int] = None,
    zeros_padded: Optional[int] = None,
    window: str = "hann",
) -> pd.DataFrame:
    """Computes the power spectral density of a signal using Welch's method.

    This function estimates the power spectral density (PSD) of a signal by
    splitting it into overlapping segments, computing a modified periodogram
    for each segment, and averaging the periodograms. This is Welch's
    method, as implemented by :func:`scipy.signal.welch`.

    Args:
        signal_series (pd.Series): A pandas Series containing data values measured at a constant time interval.
        frame_rate (int): The frame rate of the signal data. The frame rate
            has to remain constant throughout the whole dataset.
        segments_length (int, optional): Length of each segment used to
            estimate the PSD. Defaults to one third of `signal_series` length.
        overlap_length (int, optional): Number of overlapping points between
            segments. Defaults to None (half of `segments_length` is used).
        zeros_padded (int, optional): Number of FFT points. Defaults to None
            (5 times `segments_length`).
        window (str, optional): The window function to apply before
            computing the PSD. Defaults to `'hann'`. Other options are
            `'hamming'`, `'bartlett'`, `'blackman'`, `'boxcar'`, `'triang'`, etc.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - `"Frequency"`: The frequency bins of the spectral distribution.
            - `"Power"`: The power spectral density at each frequency bin.
    """
    if segments_length is None:
        segments_length = len(signal_series) // 3

    if overlap_length is None:
        overlap_length = segments_length // 2

    if zeros_padded is None:
        zeros_padded = 5 * segments_length

    f, pxx = welch(
        signal_series.values,
        fs=frame_rate,
        nperseg=segments_length,
        noverlap=overlap_length,
        nfft=zeros_padded,
        window=window,
    )

    return pd.DataFrame(
        {
            "Frequency": f,
            "Power": pxx,
        }
    )
