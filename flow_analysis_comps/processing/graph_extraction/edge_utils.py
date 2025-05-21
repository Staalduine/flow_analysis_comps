import numpy as np
from scipy.signal import butter, sosfiltfilt


def low_pass_filter(coords, cutoff_freq=0.01, order=2):
    """
    Applies a low-pass Butterworth filter to (x, y) coordinates.

    Parameters:
    - coords: np.ndarray of shape (N, 2), where N is the number of points [(x1, y1), (x2, y2), ...]
    - cutoff_freq: Cutoff frequency (0 < f < 0.5, relative to Nyquist frequency)
    - order: Order of the Butterworth filter (higher means sharper cutoff)

    Returns:
    - Filtered np.ndarray of shape (N, 2)
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "Input must be an array of shape (N, 2) representing (x, y) coordinates."
        )

    # Create Butterworth low-pass filter
    b = butter(N=order, Wn=cutoff_freq, btype="lowpass", analog=False, output="sos")

    # Apply the filter to both x and y coordinates separately
    x_filtered = sosfiltfilt(b, coords[:, 0], padlen=100)
    y_filtered = sosfiltfilt(b, coords[:, 1], padlen=100)

    return np.column_stack((x_filtered, y_filtered))


def resample_trail(trail):
    trail = np.array(trail)  # Ensure it's an array
    distances = np.sqrt(np.sum(np.diff(trail, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(
        np.cumsum(distances), 0, 0
    )  # Insert 0 at the start

    new_distances = np.arange(
        0, cumulative_distances[-1], 1
    )  # New samples at distance 1
    new_trail = np.array(
        [
            np.interp(new_distances, cumulative_distances, trail[:, dim])
            for dim in range(trail.shape[1])
        ]
    ).T  # Interpolate each coordinate separately

    return new_trail  # Ensure integer pixel coordinates
