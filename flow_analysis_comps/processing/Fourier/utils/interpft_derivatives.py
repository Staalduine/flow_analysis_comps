import numpy as np
from flow_analysis_comps.processing.Fourier.utils.Interpolation import (
    interpolate_fourier_series,
)


# TODO: Check if this works
def interpft1_derivatives(
    fourier_coeffs,
    query_points,
    derivative_orders,
    period=None,
    is_freq_domain=False,
    method="horner_freq",
):
    """
    Interpolate the derivatives of a Fourier series.

    Parameters
    ----------
    fourier_coeffs : np.ndarray
        Fourier coefficients or values.
    query_points : np.ndarray
        Query points.
    derivative_orders : array-like
        Derivative orders to compute.
    period : float, optional
        Period of the domain. Default is 2*pi.
    is_freq_domain : bool, optional
        If True, fourier_coeffs is already in frequency domain. Default is False.
    method : str, optional
        Interpolation method. Default is 'horner_freq'.

    Returns
    -------
    interpolated_derivatives : np.ndarray
        Interpolated derivatives at query_points.
    """
    if period is None:
        period = 2 * np.pi
        period_scaling_factor = 1
    else:
        period_scaling_factor = 2 * np.pi / period

    num_modes = int(np.floor(fourier_coeffs.shape[0] / 2))
    freq_indices = np.fft.ifftshift(np.arange(-num_modes, num_modes + 1)) * 1j

    if not is_freq_domain:
        fourier_coeffs_freq = np.fft.fft(fourier_coeffs, axis=0)
    else:
        fourier_coeffs_freq = fourier_coeffs

    coeffs_ndim = fourier_coeffs.ndim + 1
    derivative_orders = np.asarray(derivative_orders).reshape(
        (-1,) + (1,) * (coeffs_ndim - 1)
    )
    freq_indices_powers = freq_indices[:, None] ** derivative_orders.reshape(1, -1)

    # Multiply fourier_coeffs_freq by freq_indices_powers for each derivative
    fourier_coeffs_freq = (
        fourier_coeffs_freq[..., None] * freq_indices_powers
    )  # shape: (..., n_derivs)

    # Prepare query_points for all derivatives
    query_points_repeat = [1] * coeffs_ndim
    query_points_repeat[-1] = len(derivative_orders)
    query_points_full = np.tile(query_points, query_points_repeat)

    interpolated_derivatives = interpolate_fourier_series(
        [0, period], fourier_coeffs_freq, query_points_full, method
    )

    if period_scaling_factor != 1:
        # Adjust for period scaling
        scaling_factors = period_scaling_factor ** derivative_orders.flatten()
        # Broadcast scaling_factors to match interpolated_derivatives shape
        scaling_shape = [1] * (interpolated_derivatives.ndim - 1) + [
            len(scaling_factors)
        ]
        interpolated_derivatives = interpolated_derivatives * scaling_factors.reshape(
            scaling_shape
        )

    return interpolated_derivatives
