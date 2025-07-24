import numpy as np
from numpy.fft import fft, fftshift
from numpy.polynomial import Polynomial
from joblib import Parallel, delayed
from tqdm import tqdm
from flow_analysis_comps.processing.Fourier.utils.Interpolation import interpolate_fourier_series


def roots_batch(coeffs_batch):
    roots_out = []
    for coeffs in coeffs_batch:
        try:
            roots = np.roots(coeffs[::-1])  # MATLAB-style coeff order
        except Exception as e:
            print(e)
            roots = np.array([np.nan])
        roots_out.append(roots)
    return roots_out


def interpft_extrema_fast(
    filter_response, dim=1, sorted_output=False, TOL=1e-10, do_fft=False, n_jobs=-1
) -> dict:
    # Ensure filter_response is a numpy array
    filter_response = np.asarray(filter_response) # dims= (D, x, y)

    # Make sure response is in first dimension
    if dim != 0:
        filter_response = np.moveaxis(filter_response, dim, 0)

    # Get response shape
    response_shape = filter_response.shape
    response_depth = response_shape[0]

    # Check if response depth is valid
    if response_depth == 1:
        empty = np.zeros_like(filter_response)
        raise ValueError
        return tuple(np.moveaxis(arr, 0, dim) for arr in (empty, empty, empty, filter_response, filter_response, filter_response))

    # Ensure response is in frequency domain
    filter_response_fft = fft(filter_response, axis=0) if do_fft else filter_response

    # Get Nyquist frequency of response
    nyquist = int(np.ceil((response_depth + 1) / 2))

    # Adjust filter response for even/odd response depth
    if response_depth % 2 == 0:
        filter_response_fft = filter_response_fft.copy()  # Create a copy to avoid modifying the input array
        filter_response_fft[nyquist - 1] /= 2
        filter_response_fft = np.insert(filter_response_fft, nyquist, filter_response_fft[nyquist - 1], axis=0)

    # Create frequency array
    response_frequencies = np.concatenate([np.arange(nyquist), -np.arange(nyquist - 1, 0, -1)])
    response_frequencies = response_frequencies[..., np.newaxis, np.newaxis]  # Add new axis for broadcasting

    # Compute derivatives
    response_fft_derivative1 = filter_response_fft * (1j * response_frequencies)
    response_fft_derivative2 = filter_response_fft * -(response_frequencies**2)

    # Shift first derivative to center, flatten it
    response_fft_derivative1 = fftshift(response_fft_derivative1, axes=0)
    response_fft_deriv1_flat = response_fft_derivative1.reshape((response_fft_derivative1.shape[0], -1))

"""Recode below to the actual method"""
    # # Compute coefficients for polynomial roots
    # coefficients = [response_fft_deriv1_flat[:, i] for i in range(response_fft_deriv1_flat.shape[1])]
    # roots_out = list(Parallel(n_jobs=n_jobs)(
    #     delayed(np.roots)(coefficient[::-1]) for coefficient in tqdm(coefficients, desc="Finding roots", total=len(coefficients))
    # ))

    # # Find maximum number of roots, allocate space for output
    # safe_roots_out = [r if r is not None else [] for r in roots_out]
    # max_root_count = max(len(r) for r in safe_roots_out)
    # roots_array = np.full((max_root_count, len(safe_roots_out)), np.nan, dtype=complex)

    # # Fill roots array with roots
    # for i, ri in enumerate(roots_out):
    #     if ri is None:
    #         ri = []
    #     roots_array[: len(ri), i] = ri

    # # Reshape roots array to match filter response shape
    # roots_array = roots_array.reshape((max_root_count,) + filter_response.shape[1:])

    # # Compute magnitude and real map
    # epsilon = 1e-12  # Small value to avoid log(0)
    # magnitude = np.abs(np.log(np.abs(roots_array) + epsilon))
    # real_map = magnitude <= abs(TOL)
    # # If TOL is negative, ensure that at least one root is considered real by relaxing the tolerance.
    # if TOL < 0:
    #     no_real = ~np.any(real_map, axis=0)
    #     for i in np.where(no_real.flatten())[0]:
    #         idx = np.unravel_index(i, no_real.shape)
    #         min_mag = np.min(magnitude[:, idx[0]])
    #         real_map[:, idx[0]] = magnitude[:, idx[0]] <= min_mag * 10

    # # Assign real angles to output
    # response_angles = -np.angle(roots_array)
    # response_angles[response_angles < 0] += 2 * np.pi
    # real_response_angles = np.full_like(response_angles, np.nan)
    # real_response_angles[real_map] = response_angles[real_map]

    # Interpolate to find extrema
    response_deriv2_interpolated = interpolate_fourier_series([0, 2 * np.pi], response_fft_derivative2, real_response_angles)

    angles_maxima = np.full_like(real_response_angles, np.nan)
    angles_minima = np.full_like(real_response_angles, np.nan)

    maxima_map = response_deriv2_interpolated < 0
    minima_map = response_deriv2_interpolated > 0

    # Find maxima and minima
    angles_maxima[maxima_map] = real_response_angles[maxima_map]
    angles_minima[minima_map] = real_response_angles[minima_map]

    # Handle other extrema
    angles_other = np.full_like(real_response_angles, np.nan)
    angles_other[(~maxima_map & ~minima_map) & real_map] = real_response_angles[
        (~maxima_map & ~minima_map) & real_map
    ]

    # Interpolate values at extrema
    maxima_value = interpolate_fourier_series([0, 2 * np.pi], filter_response_fft, angles_maxima)
    minima_value = interpolate_fourier_series([0, 2 * np.pi], filter_response_fft, angles_minima)
    other_value = interpolate_fourier_series([0, 2 * np.pi], filter_response_fft, angles_other)

    # Reshape output arrays
    if dim != 0:
        angles_maxima = np.moveaxis(angles_maxima, 0, dim)
        angles_minima = np.moveaxis(angles_minima, 0, dim)
        maxima_value = np.moveaxis(maxima_value, 0, dim)
        minima_value = np.moveaxis(minima_value, 0, dim)
        angles_other = np.moveaxis(angles_other, 0, dim)
        other_value = np.moveaxis(other_value, 0, dim)

    # Return results in a dictionary
    output_dict = {
        "angles_maxima": angles_maxima,
        "angles_minima": angles_minima,
        "maxima_value": maxima_value,
        "minima_value": minima_value,
        "angles_other": angles_other,
        "other_value": other_value,
    }
    return output_dict