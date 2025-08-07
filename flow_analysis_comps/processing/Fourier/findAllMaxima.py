import numpy as np
from flow_analysis_comps.data_structs.AOS_structs import angle_filter_values


def find_all_extrema_in_filter_response(
    filter_stack: np.ndarray,
    filter_vals: angle_filter_values | None = None,
    do_fft: bool = True,
):
    if not filter_vals:
        filter_vals = angle_filter_values()

    # Convert input stack into flat arrays. Each entry is a pixel which has a filter response F(z)
    # Arrays are of frequency spectrum, plus first and second derivatives
    flat_arrays = preprocess_filter_stack(filter_stack, do_fft)

    # Parallel process each pixel response, then filter based on F(z)' and F(z)'' values
    filtered_angles_dict = process_filter_stack(*flat_arrays, filter_vals)
    # output_dict = reshape_flat_stack_to_image(filter_stack, filtered_angles_dict)
    return filtered_angles_dict


def get_value_from_fft_series(coeffs: np.ndarray, xv: np.ndarray, period: float):
    """
    Sample trigonometric polynomial using the complex fourier series coefficients

    Args:
        coeffs (np.ndarray): Series coefficients
        xv (np.ndarray): Points on which to sample
        period (float): Period of the function

    Returns:
        np.ndarray: Sampled points on trigonometric polynomial
    """

    # Normalize points to a [-1, 1] domain
    xv = xv / period
    size = len(coeffs)

    # adjust the spatial frequencies to period and number of coefficients
    kn = np.fft.fftfreq(size, period / size) * period

    # Sample contributions from each exponential
    eikx = np.exp(2.0j * np.pi * np.outer(xv, kn))

    # Sum and return
    return np.einsum("ab,b->a", eikx, coeffs) / size


def preprocess_filter_stack(aos_filter_response: np.ndarray, do_fft: bool = True):
    """
    Take in a filter response stack in real-space, and return its Fourier-transform, as well as its first and second derivative.

    Args:
        filter_stack (np.ndarray): (D, x,y) array with the real filter response along D axis.

    Returns:
        np.ndarray: 3 flattened arrays with (x*y, D) axes, providing a Fourier series for each pixel in an image. These are the Fourier coefficients of the original function and first and second derivative
    """

    # Normalize all values to range [0, 1]
    aos_filter_response /= np.max(aos_filter_response.flatten())

    response_shape = aos_filter_response.shape
    response_depth = response_shape[0]
    # output_depth = response_depth - 1

    # Calculate fft along filter axis
    if do_fft:
        aos_filter_response_fft = np.fft.fft(aos_filter_response.real, axis=0)
    else:
        aos_filter_response_fft = aos_filter_response

    # Get nyquist rate
    nyquist = int(np.ceil((len(aos_filter_response) + 1) / 2))

    # Extend filter stack if depth is even
    if response_depth % 2 == 0:
        filter_response_fft = (
            aos_filter_response_fft.copy()
        )  # Create a copy to avoid modifying the input array
        filter_response_fft[nyquist - 1] /= 2
        filter_response_fft = np.insert(
            filter_response_fft, nyquist, filter_response_fft[nyquist - 1], axis=0
        )
        aos_filter_response_fft = filter_response_fft
        # output_depth = output_depth + 1

    # Find values of frequency axis
    frequency_axis = np.concatenate(
        [np.arange(nyquist), -np.arange(nyquist - 1, 0, -1)]
    )
    frequency_axis_broadcastable = frequency_axis[
        ..., np.newaxis, np.newaxis
    ]  # Add new axis for broadcasting

    # Calculate first and second derivative
    aos_filter_response_deriv1_fft = aos_filter_response_fft * (
        1j * frequency_axis_broadcastable
    )
    aos_filter_response_deriv2_fft = aos_filter_response_fft * -(
        frequency_axis_broadcastable**2
    )

    return (
        aos_filter_response_fft,
        aos_filter_response_deriv1_fft,
        aos_filter_response_deriv2_fft,
    )


def process_filter_stack(
    filter_response_fft: np.ndarray,
    filter_response_deriv1_fft: np.ndarray,
    filter_response_deriv2_fft: np.ndarray,
    multi_ori_filter_params: angle_filter_values,
):
    # get values for reshapes
    response_depth = filter_response_fft.shape[0]
    img_size = filter_response_fft.shape[1:]

    # Ensure 2D case for now
    assert len(img_size) == 2

    output_depth = response_depth - 1

    # Reshape (D, x, y, (z)) arrays into (x*y*(z), D) arrays
    function_fft_flat = np.reshape(
        np.moveaxis(filter_response_fft, 0, 2), (-1, response_depth)
    )
    deriv1_fft_flat = np.reshape(
        np.moveaxis(filter_response_deriv1_fft, 0, 2), (-1, response_depth)
    )
    deriv2_fft_flat = np.reshape(
        np.moveaxis(filter_response_deriv2_fft, 0, 2), (-1, response_depth)
    )

    # Pre-allocate output arrays
    output_dict = {
        "maxima": np.full(
            (function_fft_flat.shape[0], output_depth), fill_value=np.nan
        ),
        "minima": np.full(
            (function_fft_flat.shape[0], output_depth), fill_value=np.nan
        ),
        "values_max": np.full(
            (function_fft_flat.shape[0], output_depth), fill_value=np.nan
        ),
        "values_min": np.full(
            (function_fft_flat.shape[0], output_depth), fill_value=np.nan
        ),
    }

    for i, (func, der1, der2) in enumerate(
        zip(function_fft_flat, deriv1_fft_flat, deriv2_fft_flat)
    ):
        extrema_results = find_extrema_with_function_deriv1_deriv2(func, der1, der2)

        angles_maxima, angles_minima, values_maxima, values_minima, _, _ = (
            filter_angle_outputs(extrema_results, multi_ori_filter_params)
        )

        output_dict["maxima"][i, : len(angles_maxima)] = angles_maxima.real
        output_dict["minima"][i, : len(angles_minima)] = angles_minima.real
        output_dict["values_max"][i, : len(values_maxima)] = values_maxima.real
        output_dict["values_min"][i, : len(values_minima)] = values_minima.real

    for key, item in output_dict.items():
        output_dict[key] = np.reshape(item.T, (output_depth, *img_size))

    return output_dict


def find_extrema_with_function_deriv1_deriv2(
    func_fft: np.ndarray, der1_fft: np.ndarray, der2_fft: np.ndarray
) -> dict:
    """
    Finds extrema using companion matrix approach (n fourier coefficients yields n-1 roots). All input arrays need to be of the same size. Values are sorted along the function values to make significant extrema

    Args:
        func_fft (np.ndarray): Fourier coefficients of original function
        der1_fft (np.ndarray): Fourier coefficients of first derivative function
        der2_fft (np.ndarray): Fourier coefficients of second function

    Returns:
        np.ndarray: angles of the function extrema, along with magnitude, sampled function, derivative and second derivative values
    """
    output_depth = len(func_fft) - 1
    coeffs = -np.fft.fftshift(der1_fft)
    # Assemble Matrix
    mat = np.zeros((output_depth, output_depth), dtype=np.complex128)
    mat[:-1, 1:] = np.eye(output_depth - 1)
    mat[-1] = coeffs[:-1] / coeffs[-1]

    eigenvalues = np.linalg.eigvals(mat)
    angles = (np.angle(eigenvalues) - np.log(abs(eigenvalues)) * 1j).real % (
        2 * np.pi
    ) - np.pi

    magnitudes = np.log(abs(eigenvalues))
    origin_values = get_value_from_fft_series(func_fft, angles + np.pi, 2 * np.pi)
    deriv1_values = get_value_from_fft_series(der1_fft, angles + np.pi, 2 * np.pi)
    deriv2_values = get_value_from_fft_series(der2_fft, angles + np.pi, 2 * np.pi)

    sort = np.argsort(origin_values)[::-1]

    return (
        angles[sort],
        magnitudes[sort],
        origin_values[sort],
        deriv1_values[sort],
        deriv2_values[sort],
    )


def filter_angle_outputs(extrema_results, filter_vals: angle_filter_values):
    angles, magnitudes, origin_vals, deriv1_vals, deriv2_vals = extrema_results

    # Ensure root is near the unit circle
    low_magnitude = abs(magnitudes) < filter_vals.magnitude

    # Ensure root is a root
    low_deriv1 = abs(deriv1_vals) < filter_vals.first_derivative

    # Ensure root is a sharp extreme
    high_deriv2_max = deriv2_vals < -filter_vals.second_derivative
    high_deriv2_min = deriv2_vals > filter_vals.second_derivative

    # Gather maxima indices
    maxima_index = [
        mag * der1 * der2
        for mag, der1, der2 in zip(low_magnitude, low_deriv1, high_deriv2_max)
    ]

    # Gather minima indices
    minima_index = [
        mag * der1 * der2
        for mag, der1, der2 in zip(low_magnitude, low_deriv1, high_deriv2_min)
    ]

    # Filter angles and original function values
    angles_maxima = angles[maxima_index]
    angles_minima = angles[minima_index]
    values_maxima = origin_vals[maxima_index]
    values_minima = origin_vals[minima_index]

    return (
        angles_maxima,
        angles_minima,
        values_maxima,
        values_minima,
        maxima_index,
        minima_index,
    )


# def reshape_flat_stack_to_image(filter_stack, filtered_angles_dict):
#     for key, item in filtered_angles_dict.items():
#         filtered_angles_dict[key] = np.reshape(
#             item.T, (filtered_angles_dict, *filter_stack.shape[1:])
#         )
#     return filtered_angles_dict
