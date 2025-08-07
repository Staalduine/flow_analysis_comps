import numba
import numpy as np
from flow_analysis_comps.data_structs.AOS_structs import angle_filter_values
from joblib import Parallel, delayed
from flow_analysis_comps.processing.Fourier.utils.Interpolation import interpolate_fourier_series


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

    filtered_angles_dict = postprocess_angles(filtered_angles_dict)
    
    return filtered_angles_dict

def postprocess_angles(output_dict):
    output_dict["maxima"] = (output_dict["maxima"]  + np.pi) / 2
    output_dict["minima"] = (output_dict["minima"]  + np.pi) / 2
    return output_dict

def get_value_from_fft_series(coeffs: np.ndarray, xv: np.ndarray, period: float):
    """
    Sample trigonometric polynomial using the complex fourier series coefficients.
    Uses Horner method for real-valued query points

    Args:
        coeffs (np.ndarray): Series coefficients
        xv (np.ndarray): Points on which to sample
        period (float): Period of the function

    Returns:
        np.ndarray: Sampled points on trigonometric polynomial
    """

    # # Normalize points to a [-1, 1] domain
    # xv = xv / period
    # size = len(coeffs)

    # # adjust the spatial frequencies to period and number of coefficients
    # kn = np.fft.fftfreq(size, period / size) * period

    # # Sample contributions from each exponential
    # eikx = np.exp(2.0j * np.pi * np.outer(xv, kn))

    # # Sum and return
    # return np.einsum("ab,b->a", eikx, coeffs) / size

    return interpolate_fourier_series([0, period], coeffs, xv, method="horner_freq")


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

    # Pre-make Frobenius matrices
    big_frob = np.zeros(
        (function_fft_flat.shape[0], output_depth, output_depth), dtype=np.complex128
    )
    big_frob[:, :-1, 1:] = np.eye(output_depth - 1)
    deriv1_fft_flat_fftshift = np.fft.fftshift(deriv1_fft_flat, axes=1)
    coeff_array = (
        deriv1_fft_flat_fftshift[:, :-1]
        / (deriv1_fft_flat_fftshift[:, -1])[..., np.newaxis]
    )
    big_frob[:, -1] = coeff_array

    eigenvalues = np.array(
        Parallel(n_jobs=-1)(delayed(np.linalg.eigvals)(frob) for frob in big_frob)
    )
    angles = (np.angle(eigenvalues) - np.log(abs(eigenvalues)) * 1j).real % (
        2 * np.pi
    ) - np.pi
    magnitudes = np.abs(np.log(abs(eigenvalues)))

    full_properties_array = np.array(
        Parallel(n_jobs=-1)(
            delayed(sample_and_filter_angles)(
                multi_ori_filter_params, angle, mag, func, der1, der2
            )
            for angle, mag, func, der1, der2 in zip(
                angles, magnitudes, function_fft_flat, deriv1_fft_flat, deriv2_fft_flat
            )
        )
    )

    output_dict = {}
    output_dict["maxima"] = full_properties_array[:, 0]
    output_dict["minima"] = full_properties_array[:, 1]
    output_dict["values_max"] = full_properties_array[:, 2]
    output_dict["values_min"] = full_properties_array[:, 3]

    for key, item in output_dict.items():
        output_dict[key] = np.reshape(item.T, (output_depth, *img_size))

    return output_dict


def sample_and_filter_angles(multi_ori_filter_params, angle, mag, func, der1, der2):
    extrema_results = sample_function_deriv1_deriv2_values(angle, mag, func, der1, der2)
    filtered_outputs = filter_angle_outputs(extrema_results, multi_ori_filter_params)
    return filtered_outputs


def sample_function_deriv1_deriv2_values(
    angles: np.ndarray,
    magnitudes: np.ndarray,
    func_fft: np.ndarray,
    der1_fft: np.ndarray,
    der2_fft: np.ndarray,
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
    origin_values = get_value_from_fft_series(func_fft, angles + np.pi, 2 * np.pi).real
    deriv1_values = get_value_from_fft_series(der1_fft, angles + np.pi, 2 * np.pi).real
    deriv2_values = get_value_from_fft_series(der2_fft, angles + np.pi, 2 * np.pi).real

    sort = np.argsort(origin_values)[::-1]

    return (
        angles[sort],
        magnitudes[sort],
        origin_values[sort],
        deriv1_values[sort],
        deriv2_values[sort],
    )


def filter_angle_outputs(extrema_results, filter_vals: angle_filter_values):
    # angles, magnitudes, origin_vals, deriv1_vals, deriv2_vals = extrema_results

    return fast_filter_angle_outputs(
        *extrema_results,
        filter_vals.magnitude,
        filter_vals.first_derivative,
        filter_vals.second_derivative,
    )


@numba.jit
def fast_filter_angle_outputs(
    angles,
    magnitudes,
    origin_vals,
    deriv1_vals,
    deriv2_vals,
    mag_constr: float,
    df1_constr: float,
    df2_constr: float,
):
    # 6 entries for 6 results:

    """
    angles_maxima,
    angles_minima,
    values_maxima,
    values_minima,
    maxima_index,
    minima_index,
    """
    output_array = np.full((6, len(angles)), fill_value=np.nan, dtype=np.float32)

    # Ensure root is near the unit circle
    low_magnitude = np.abs(magnitudes) < mag_constr

    # Ensure root is a root
    low_deriv1 = np.abs(deriv1_vals) < df1_constr

    # Ensure root is a sharp extreme
    high_deriv2_max = deriv2_vals < -df2_constr
    high_deriv2_min = deriv2_vals > df2_constr

    # Gather maxima indices
    maxima_index = low_magnitude * low_deriv1 * high_deriv2_max

    # Gather minima indices
    minima_index = low_magnitude * low_deriv1 * high_deriv2_min

    # Filter angles and original function values
    output_array[0, : np.sum(maxima_index)] = angles[maxima_index]
    output_array[1, : np.sum(minima_index)] = angles[minima_index]
    output_array[2, : np.sum(maxima_index)] = origin_vals[maxima_index]
    output_array[3, : np.sum(minima_index)] = origin_vals[minima_index]
    output_array[4] = maxima_index
    output_array[5] = minima_index

    return output_array
