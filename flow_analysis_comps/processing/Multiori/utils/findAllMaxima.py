import numba
import numpy as np
from tqdm import tqdm
from flow_analysis_comps.data_structs.multiori_config_struct import angle_filter_values, multiOriOutput
from joblib import Parallel, delayed
from flow_analysis_comps.data_structs.video_metadata_structs import videoInfo
from flow_analysis_comps.processing.Multiori.utils.Interpolation import (
    interpolate_fourier_series,
)

MEMORY_THRESHOLD_NBYTES = 1024 * 1024  # Threshold for low memory processing


def find_all_extrema_in_filter_response(
    filter_stack: np.ndarray,
    filter_vals: angle_filter_values | None = None,
    do_fft: bool = True,
):
    if not filter_vals:
        filter_vals = angle_filter_values()

    # Convert input stack into flat arrays. Each entry is a pixel which has a filter response F(z)
    # Arrays are of frequency spectrum, plus first and second derivatives
    filter_stack_preprocessed = _preprocess_filter_stack(filter_stack, do_fft)

    # Parallel process each pixel response, then filter based on F(z)' and F(z)'' values
    filtered_angles_dict = _process_filter_stack(
        *filter_stack_preprocessed, filter_vals
    )

    filtered_angles_dict = postprocess_angles(filtered_angles_dict)

    # output = multiOriOutput(
    #     metadata=video_metadata,
    #     angles_maxima=filtered_angles_dict["maxima"],
    #     angles_minima=filtered_angles_dict["minima"],
    #     values_maxima=filtered_angles_dict["values_max"],
    #     values_minima=filtered_angles_dict["values_min"],
    # )

    return filtered_angles_dict


def postprocess_angles(output_dict):
    """
    Place angles in expected range [-pi, pi] and put the zero velocity in the center.

    Args:
        output_dict (dict): Dictionary containing angle information.

    Returns:
        dict: Postprocessed dictionary with angles in expected range.
    """
    output_dict["maxima"] = -1 * ((((output_dict["maxima"] + np.pi) / 2) % np.pi ) - np.pi / 2)
    output_dict["minima"] = -1 * ((((output_dict["minima"] + np.pi) / 2) % np.pi ) - np.pi / 2)
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

    return interpolate_fourier_series([0, period], coeffs, xv, method="horner_freq")


def _preprocess_filter_stack(aos_filter_response: np.ndarray, do_fft: bool = True):
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


def _process_filter_stack(
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
    # Technically, what we are doing here is set up a companion matrix for a trigonometric polynomial of order n, where n is the number of Fourier coefficients minus 1.
    # The companion matrix is used to find the roots of the polynomial, which correspond to the extrema of the function.
    # The Frobenius matrix is a square matrix that has ones on the first sub-diagonal and the coefficients of the polynomial on the last row.
    # In theory, the same output can be achieved with np.roots, which might be more memory efficient. I have not gotten an equivalent output with np.roots yet, so I am sticking with this for now.

    print(
        f"Calculating eigenvalues, using {'low memory' if deriv1_fft_flat.nbytes > MEMORY_THRESHOLD_NBYTES else 'full memory'}"
    )
    if deriv1_fft_flat.nbytes > MEMORY_THRESHOLD_NBYTES:
        # If the input is large, use a low memory version
        eigenvalues = calculate_eigenvalues_from_stack_low_mem(deriv1_fft_flat)
    else:
        eigenvalues = calculate_eigenvalues_from_stack_high_mem(deriv1_fft_flat)

    # Calculate angles and magnitudes of eigenvalues
    # The real eigenvalues are the arguments of the complex roots of the companion matrix. For an adjustment, we also use the complex logarithm.
    # Real angles are only supposed to be on the unit circle, which means that their magnitudes should be close to 1.
    # The magnitudes are calculated with the logarithm of the absolute value of the eigenvalues, such that the magnitudes are close to 0 for angles on the unit circle.
    angles = (np.angle(eigenvalues) - np.log(np.abs(eigenvalues)) * 1j).real % (
        2 * np.pi
    )  # - np.pi
    magnitudes = np.abs(np.log(np.abs(eigenvalues)))

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


def calculate_eigenvalues_from_stack_high_mem(deriv1_fft_flat: np.ndarray):
    """
    Create Frobenius matrices from the first derivative Fourier coefficients and calculate eigenvalues.
    This function creates a large array for the entire image, so it can be memory intensive. It is fast tho.

    Args:
        deriv1_fft_flat (np.ndarray): First derivative Fourier coefficients.

    Returns:
        np.ndarray: Eigenvalues for each sequence of Fourier coefficients.
    """
    output_depth = deriv1_fft_flat.shape[1] - 1
    big_frob = np.zeros(
        (deriv1_fft_flat.shape[0], output_depth, output_depth), dtype=np.complex128
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

    return eigenvalues


def calculate_eigenvalues_from_stack_low_mem(deriv1_fft_flat: np.ndarray):
    """
    Calculate eigenvalues from the first derivative Fourier coefficients without creating a large array.
    This function is more memory efficient but slower than the full version.

    Args:
        deriv1_fft_flat (np.ndarray): First derivative Fourier coefficients.

    Returns:
        np.ndarray: Eigenvalues for each sequence of Fourier coefficients.
    """
    eigenvalues = np.array(
        Parallel(n_jobs=-1)(
            delayed(eigenvalues_single_sequence)(deriv1_fft_flat[i])
            for i in tqdm(range(deriv1_fft_flat.shape[0]))
        )
    )
    return eigenvalues


def eigenvalues_single_sequence(deriv1_fft_flat: np.ndarray) -> np.ndarray:
    """
    Calculates the eigenvalues for a single sequence of Fourier coefficients using a Frobenius matrix.
    Expected input is a single row of the first derivative Fourier coefficients.

    Args:
        deriv1_fft_flat (np.ndarray): First derivative Fourier coefficients. Direct result of fft.

    Returns:
        np.ndarray: Eigenvalues for the given sequence of Fourier coefficients.
    """
    output_depth = deriv1_fft_flat.shape[0] - 1
    deriv1_fftshift = np.fft.fftshift(deriv1_fft_flat)
    frob_matrix = np.zeros((output_depth, output_depth), dtype=np.complex128)
    frob_matrix[:-1, 1:] = np.eye(output_depth - 1)
    frob_matrix[-1, :] = deriv1_fftshift[:-1] / deriv1_fftshift[-1]
    eigvals = np.linalg.eigvals(frob_matrix)
    return eigvals


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds extrema using companion matrix approach (n fourier coefficients yields n-1 roots). All input arrays need to be of the same size. Values are sorted along the function values to make significant extrema

    Args:
        func_fft (np.ndarray): Fourier coefficients of original function
        der1_fft (np.ndarray): Fourier coefficients of first derivative function
        der2_fft (np.ndarray): Fourier coefficients of second function

    Returns:
        np.ndarray: angles of the function extrema, along with magnitude, sampled function, derivative and second derivative values
    """
    origin_values = get_value_from_fft_series(func_fft, angles, 2 * np.pi).real
    deriv1_values = get_value_from_fft_series(der1_fft, angles, 2 * np.pi).real
    deriv2_values = get_value_from_fft_series(der2_fft, angles, 2 * np.pi).real

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

    return fast_filter_angle_outputs(
        angles,
        magnitudes,
        origin_vals,
        deriv1_vals,
        deriv2_vals,
        filter_vals.magnitude,
        filter_vals.first_derivative,
        filter_vals.second_derivative,
    )


@numba.njit
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
