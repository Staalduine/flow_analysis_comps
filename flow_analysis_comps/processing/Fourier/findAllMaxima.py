import numpy as np
from pydantic import BaseModel


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


def find_extrema(
    func_fft: np.ndarray, der1_fft: np.ndarray, der2_fft: np.ndarray
) -> dict:
    """
    Finds extrema using companion matrix approach (n fourier coefficients yields n-1 roots). All input arrays need to be of the same size. Values are sorted along the function values to make significant extrema

    Args:
        func_fft (np.ndarray): Fourier coefficients of original function
        der1_fft (np.ndarray): _description_
        der2_fft (np.ndarray): _description_

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

    return {
        "angles": angles[sort],
        "magnit": magnitudes[sort],
        "origin": origin_values[sort],
        "deriv1": deriv1_values[sort],
        "deriv2": deriv2_values[sort],
    }

class angle_filter_values(BaseModel):
    magnitude: float = 0.1
    first_derivative: float = 0.1
    second_derivative: float = 5.0

def filter_angle_outputs(extrema_results, filter_vals: angle_filter_values):
    # Ensure root is near the unit circle
    low_magnitude = abs(extrema_results["magnit"]) < filter_vals.magnitude
    
    # Ensure root is a root
    low_deriv1 = abs(extrema_results["deriv1"]) < filter_vals.first_derivative
    
    # Ensure root is a sharp extreme
    high_deriv2_max = extrema_results["deriv2"] < - filter_vals.second_derivative
    high_deriv2_min = extrema_results["deriv2"] > filter_vals.second_derivative
    
    # Gather maxima indices
    maxima_index = [
        mag * der1 * der2 for mag, der1, der2 in zip(low_magnitude, low_deriv1, high_deriv2_max)
    ]
    
    # Gather minima indices
    minima_index = [
        mag * der1 * der2 for mag, der1, der2 in zip(low_magnitude, low_deriv1, high_deriv2_min)
    ]

    # Filter angles and original function values
    angles_maxima = extrema_results["angles"][maxima_index]
    angles_minima = extrema_results["angles"][minima_index]
    values_maxima = extrema_results["origin"][maxima_index]
    values_minima = extrema_results["origin"][minima_index]

    return (
        angles_maxima,
        angles_minima,
        values_maxima,
        values_minima,
        maxima_index,
        minima_index,
    )


def preprocess_filter_stack(filter_stack: np.ndarray):
    """
    Take in a filter response stack and convert it to a readily processable stack of Fourier coefficients, along with first and second derivatives

    Args:
        filter_stack (np.ndarray): (D, x,y) array with the real filter response along D axis.

    Returns:
        np.ndarray: 3 flattened arrays with (x*y, D) axes, providing a Fourier series for each pixel in an image. These are the Fourier coefficients of the original function and first and second derivative
    """
    filter_stack /= np.max(filter_stack.flatten())

    response_shape = filter_stack.shape
    response_depth = response_shape[0]
    output_depth = response_depth - 1

    filter_stack_fft = np.fft.fft(filter_stack.real, axis=0)
    # filter_stack_fft = filter_stack

    nyquist = int(np.ceil((len(filter_stack) + 1) / 2))

    # Extend filter stack if depth is even
    if response_depth % 2 == 0:
        filter_response_fft = (
            filter_stack_fft.copy()
        )  # Create a copy to avoid modifying the input array
        filter_response_fft[nyquist - 1] /= 2
        filter_response_fft = np.insert(
            filter_response_fft, nyquist, filter_response_fft[nyquist - 1], axis=0
        )
        filter_stack_fft = filter_response_fft
        output_depth = output_depth + 1

    response_frequencies = np.concatenate(
        [np.arange(nyquist), -np.arange(nyquist - 1, 0, -1)]
    )
    response_frequencies_full = response_frequencies[
        ..., np.newaxis, np.newaxis
    ]  # Add new axis for broadcasting

    response_fft_derivative1 = filter_stack_fft * (1j * response_frequencies_full)
    response_fft_derivative2 = filter_stack_fft * -(response_frequencies_full**2)

    function_flat = np.reshape(
        np.moveaxis(filter_stack_fft, 0, 2), (-1, response_depth)
    )
    derivat1_flat = np.reshape(
        np.moveaxis(response_fft_derivative1, 0, 2), (-1, response_depth)
    )
    derivat2_flat = np.reshape(
        np.moveaxis(response_fft_derivative2, 0, 2), (-1, response_depth)
    )
    return function_flat, derivat1_flat, derivat2_flat


def process_filter_stack(function_flat, derivat1_flat, derivat2_flat, filter_vals):
    output_depth = function_flat.shape[-1] - 1

    output_dict = {
        "maxima": np.full((function_flat.shape[0], output_depth), fill_value=np.nan),
        "minima": np.full((function_flat.shape[0], output_depth), fill_value=np.nan),
        "values_max": np.full(
            (function_flat.shape[0], output_depth), fill_value=np.nan
        ),
        "values_min": np.full(
            (function_flat.shape[0], output_depth), fill_value=np.nan
        ),
    }

    for i, (func, der1, der2) in enumerate(
        zip(function_flat, derivat1_flat, derivat2_flat)
    ):
        extrema_results = find_extrema(func, der1, der2)

        angles_maxima, angles_minima, values_maxima, values_minima, _, _ = (
            filter_angle_outputs(extrema_results, filter_vals)
        )

        output_dict["maxima"][i, : len(angles_maxima)] = angles_maxima.real
        output_dict["minima"][i, : len(angles_minima)] = angles_minima.real
        output_dict["values_max"][i, : len(values_maxima)] = values_maxima.real
        output_dict["values_min"][i, : len(values_minima)] = values_minima.real

    return output_dict
        
def find_maxima(filter_stack: np.ndarray):
    flat_arrays = preprocess_filter_stack(filter_stack)
    
    
    for key, item in output_dict.items():
        output_dict[key] = np.reshape(item.T, (output_depth, *filter_stack.shape[1:]))
    
