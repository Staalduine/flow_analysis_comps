import numpy as np

from flow_analysis_comps.util.coord_space_util import wraparoundN

from flow_analysis_comps.processing.Fourier.utils.Interpolation import (
    interpolate_fourier_series,
)


def halleyft(
    fourier_coeffs: np.ndarray, # [K, M]
    initial_guesses: np.ndarray, # [N, M]
    is_frequency_domain: bool = False,
    derivative_order: int = 0,
    tolerance: float = 1e-12,
    max_iterations: int = 10,
    avoid_nan: bool | None = None,
    unique_zeros: bool | float = False,
    *args,
):
    """
    Refine roots of a Fourier series using Halley's method.
    Parameters:
        fourier_coeffs: ndarray, Fourier series values or its FFT (shape: [N, M])
        initial_guesses: ndarray, initial guesses for roots (shape: [G, M])
        is_frequency_domain: bool, if True, fourier_coeffs is already in frequency domain
        derivative_order: int, which derivative to use (default 0)
        tolerance: float, tolerance for convergence
        max_iterations: int, maximum number of iterations
        avoid_nan: bool, avoid processing NaN guesses (default: not isvector(initial_guesses))
        unique_zeros: bool or float, return unique zeros (default False)
    Returns:
        refined_zeros: ndarray, refined zeros (shape: initial_guesses.shape)
        refined_derivatives: ndarray, derivatives at zeros (shape: initial_guesses.shape + (3,))
        iterations_per_guess: ndarray, iterations per guess (shape: initial_guesses.shape)
        total_iterations: int, total number of iterations
    """
    if avoid_nan is None:
        avoid_nan = initial_guesses.ndim > 1 and initial_guesses.shape[0] > 1

    derivative_indices = np.array([0, 1, 2]) + derivative_order

    if not is_frequency_domain:
        fourier_coeffs_freq = np.fft.fft(fourier_coeffs, axis=0)
    else:
        fourier_coeffs_freq = fourier_coeffs

    if avoid_nan:
        guess_shape = initial_guesses.shape
        guesses_flat = initial_guesses.reshape(initial_guesses.shape[0], -1)
        is_nan_guess = np.isnan(guesses_flat)
        sort_indices = np.argsort(is_nan_guess, axis=0)
        guesses_sorted = np.take_along_axis(guesses_flat, sort_indices, axis=0)
        valid_guess_count = initial_guesses.shape[0] - np.sum(is_nan_guess, axis=0)
        refined_zeros = np.full_like(guesses_flat, np.nan)
        refined_derivatives = np.full(guesses_flat.shape + (3,), np.nan)
        for num_valid in range(1, initial_guesses.shape[0] + 1):
            valid_columns = valid_guess_count == num_valid
            if np.any(valid_columns):
                res = halleyft(
                    fourier_coeffs_freq[:, valid_columns],
                    guesses_sorted[:num_valid, valid_columns],
                    True,
                    derivative_order,
                    tolerance,
                    max_iterations,
                    False,
                    unique_zeros,
                )
                refined_zeros[:num_valid, valid_columns] = res[0]
                refined_derivatives[:num_valid, valid_columns, :] = res[1]
        unsort_indices = np.argsort(sort_indices, axis=0)
        for col in range(refined_zeros.shape[1]):
            refined_zeros[:, col] = refined_zeros[unsort_indices[:, col], col]
            refined_derivatives[:, col, :] = refined_derivatives[
                unsort_indices[:, col], col, :
            ]
        refined_zeros = refined_zeros.reshape(guess_shape)
        refined_derivatives = refined_derivatives.reshape(guess_shape + (3,))
        return refined_zeros, refined_derivatives

    num_coeffs = int(np.floor(fourier_coeffs.shape[0] / 2))
    freq_multipliers = np.fft.ifftshift(np.arange(-num_coeffs, num_coeffs + 1)) * 1j
    freq_multipliers_stack = np.stack(
        [freq_multipliers**d for d in derivative_indices], axis=-1
    )
    fourier_coeffs_derivs = (
        fourier_coeffs_freq[..., None] * freq_multipliers_stack
    )  # shape: (N, M, 3)

    guess_shape = initial_guesses.shape
    guesses = initial_guesses.reshape(initial_guesses.shape[0], -1)
    refined_derivatives = np.full(guesses.shape + (3,), np.nan)
    iterations_per_guess = np.zeros(guesses.shape, dtype=np.uint8)
    refined_zeros = np.copy(guesses)
    total_iterations = 0

    guess_values = interpolate_fourier_series(
        [0, 2 * np.pi],
        fourier_coeffs_derivs,
        np.tile(guesses[..., None], (1, 1, 3)),
        "horner_freq",
    )
    columns_not_converged = np.ones(guesses.shape[1], dtype=bool)
    new_guess_is_better = np.ones(guesses.shape, dtype=bool)

    while total_iterations == 0 or np.any(columns_not_converged):
        func_vals = guess_values[:, :, 0]
        first_deriv_vals = guess_values[:, :, 1]
        second_deriv_vals = guess_values[:, :, 2]
        updated_guesses = guesses[:, columns_not_converged] - 2 * func_vals[
            :, columns_not_converged
        ] * first_deriv_vals[:, columns_not_converged] / (
            2 * first_deriv_vals[:, columns_not_converged] ** 2
            - func_vals[:, columns_not_converged]
            * second_deriv_vals[:, columns_not_converged]
        )
        updated_guesses = wraparoundN(updated_guesses, 0, 2 * np.pi)
        updated_guess_values = interpolate_fourier_series(
            [0, 2 * np.pi],
            fourier_coeffs_derivs[:, columns_not_converged, :],
            np.tile(updated_guesses[..., None], (1, 1, 3)),
            "horner_freq",
        )
        is_better_now = np.abs(updated_guess_values[:, :, 0]) < np.abs(
            guess_values[:, :, 0]
        )
        new_guess_is_better[:, columns_not_converged] = is_better_now
        guesses[:, columns_not_converged][is_better_now] = updated_guesses[
            is_better_now
        ]
        refined_derivatives[:, columns_not_converged, :][is_better_now] = (
            updated_guess_values[is_better_now]
        )
        zero_values = np.minimum(
            np.abs(guess_values[:, :, 0]), np.abs(updated_guess_values[:, :, 0])
        )
        not_converged = zero_values > tolerance

        total_iterations += 1
        if total_iterations > max_iterations:
            temp_guesses = guesses[:, columns_not_converged]
            temp_guesses[not_converged] = np.nan
            guesses[:, columns_not_converged] = temp_guesses
            temp_derivs = refined_derivatives[:, columns_not_converged, :]
            temp_derivs[not_converged[..., None].repeat(3, axis=2)] = np.nan
            refined_derivatives[:, columns_not_converged, :] = temp_derivs
            iterations_per_guess[:, columns_not_converged] = np.inf
            break
        else:
            temp_guesses = guesses[:, columns_not_converged]
            invalid = ~is_better_now & not_converged
            temp_guesses[invalid] = np.nan
            guesses[:, columns_not_converged] = temp_guesses
            temp_derivs = refined_derivatives[:, columns_not_converged, :]
            temp_derivs[invalid[..., None].repeat(3, axis=2)] = np.nan
            refined_derivatives[:, columns_not_converged, :] = temp_derivs
            iterations_per_guess[:, columns_not_converged] = total_iterations
            is_better_now = is_better_now & not_converged
            new_not_converged = np.any(is_better_now, axis=0)
            guess_values = updated_guess_values[:, new_not_converged, :]
            columns_not_converged[columns_not_converged] = new_not_converged

    refined_zeros = guesses.reshape(guess_shape)
    refined_derivatives = refined_derivatives.reshape(guess_shape + (3,))
    iterations_per_guess = iterations_per_guess.reshape(guess_shape)

    if unique_zeros and refined_zeros.shape[0] > 1:
        sorted_zeros = np.sort(refined_zeros, axis=0)
        sorted_zeros_diff = np.diff(sorted_zeros, axis=0)
        sorted_zeros_diff = np.concatenate(
            [
                (sorted_zeros[0:1, :] + 2 * np.pi - np.max(sorted_zeros, axis=0)),
                sorted_zeros_diff,
            ],
            axis=0,
        )
        uniq_tol = 1e-6 if isinstance(unique_zeros, bool) else unique_zeros
        sorted_zeros[sorted_zeros_diff < uniq_tol] = np.nan
        sort_indices = np.argsort(refined_zeros, axis=0)
        for col in range(refined_zeros.shape[1]):
            refined_zeros[:, col] = sorted_zeros[sort_indices[:, col], col]
        isnan_refined = np.isnan(refined_zeros)
        for i in range(3):
            temp = refined_derivatives[..., i]
            temp[isnan_refined] = np.nan
            refined_derivatives[..., i] = temp
        iterations_per_guess[isnan_refined] = np.nan

    return refined_zeros, refined_derivatives
