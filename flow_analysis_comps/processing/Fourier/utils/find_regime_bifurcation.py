import numpy as np
from flow_analysis_comps.processing.Fourier.utils.halleyft import halleyft
from flow_analysis_comps.processing.Fourier.findAllMaxima import interpft_extrema_fast


def find_regime_bifurcation(
    response_fft: np.ndarray,
    K_response,
    K_high,
    K_low=None,
    maxima=[],
    minima=[],
    maxIterations=10,
    tolerance=None,
    freq=False,
    debug=False,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isscalar(K_response):
        K_response = np.full((response_fft.shape[1],), K_response)

    if K_low is None or len(np.atleast_1d(K_low)) == 0:
        K_low = np.ones(response_fft.shape[1])

    useMinima = (
        False
        if minima is None or (isinstance(minima, list) and len(minima) == 0)
        else True
    )

    if np.isscalar(K_high):
        K_high = np.full((response_fft.shape[1],), K_high)
    else:
        K_high = np.atleast_1d(K_high)
    # Make K_low match 2nd dimension of response
    K_low = np.atleast_1d(K_low)
    if K_low.shape[0] == 1:
        K_low = np.full((response_fft.shape[1],), K_low[0])

    if useMinima:
        extrema_working = np.sort(np.concatenate([maxima, minima], axis=0), axis=0)
        # Pad extrema so that it matches first dimension of response
        if extrema_working.shape[0] < response_fft.shape[0]:
            pad_rows = response_fft.shape[0] - extrema_working.shape[0]
            pad = np.full((pad_rows, extrema_working.shape[1]), np.nan)
            extrema_working = np.vstack([extrema_working, pad])
    else:
        extrema_working = maxima

    # Initialize output to input, should be the same size
    extrema_high = np.copy(extrema_working)

    # Count non-NaN extrema in each column
    nExtrema_working = np.sum(~np.isnan(extrema_working), axis=0)
    # If there are one or less extrema we are done
    not_done = nExtrema_working > 1

    # The working variables represent the data we are not done with
    K_high_working = K_high[not_done]
    K_low_working = K_low[not_done]
    K_response_working = K_response[not_done]
    extrema_working = extrema_working[:, not_done]
    response_working_hat = response_fft[:, not_done]

    for i in range(maxIterations):
        # Delegate the real work to a stripped down function
        K_high_working, K_low_working, extrema_working = find_regime_bifurcation_hat(
            response_working_hat,
            K_response_working,
            K_high_working,
            K_low_working,
            extrema_working,
            useMinima,
        )

        # Update the output variables with the data we are not done with
        K_high[not_done] = K_high_working
        K_low[not_done] = K_low_working
        extrema_high[:, not_done] = extrema_working

        # We are not done if the difference in K exceeds the tolerance
        not_done_working = (K_high_working - K_low_working) > tolerance
        not_done_indices = np.where(not_done)[0]
        not_done[not_done_indices] = not_done_working

        # Update the working variables
        K_response_working = K_response_working[not_done_working]
        K_high_working = K_high_working[not_done_working]
        K_low_working = K_low_working[not_done_working]
        response_working_hat = response_working_hat[:, not_done_working]
        extrema_working = extrema_working[:, not_done_working]

        # if debug:
        #     print("K_high[debugIdx]:", K_high[debugIdx])
        #     print("K_low[debugIdx]:", K_low[debugIdx])
        #     print("extrema_high[:, debugIdx]:", extrema_high[:, debugIdx])

        if K_high_working.size == 0:
            if debug:
                print("Iteration:", i)
            break
        elif debug:
            print("length(K_high_working):", len(K_high_working))

    # Optionally return additional outputs if needed
    # if nargout > 3 in MATLAB, in Python just always compute and return
    # You must implement get_response_at_order_vec_hat and interpft_extrema
    response_low_hat = get_response_at_order_vec_hat(response_fft, K_response, K_low)
    maxima_low, minima_low = interpft_extrema_fast(response_low_hat, 1, False)
    extrema_low = np.sort(np.concatenate([maxima_low, minima_low], axis=0), axis=0)

    return K_high, K_low


def find_regime_bifurcation_hat(
    response_hat, K_response, K_high, K_low, extrema, useMinima
):
    """
    Python translation of the first half of findRegimeBifurcationHat.
    """
    nExtrema = extrema.shape[0] - np.sum(np.isnan(extrema), axis=0)
    K_midpoint = (K_high + K_low) / 2

    # You must implement get_response_at_order_vec_hat in Python
    response_midpoint_hat = get_response_at_order_vec_hat(
        response_hat, K_response, K_midpoint
    )

    # You must implement halleyft in Python
    extrema_midpoint, xdg = halleyft(
        response_midpoint_hat, extrema, True, 1, 1e-12, 10, True, 1e-4
    )
    # [maxima_midpoint, xdg] = halleyft_parallel_by_guess(...) # not used here

    # Only keep maxima if not using minima
    if not useMinima:
        extrema_midpoint[xdg[:, :, 2] > 0] = np.nan

    # Eliminate duplicates (should be done by halleyft with uniq=True)
    extrema_midpoint = np.sort(extrema_midpoint, axis=0)
    max_extrema_midpoint = np.max(extrema_midpoint, axis=0) - 2 * np.pi
    # Remove duplicates: difference less than 1e-4
    diff_extrema = np.diff(np.vstack([max_extrema_midpoint, extrema_midpoint]), axis=0)
    mask = np.abs(diff_extrema) < 1e-4
    extrema_midpoint[1:][mask] = np.nan
    extrema_midpoint = np.sort(extrema_midpoint, axis=0)

    nExtremaMidpoint = extrema_midpoint.shape[0] - np.sum(
        np.isnan(extrema_midpoint), axis=0
    )

    # Do error correction
    if useMinima:
        # Maxima and minima should occur in pairs.
        # An odd number of extrema would indicate an error
        oddIdx = (nExtremaMidpoint % 2) == 1

        # Find extrema that are close together, which may indicate an error
        max_extrema = np.max(extrema_midpoint, axis=0) - 2 * np.pi
        closeExtremaIdx = np.any(
            np.diff(np.vstack([max_extrema, extrema_midpoint]), axis=0) < 0.02, axis=0
        )

        # Mark columns with close extrema as odd
        oddIdx = np.logical_or(oddIdx, closeExtremaIdx)

        if np.any(oddIdx):
            # You must implement interpft_extrema_fast or similar
            odd_maxima, odd_minima = interpft_extrema_fast(
                response_midpoint_hat[:, oddIdx], 1, True
            )
            oddExtrema = np.sort(
                np.concatenate([odd_maxima, odd_minima], axis=0), axis=0
            )
            # Truncate or pad oddExtrema to match extrema_midpoint rows
            sExtrema = oddExtrema.shape[0]
            max_rows = extrema_midpoint.shape[0]
            if sExtrema > max_rows:
                oddExtrema = oddExtrema[:max_rows, :]
            elif sExtrema < max_rows:
                pad = np.full((max_rows - sExtrema, oddExtrema.shape[1]), np.nan)
                oddExtrema = np.vstack([oddExtrema, pad])
            extrema_midpoint[:, oddIdx] = oddExtrema
            nExtremaMidpoint[oddIdx] = extrema_midpoint.shape[0] - np.sum(
                np.isnan(extrema_midpoint[:, oddIdx]), axis=0
            )

    if useMinima:
        bifurcationInHigh = (nExtrema - nExtremaMidpoint) >= 2
    else:
        bifurcationInHigh = nExtremaMidpoint != nExtrema

    bifurcationInLow = ~bifurcationInHigh

    K_low[bifurcationInHigh] = K_midpoint[bifurcationInHigh]
    K_high[bifurcationInLow] = K_midpoint[bifurcationInLow]

    extrema_high = np.copy(extrema)
    # Update columns where bifurcation is in low with sorted extrema_midpoint
    if np.any(bifurcationInLow):
        extrema_high[:, bifurcationInLow] = np.sort(
            extrema_midpoint[:, bifurcationInLow], axis=0
        )

    return K_high, K_low, extrema_high


def get_response_at_order_vec_hat(response_fft: np.ndarray, K_original: np.ndarray | float, K_target: np.ndarray):
    """
    Python translation of getResponseAtOrderVecHat from MATLAB.
    response_fft: (N, M) array (N: frequency, M: columns)
    K_original: scalar or array of length M
    K_target: scalar or array of length M
    Returns: response_at_order (N, len(K_target)) array
    """
    import numpy as np

    if K_target is None or (isinstance(K_target, (list, np.ndarray)) and len(np.atleast_1d(K_target)) == 0):
        return np.full((response_fft.shape[0], np.size(K_target)), np.nan)

    half_length = int(np.floor(response_fft.shape[0] / 2))
    freq_indices = np.concatenate([np.arange(0, half_length + 1), np.arange(-half_length, 0)])

    K_original = np.atleast_1d(K_original)
    K_target = np.atleast_1d(K_target)
    n_original = 2 * K_original + 1
    n_target = 2 * K_target + 1

    # Calculate scaling factors for the Gaussian filter
    scaling_factor_inv = np.sqrt(n_original ** 2 * n_target ** 2 / (n_original ** 2 - n_target ** 2))
    scaling_factor_hat = scaling_factor_inv / (2 * np.pi)

    # Gaussian filter in frequency domain, shape: (len(freq_indices), len(K_target))
    gaussian_filter = np.exp(-0.5 * (freq_indices[:, None] / scaling_factor_hat[None, :]) ** 2)

    # Ensure response_fft has the correct number of columns
    if response_fft.shape[1] != len(K_target):
        if response_fft.shape[1] == 1:
            response_fft = np.tile(response_fft, (1, len(K_target)))
        else:
            raise ValueError("response_fft shape does not match K_target length.")

    response_at_order = response_fft * gaussian_filter
    return response_at_order
