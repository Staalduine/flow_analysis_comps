import numpy as np
from flow_analysis_comps.processing.Multiori.utils.halleyft import halleyft
from flow_analysis_comps.processing.Multiori.utils.findAllMaxima import (
    find_all_extrema_in_filter_response,
)
from flow_analysis_comps.processing.Multiori.utils.interpft_derivatives import (
    interpft1_derivatives,
)


# TODO: Fix this. Does not work now
def orientation_maxima_first_derivative(
    response_hat: np.ndarray,
    response_K: np.ndarray | float,
    maxima: np.ndarray,
    period: float | None = None,
    refine: bool = False
):
    """
    Get first derivative of the maxima with respect to K and t.

    Parameters
    ----------
    response_hat : np.ndarray
        Fourier transform of response.
    response_K : float or np.ndarray
        K of response.
    maxima : np.ndarray
        Maxima positions.
    period : float, optional
        Period of the domain. Default is 2*pi.
    refine : bool, optional
        Whether to refine maxima using Halley's method. Default is False.

    Returns
    -------
    dm_dK : np.ndarray
        First derivative of maxima with respect to K.
    dm_dt : np.ndarray
        First derivative of maxima with respect to t.
    maxima : np.ndarray
        Possibly refined maxima.
    """
    if period is None or period == 0:
        period = 2 * np.pi

    D = period**2 / 2

    if refine:
        # Refine maxima using Halley's method
        new_maxima, partial_derivs, *_ = halleyft(
            response_hat,
            maxima,
            is_frequency_domain=True,
            derivative_order=1,
            unique_zeros=True,
        )
        # Only keep maxima (second derivative negative)
        new_maxima[partial_derivs[:, :, 1] > 0] = np.nan

        # Error correction: check if number of maxima changed
        nMaxima = maxima.shape[0] - np.sum(np.isnan(maxima), axis=0)
        nNewMaxima = new_maxima.shape[0] - np.sum(np.isnan(new_maxima), axis=0)
        error = nMaxima != nNewMaxima
        if np.any(error):
            temp_maxima, _ = find_all_extrema_in_filter_response(
                response_hat[:, error], do_fft=False
            )
            # Pad or truncate temp_maxima to match new_maxima rows
            s = temp_maxima.shape[0]
            new_maxima[:s, error] = temp_maxima
            if s < new_maxima.shape[0]:
                new_maxima[s:, error] = np.nan
        maxima = new_maxima
        partial_derivs = partial_derivs[:, :, 1:3]
    else:
        # Compute derivatives at maxima
        partial_derivs = interpft1_derivatives(
            response_hat, maxima, [2, 3], period, True
        )

    # dm_dt = -partial_derivs(:,:,2)./partial_derivs(:,:,1)*D;
    dm_dt = -partial_derivs[:, :, 1] / partial_derivs[:, :, 0] * D
    # dm_dK = dm_dt.*-4./(2*response_K+1).^3;
    dm_dK = dm_dt * -4.0 / (2 * response_K + 1) ** 3

    return dm_dK, dm_dt, maxima
