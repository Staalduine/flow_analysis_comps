import numpy as np
import scipy.fftpack as fftpack


def interpolate_fourier_series(input_positions, input_values, output_positions=None, method="horner", fine_grid_factor=None, legacy=False):
    # Ensure input values are numpy arrays
    input_values = np.asarray(input_values)
    output_positions = np.asarray(output_positions)

    # Check if input values are 1D and reshape if necessary
    if input_values.ndim == 1 and input_values.shape[0] == 1:
        input_values = input_values.T
    # Check if output positions are 1D and reshape if necessary
    if output_positions.ndim == 1 and output_positions.shape[0] == 1 and (input_values.ndim > 1 or legacy):
        output_positions = output_positions.T
    # Create input positions if not provided
    if input_positions is None:
        input_positions = np.array([1, input_values.shape[0] + 1])
    elif len(input_positions) != 2:
        raise ValueError(
            "interpft1:IncorrectX: x must either be empty or have 2 elements."
        )

    period = input_positions[1] - input_positions[0]
    output_positions = output_positions - input_positions[0]
    output_positions = np.mod(np.real(output_positions), period) + 1j * np.imag(output_positions)
    output_positions = output_positions / period

    match method:
        case "horner":
            output_positions = output_positions * 2
            output_values = (
                horner_vec_real(input_values, output_positions)
                if np.isreal(output_positions).all()
                else horner_vec_complex(input_values, output_positions)
            )

        case "horner_freq":
            output_positions = output_positions * 2
            if np.isreal(output_positions).all():
                org_sz = output_positions.shape
                input_values = input_values.reshape((-1, *input_values.shape[1:]))  # Ensure v is 2D
                output_positions = output_positions.reshape((-1, *output_positions.shape[1:]))  # Ensure xq is 2D
                output_values = horner_vec_real_freq_simpler(input_values, output_positions).reshape(org_sz)
            else:
                output_values = horner_vec_complex_freq(input_values, output_positions)

        case "horner_complex":
            output_positions = output_positions * 2
            output_values = horner_vec_complex(input_values, output_positions)

        case "horner_complex":
            output_positions = output_positions * 2
            output_values = horner_vec_complex(input_values, output_positions)

        case "horner_complex_freq":
            output_positions = output_positions * 2
            output_values = horner_vec_complex_freq(input_values, output_positions)

        case "mmt":
            output_positions = output_positions * 2 * np.pi
            output_values = matrixMultiplicationTransform(input_values, output_positions)

        case "mmt_freq":
            output_positions = output_positions * 2 * np.pi
            output_values = matrixMultiplicationTransformFreq(input_values, output_positions)
        case _:
            # Use interp1 methods by expanding grid points using interpft
            fineGridFactor = parse_fine_grid_factor(fine_grid_factor, method)
            vft3 = interpft(input_values, input_values.shape[0] * fineGridFactor)
            vft3 = np.vstack([vft3[-3:, :], vft3[:, :], vft3[:4, :]])

            # Map indices from [0,1) to [4, size(v,1) * fineGridFactor + 4)
            output_positions = output_positions * (input_values.shape[0] * fineGridFactor)
            output_positions = output_positions + 4

            if legacy or output_positions.ndim == 1:  # Legacy mode or xq is a column vector
                output_values = np.interp(
                    output_positions, np.arange(len(vft3)), vft3, left=np.nan, right=np.nan
                )
            else:
                # Break xq into columns and apply interp1 to each column of v
                output_values = np.array(
                    [
                        np.interp(
                            output_positions[:, i],
                            np.arange(len(vft3)),
                            vft3[:, i],
                            left=np.nan,
                            right=np.nan,
                        )
                        for i in range(output_positions.shape[1])
                    ]
                ).T
                output_values = output_values.reshape((output_positions.shape[0],) + input_values.shape[1:])
    return output_values


def parse_fine_grid_factor(fine_grid_factor, method):
    """
    Set the fine grid factor based on the specified method. If no fine grid factor is provided,
    a default value will be set based on the method.

    Parameters:
    - fine_grid_factor : float or None
        The specified fine grid factor. If None, the function will set a default value based on the method.
    - method : str
        The method for which the fine grid factor is being set. It can be one of the following:
        'pchip', 'cubic', 'v5cubic', 'spline', or others.

    Returns:
    - fine_grid_factor : float
        The fine grid factor based on the method.
    """
    if fine_grid_factor is None:
        if method in ["pchip", "cubic", "v5cubic"]:
            fine_grid_factor = 6
        elif method == "spline":
            fine_grid_factor = 3
        else:
            fine_grid_factor = 10  # Default value for unspecified methods
    return fine_grid_factor


def matrixMultiplicationTransform(v, xq):
    """
    Wrapper function for matrixMultiplicationTransformFreq.

    Returns:
    - vq : numpy.ndarray
        Evaluated polynomial values at the query points.
    """
    return matrixMultiplicationTransformFreq(fftpack.fft(v), xq)


def matrixMultiplicationTransformFreq(v_h: np.ndarray, xq: np.ndarray):
    """
    Perform matrix multiplication transformation on Fourier coefficients.

    Parameters:
    - v_h : numpy.ndarray
        Fourier coefficients of the polynomial, shape (D, N), where D is the degree and N is the number of polynomials.
    - xq : numpy.ndarray
        Query points (Q x N), where Q is the number of query points per polynomial, and N is the number of polynomials.

    Returns:
    - vq : numpy.ndarray
        Evaluated polynomial values at the query points.
    """
    s = v_h.shape
    scale_factor = s[0]

    # Calculate Nyquist frequency
    nyquist = (s[0] + 1) // 2

    # If there's an even number of coefficients, split the Nyquist frequency
    if s[0] % 2 == 0:
        v_h[nyquist - 1, :] = v_h[nyquist - 1, :] / 2
        v_h = v_h[:nyquist, :]
        v_h = np.reshape(v_h, (s[0] + 1, *s[1:]))  # Adjust dimensions after reshaping

    # Wave number (unnormalized by the number of points)
    freq = np.concatenate(
        [np.arange(0, nyquist), np.arange(-nyquist + 1, 0)]
    )  # Frequency range

    # Calculate angles multiplied by wave number
    theta = np.outer(xq, freq)  # Matrix of angles
    waves = np.exp(1j * theta)  # Waves

    # Sum across waves weighted by Fourier coefficients
    # Permute v_h to handle dimensions appropriately
    v_h_permuted = np.transpose(
        v_h, (0, *range(2, len(v_h.shape)), 1)
    )  # Move last axis to second place

    # Perform weighted summation
    vq = np.sum(np.real(np.multiply(waves, v_h_permuted)), axis=-1) / scale_factor

    return vq


def interpft(input_values, output_array_length):
    """
    Interpolate the data 'x' to a new length 'N' using Fourier interpolation.

    """
    input_array_length = len(input_values)

    # Perform FFT
    input_vals_fft = fftpack.fft(input_values)

    # Zero pad the FFT to the new length N
    X_new = np.zeros(output_array_length, dtype=complex)
    X_new[:input_array_length] = input_vals_fft  # Keep the original FFT components

    # Perform inverse FFT to get the interpolated data
    output_values = np.real(fftpack.ifft(X_new))

    return output_values


def horner_vec_real(v, xq):
    """
    Port of the MATLAB function horner_vec_real.
    This function transforms the vector 'v' using FFT and passes it to horner_vec_real_freq.
    """
    v_fft = fftpack.fft(v)
    vq = horner_vec_real_freq(v_fft, xq)
    return vq


def horner_vec_complex(v, xq):
    vq = horner_vec_complex_freq(fftpack.fft(v), xq)
    return vq


def horner_vec_real_freq(v_h: np.ndarray, xq: np.ndarray):
    """
    Port of the MATLAB function horner_vec_real_freq.

    Parameters:
    - v_h : numpy.ndarray
        A 2D array of size (D, N) where D is the degree of the polynomial - 1, and N is the number of polynomials.
    - xq : numpy.ndarray
        A 1D array of query points (Q x N), where Q is the number of query points per polynomial, and N is the number of polynomials.

    Returns:
    - vq : numpy.ndarray
        A 2D array (Q x N), where each row contains the values of each polynomial evaluated at Q query points.
    """
    # Get the shape of v_h
    s = v_h.shape
    scale_factor = s[0]

    # Calculate Nyquist frequency
    nyquist = (s[0] + 1) // 2

    # If there's an even number of coefficients, split the Nyquist frequency
    if s[0] % 2 == 0:
        v_h[nyquist - 1, :] = np.real(v_h[nyquist - 1, :]) / 2

    # z is Q x N
    z = np.exp(1j * np.pi * xq)

    # Initialize vq as a 1 x N array
    vq = v_h[nyquist - 1, :]

    for j in range(nyquist - 2, 0, -1):
        vq = z * vq
        vq = v_h[j, :] + vq

    # Last multiplication
    vq = z * vq  # We only care about the real part
    vq = np.real(vq)

    # Add the constant term and scale
    vq = v_h[0, :] + vq * 2
    vq = vq / scale_factor

    return vq


def horner_vec_complex_freq(v_h: np.ndarray, xq: np.ndarray):
    """
    Port of the MATLAB function horner_vec_complex_freq.

    Parameters:
    - v_h : numpy.ndarray
        A 2D array of size (D, N) where D is the degree of the polynomial - 1, and N is the number of polynomials.
    - xq : numpy.ndarray
        A 1D array of query points (Q x N), where Q is the number of query points per polynomial, and N is the number of polynomials.

    Returns:
    - vq : numpy.ndarray
        A 2D array (Q x N), where each row contains the values of each polynomial evaluated at Q query points.
    """
    # Get the shape of v_h
    s = v_h.shape
    scale_factor = s[0]

    # Calculate Nyquist frequency
    nyquist = (s[0] + 1) // 2

    # If there's an even number of coefficients, split the Nyquist frequency
    if s[0] % 2 == 0:
        v_h[nyquist - 1, :] = np.real(v_h[nyquist - 1, :]) / 2

    # z is Q x N
    z = np.exp(1j * np.pi * xq)

    # Initialize vq as a 1 x N array
    vq = v_h[nyquist - 1, :]

    for j in list(range(nyquist - 2, 0, -1)) + list(range(s[0] - 1, nyquist, -1)):
        vq = z * vq
        vq = v_h[j, :] + vq

    # Last multiplication
    vq = z * vq  # We only care about the real part

    # Add the constant term and scale
    vq = v_h[nyquist, :] + vq
    vq = vq / scale_factor

    return vq


def horner_vec_real_freq_simpler(v_h: np.ndarray, xq: np.ndarray):
    """
    Simplified version of horner_vec_real_freq, evaluating polynomials with real coefficients at query points.

    Parameters:
    - v_h : numpy.ndarray
        Fourier coefficients of the polynomials, shape (D, N), where D is the degree of the polynomial minus 1 and N is the number of polynomials.
    - xq : numpy.ndarray
        Query points, shape (Q, N), where Q is the number of query points per polynomial and N is the number of polynomials.

    Returns:
    - vq : numpy.ndarray
        Evaluated polynomial values at the query points, shape (Q, N).
    """
    nQ = xq.shape[0]
    s = v_h.shape
    scale_factor = s[0]

    # Calculate Nyquist frequency
    nyquist = (s[0] + 1) // 2

    # If there's an even number of coefficients, split the Nyquist frequency
    if s[0] % 2 == 0:
        v_h[nyquist - 1, :] = np.real(v_h[nyquist - 1, :]) / 2

    # z is Q x N
    z = np.exp(1j * np.pi * xq)

    # Initialize vq with the Nyquist coefficient
    vq = v_h[nyquist - 1, :]
    vq = np.tile(vq, (nQ, 1))  # Replicate the Nyquist coefficient for all query points

    # Iterate over the Fourier coefficients and apply the Horner method
    for j in range(nyquist - 2, 0, -1):
        vq = z * vq  # Multiply by z (the wave)
        vq = v_h[j, :] + vq  # Add the Fourier coefficients

    # Last multiplication with z and take the real part
    vq = z * vq
    vq = np.real(vq)  # Only care about the real part

    # Add constant term and scale the result
    vq = v_h[0, :] + vq * 2
    vq = vq / scale_factor

    return vq
