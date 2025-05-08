import numpy as np
from numpy.fft import fft, fftshift
from numpy.polynomial import Polynomial
from joblib import Parallel, delayed
from flow_analysis_comps.Fourier.utils.Interpolation import interpft, interpft1


def roots_batch(coeffs_batch):
    roots_out = []
    for coeffs in coeffs_batch:
        try:
            roots = np.roots(coeffs[::-1])  # MATLAB-style coeff order
        except:
            roots = np.array([np.nan])
        roots_out.append(roots)
    return roots_out


def interpft_extrema_fast(
    x, dim=1, sorted_output=False, TOL=1e-10, dofft=False, n_jobs=-1
):
    x = np.asarray(x)

    if dim != 0:
        x = np.moveaxis(x, dim, 0)

    s = x.shape
    n = s[0]

    if n == 1:
        empty = np.zeros_like(x)
        return tuple(np.moveaxis(arr, 0, dim) for arr in (empty, empty, empty, x, x, x))

    x_h = fft(x, axis=0) if dofft else x

    nyquist = int(np.ceil((n + 1) / 2))

    if n % 2 == 0:
        x_h[nyquist - 1] /= 2
        x_h = np.insert(x_h, nyquist, x_h[nyquist - 1], axis=0)

    freq = np.concatenate([np.arange(nyquist), -np.arange(nyquist - 1, 0, -1)])
    freq = freq[:, np.newaxis]

    dx_h = x_h * (1j * freq)
    dx2_h = x_h * -(freq**2)

    dx_h = fftshift(dx_h, axes=0)
    dx_h_flat = dx_h.reshape((dx_h.shape[0], -1))

    coeffs_batch = [dx_h_flat[:, i] for i in range(dx_h_flat.shape[1])]
    roots_out = Parallel(n_jobs=n_jobs)(
        delayed(np.roots)(coeff[::-1]) for coeff in coeffs_batch
    )

    max_roots = max(len(r) for r in roots_out)
    r = np.full((max_roots, len(roots_out)), np.nan, dtype=complex)

    for i, ri in enumerate(roots_out):
        r[: len(ri), i] = ri

    r = r.reshape((max_roots,) + x.shape[1:])

    magnitude = np.abs(np.log(np.abs(r)))
    real_map = magnitude <= abs(TOL)

    if TOL < 0:
        no_real = ~np.any(real_map, axis=0)
        for i in np.where(no_real.flatten())[0]:
            idx = np.unravel_index(i, no_real.shape)
            min_mag = np.min(magnitude[:, idx[0]])
            real_map[:, idx[0]] = magnitude[:, idx[0]] <= min_mag * 10

    r_ang = -np.angle(r)
    r_ang[r_ang < 0] += 2 * np.pi
    extrema = np.full_like(r_ang, np.nan)
    extrema[real_map] = r_ang[real_map]

    dx2_vals = interpft1([0, 2 * np.pi], dx2_h, extrema)

    maxima = np.full_like(extrema, np.nan)
    minima = np.full_like(extrema, np.nan)

    maxima_map = dx2_vals < 0
    minima_map = dx2_vals > 0

    maxima[maxima_map] = extrema[maxima_map]
    minima[minima_map] = extrema[minima_map]

    other = np.full_like(extrema, np.nan)
    other[(~maxima_map & ~minima_map) & real_map] = extrema[
        (~maxima_map & ~minima_map) & real_map
    ]

    maxima_value = interpft1([0, 2 * np.pi], x_h, maxima)
    minima_value = interpft1([0, 2 * np.pi], x_h, minima)
    other_value = interpft1([0, 2 * np.pi], x_h, other)

    if dim != 0:
        maxima = np.moveaxis(maxima, 0, dim)
        minima = np.moveaxis(minima, 0, dim)
        maxima_value = np.moveaxis(maxima_value, 0, dim)
        minima_value = np.moveaxis(minima_value, 0, dim)
        other = np.moveaxis(other, 0, dim)
        other_value = np.moveaxis(other_value, 0, dim)

    return maxima, minima, maxima_value, minima_value, other, other_value
