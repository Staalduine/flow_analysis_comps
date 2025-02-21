from typing import Optional
import numpy as np
from pydantic import BaseModel
from util.coord_transforms import freqSpaceCoords


class OSFilterParams(BaseModel):
    freq_central: float
    freq_width: Optional[float] = None
    K: float = 5
    sample_factor: int = 1
    n: Optional[int] = None


class OrientationSpaceFilter:
    def __init__(self, params: OSFilterParams):
        self.params = params

        if self.params.freq_width is None:
            self.params.freq_width = 1 / np.sqrt(2) * self.params.freq_central

        if self.params.n is None:
            self.params.n = 2 * self.params.sample_factor * np.ceil(self.params.K) + 1

    @property
    def angles(self):
        return np.arange(self.params.n) / self.params.n * np.pi

    def get_angular_kernel(self, coords=None):
        angular_filter = angular_kernel(self.params.K, self.angles, coords)
        return angular_filter

    def get_radial_filter(self, coords=None):
        radial_filter = radial_kernel(
            self.params.freq_central, self.params.freq_width, coords
        )
        return radial_filter

    def setup_filter(self, imshape):
        coords = freqSpaceCoords(imshape)

        A = self.get_angular_kernel(coords)
        R = self.get_radial_filter(coords)
        return A * R[:, :, None]


def angular_kernel(
    K=5,
    angles: Optional[np.ndarray[1]] = None,
    coord_system: Optional[freqSpaceCoords] = None,
    N: int = 1024,
):
    if angles is None:
        n = 2 * K + 1  # optimized number of angles to sample at K
        angles = np.arange(n) * np.pi / (n)
    else:
        n = len(angles)

    if coord_system is None:
        coord_system = freqSpaceCoords(np.array([N, N]))

    match coord_system.dims:
        case 2:
            s_a = np.pi / n
            # Expand theta with all of the angles
            theta_coords = coord_system.theta[:, :, None] - angles[None, None, :]

            # Normalize data within [-pi, pi] range
            theta_coords = ((theta_coords + np.pi) % (2 * np.pi)) - np.pi
            posMask = np.abs(theta_coords) < np.pi / 2
            theta_s = theta_coords / s_a

            angularFilter = 2 * np.exp(-(theta_s**2) / 2)
            angularFilter_reversed = angularFilter[::-1, ::-1, :]
            filterKernel = 0.5 * (angularFilter + angularFilter_reversed)
            filterKernel = filterKernel * (1 + 1j * (posMask * 2 - 1))
            return filterKernel
        case 3:
            raise NotImplementedError
        case _:
            raise ValueError


def radial_kernel(
    freq_central,
    freq_width=None,
    coord_system: Optional[freqSpaceCoords] = None,
    N=1024,
):
    if freq_width is None:
        freq_width = freq_central / np.sqrt(2)
    if coord_system is None:
        coord_system = freqSpaceCoords(np.array([N, N]))

    K_f = (freq_central / freq_width) ** 2
    freq_space = coord_system.rho / freq_central

    radial_filter = freq_space**K_f
    radial_filter = radial_filter * np.exp((1 - freq_space**2) * (K_f / 2))
    return radial_filter
