from typing import Optional

import numpy as np
from scipy import fftpack
from util.coord_transforms import freqSpaceCoords


class OrientationSpaceFilter:
    def __init__(
        self,
        freq_central,
        freq_width: Optional[float] = None,
        K: float = 5,
        normEnergy=None,
    ):
        self.freq_central = freq_central
        self.freq_width = freq_width
        self.K = K
        self.normEnergy = normEnergy
        self.sampleFactor = 1
        self.size = None
        self.F = None

        if freq_width is None:
            self.freq_width = 1 / np.sqrt(2) * self.freq_central

        self.n = 2 * self.sampleFactor * np.ceil(self.K) + 1

    @property
    def angles(self):
        return np.arange(self.n) / self.n * np.pi

    def get_response(self, image: np.ndarray):
        If = fftpack.fftn(image)
        self.setup_filter(If.shape)
        ridge_resp = self.apply_ridge_filter(If)
        edge_resp = self.apply_edge_filter(If)
        ang_resp = ridge_resp + edge_resp
        return ang_resp

    def get_angular_kernel(self, coords=None):
        angular_filter = angular_kernel(self.K, self.angles, coords)
        return angular_filter

    def get_radial_filter(self, coords=None):
        radial_filter = radial_kernel(self.freq_central, self.freq_width, coords)
        return radial_filter

    def apply_ridge_filter(self, If):
        ridge_response = fftpack.ifftn(If[:, :, None] * self.F.real, axes=(0, 1)).real
        return ridge_response

    def apply_edge_filter(self, If:np.ndarray):
        edge_response = 1j * (
            fftpack.ifftn((If * -1j)[:, :, None] * (self.F.imag), axes=(0, 1)).real
        )
        return edge_response

    def setup_filter(self, imshape):
        coords = freqSpaceCoords(imshape)

        A = self.get_angular_kernel(coords)
        R = self.get_radial_filter(coords)
        self.F = A * R[:, :, None]


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
