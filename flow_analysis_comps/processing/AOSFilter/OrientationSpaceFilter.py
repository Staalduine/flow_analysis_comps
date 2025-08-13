from typing import Optional
import numpy as np
from flow_analysis_comps.util.coord_space_util import freqSpaceCoords
from flow_analysis_comps.data_structs.multiori_config_struct import multiOriParams


class OrientationSpaceFilter:
    """
    Class that returns a filter stack for a given image shape. This filter stack will then be used in AOS filter functions.

    """

    def __init__(self, params: multiOriParams):
        self.params = params

    @property
    def angles(self):
        return (
            np.arange(0, self.params.nr_of_samples) / self.params.nr_of_samples * np.pi
        )

    def calculate_numerical_filter(self, imshape: tuple[int, ...]):
        # Assembles the big filter stack, can be a lot of memory for large images.
        coords = freqSpaceCoords(
            imshape, x_spacing=self.params.x_spacing, y_spacing=self.params.y_spacing
        )

        angular_filter = calculate_angular_filter(
            self.params.orientation_accuracy, self.angles, coords
        )
        radial_filter = calculate_radial_filter(
            self.params.space_frequency_center,
            self.params.space_frequency_width,
            coords,
        )

        out_filter = angular_filter * radial_filter[:, :, None]

        filter_sums = abs(np.sum(out_filter, axis=(0, 1)))
        out_filter /= filter_sums
        return out_filter


def calculate_angular_filter(
    orientation_accuracy: float = 5,
    angles: np.ndarray | None = None,
    coord_system: freqSpaceCoords | None = None,
    N: int = 1024,
):
    if angles is None:
        n = 2 * orientation_accuracy + 1  # optimized number of angles to sample at K
        angles = np.arange(n) * np.pi / (n)
    else:
        n = len(angles)

    if coord_system is None:
        coord_system = freqSpaceCoords((N, N))

    match coord_system.dims:
        case 2:
            s_a = np.pi / n
            # Expand theta with all of the angles
            theta_coords: np.ndarray = (
                coord_system.theta[:, :, None] - angles[None, None, :]
            )

            # Normalize data within [-pi, pi] range
            theta_coords = ((theta_coords + np.pi) % (2 * np.pi)) - np.pi
            posMask = np.abs(theta_coords) < np.pi / 2
            theta_s = theta_coords / s_a

            angularFilter = 2 * np.exp(-(theta_s**2) / 2)
            angularFilter_reversed = np.fft.ifftshift(
                np.fft.fftshift(angularFilter)[::-1, ::-1]
            )
            filterKernel = 0.5 * (angularFilter + angularFilter_reversed)
            filterKernel = filterKernel * (1 + 1j * (posMask * 2 - 1))
            return filterKernel
        case 3:
            raise NotImplementedError
        case _:
            raise ValueError


def calculate_radial_filter(
    freq_central,
    freq_width=None,
    coord_system: Optional[freqSpaceCoords] = None,
    N: int = 1024,
):
    if freq_width is None:
        freq_width = freq_central / np.sqrt(2)
    if coord_system is None:
        coord_system = freqSpaceCoords((N, N))

    K_f = (freq_central / freq_width) ** 2
    freq_space = coord_system.rho / freq_central

    radial_filter = freq_space**K_f
    radial_filter = radial_filter * np.exp((1 - freq_space**2) * (K_f / 2))
    return radial_filter
