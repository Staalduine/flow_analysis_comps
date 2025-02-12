from typing import Optional
import numpy as np
from scipy import fftpack


class freqSpaceCoords:
    def __init__(
        self, dim_shape: np.ndarray, deltas: tuple[float, float, float] = (1, 1, 1)
    ):
        # dim_shape is supposed to be an image size (for 2d or 3d)
        self.x, self.y, self.z = None, None, None
        self.rho, self.theta, self.phi = None, None, None
        self.dims = 0
        self.deltas = deltas

        match len(dim_shape):
            case 2:
                self._init_2d(dim_shape)
            case 3:
                self._init_3d(dim_shape)
            case _:
                print(f"Unexpected length of img.shape {len(dim_shape)}")
                raise ValueError

    def _init_2d(self, dim_shape):
        self.dims = 2
        self.x, self.y = np.meshgrid(
            np.arange(0, dim_shape[1]) - np.floor(dim_shape[1] / 2),
            np.arange(0, dim_shape[0]) - np.floor(dim_shape[0] / 2),
        )
        self.x, self.y = fftpack.ifftshift(self.x), fftpack.ifftshift(self.y)
        self.rho, self.theta = cart2pol(
            self.x / np.floor(dim_shape[1] / 2) / 2,
            self.y / np.floor(dim_shape[0] / 2) / 2,
        )

    def _init_3d(self, dim_shape):
        self.dims = 3
        self.x, self.y, self.z = np.meshgrid(
            np.arange(0, dim_shape[1]) - np.floor(dim_shape[1] / 2),
            np.arange(0, dim_shape[0]) - np.floor(dim_shape[0] / 2),
            np.arange(0, dim_shape[2]) - np.floor(dim_shape[2] / 2),
        )
        self.x, self.y, self.z = (
            fftpack.ifftshift(self.x),
            fftpack.ifftshift(self.y),
            fftpack.ifftshift(self.z),
        )
        self.rho, self.theta, self.phi = cart2spher(
            self.x / np.floor(dim_shape[1] / 2) / 2,
            self.y / np.floor(dim_shape[0] / 2) / 2,
            self.z / np.floor(dim_shape[2] / 2) / 2,
        )


def cart2pol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 2d cartesian coordinates to polar coordinates

    Args:
        x (np.ndarray): x dim
        y (np.ndarray): y dim

    Returns:
        tuple[np.ndarray, np.ndarray]: rho and theta
    """
    rho = np.linalg.norm([x, y], axis=0)
    theta = np.arctan2(y, x)
    return rho, theta


def cart2spher(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts cartesian coordinates to spherical coordinates.
    (x,y being the horizontal plane, z being depth (or time))
    Convention is that theta is the rotation in the horizontal plane, phi is the angle [0, pi] from top to bottom.

    Args:
        x (np.ndarray): x dim
        y (np.ndarray): y dim
        z (np.ndarray): z dim

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: rho, theta and phi
    """
    rho = np.linalg.norm([x, y, z], axis=0)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / rho)

    return rho, theta, phi


