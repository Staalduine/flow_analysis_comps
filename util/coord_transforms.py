import numpy as np
from scipy import fftpack

def cart2pol(x,y):
    rho = np.linalg.norm([x,y], axis=0)
    theta = np.arctan2(y, x)
    return rho, theta

class freqSpaceCoords:
    def __init__(self, dim_shape: np.ndarray, fps: float = 1.0):
        # dim_shape is supposed to be an image size (for 2d or 3d)
        self.x, self.y, self.z = None, None, None
        self.rho, self.theta, self.phi = None, None, None
        self.dims = 0

        match len(dim_shape):
            case 2:
                self.init_2d(dim_shape)
            case 3:
                self.init_3d(dim_shape)
            case _:
                print(f"Unexpected length of img.shape {len(dim_shape)}")
                raise ValueError

    def init_2d(self, dim_shape):
        self.dims = 2
        self.x, self.y = np.meshgrid(
            np.arange(0, dim_shape[1]) - np.floor(dim_shape[1] / 2),
            np.arange(0, dim_shape[0]) - np.floor(dim_shape[0] / 2),
        )
        self.x, self.y = fftpack.ifftshift(self.x), fftpack.ifftshift(self.y)
        self.rho, self.theta = cart2pol(self.x, self.y)