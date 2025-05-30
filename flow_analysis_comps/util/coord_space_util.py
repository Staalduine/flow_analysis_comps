import numpy as np
from scipy import fftpack


class freqSpaceCoords:
    def __init__(
        self,
        dim_shape: tuple[int, int] | tuple[int, int, int],
        deltas: tuple[float, float, float] = (1.0, 1.0, 1.0),
        x_spacing=1.0,
        y_spacing=1.0,
    ):
        # dim_shape is supposed to be an image size (for 2d or 3d)
        self.x_rate, self.y_rate = x_spacing, y_spacing
        # self.x, self.y, self.z = None, None, None
        # self.rho, self.theta, self.phi = None, None, None
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
        # self.x, self.y = self.x * self.x_rate, self.y * self.y_rate
        # print(self.x, self.y)
        self.x, self.y = (fftpack.ifftshift(self.x), fftpack.ifftshift(self.y))
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


def wraparoundN(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Wrap values between an upper and lower bound. Values exceeding upper will be modulo'd.

    Args:
        values (np.ndarray): Input values
        lower (float): Lower limit
        upper (float): Upper limit

    Returns:
        np.ndarray: Wrapped values
    """
    assert lower < upper, "'lower'-value must be lower than 'upper'-value"

    wrappedValues = values - lower
    upper = upper - lower
    wrappedValues = wrappedValues.real % upper
    wrappedValues = wrappedValues + lower
    return wrappedValues


def extract_perp_lines(src, dst, linewidth=1):
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack(
        [
            np.linspace(row_i - row_width, row_i + row_width, linewidth)
            for row_i in line_row
        ]
    )
    perp_cols = np.stack(
        [
            np.linspace(col_i - col_width, col_i + col_width, linewidth)
            for col_i in line_col
        ]
    )
    return np.squeeze(np.stack([perp_rows, perp_cols]).T)


def validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype is bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the range 0-5.")

    if image_dtype is bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitely "
            "cast input image to another data type."
        )

    return order
