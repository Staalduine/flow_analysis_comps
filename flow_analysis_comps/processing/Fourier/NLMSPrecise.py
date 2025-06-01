import numpy as np
import skimage as ski
from scipy.signal import resample
from scipy.interpolate import RegularGridInterpolator


def nlms_precise(
    response: np.ndarray,
    theta_max: np.ndarray,
    suppression_value=0,
    interp_method="linear",
    mask=None,
    offset_angle=None,
    angle_multiplier=3,
)-> np.ndarray:
    """Perform nonlocal maximum suppression. Sample for each point along its strongest angle, then remove all points that are not a maximum.

    Args:
        response (np.ndarray): Filter response of input image
        theta_max (np.ndarray): Strongest orientation found along pixel
        suppression_value (int, optional): Value to set for all suppressed pixels. Defaults to 0.
        interp_method (str, optional): Interpolation method to sample response on non-discretized coordinates. Defaults to "linear".
        mask (_type_, optional): Mask to remove non-desired pixels as well. Defaults to None.
        offset_angle (_type_, optional): Angle along which to sample the image, not necessary for single orientation nlms. Defaults to None.
        angle_multiplier (int, optional): Not sure, what this value does. Defaults to 3.

    Raises:
        NotImplementedError: Raises if a 3d grid is input, not implemented yet
        ValueError: Raises if dimensionality is weird

    Returns:
        np.ndarray: Array with masked (nan) pixels, suppressed (default: 0) pixels and non-suppressed pixels which carry the response value of the filter. 
    """
    # Adjust to dimensionality
    match len(theta_max.shape):
        case 3:
            raise NotImplementedError
        case 2:
            nO = 1
        case _:
            raise ValueError

    # Initiale mask
    if mask is not None:
        mask = ski.morphology.binary_dilation(mask, footprint=ski.morphology.disk(3))
    # Initiate angle offset
    if offset_angle is None:
        offset_angle = theta_max
        offset_angle[offset_angle == np.nan] = np.nanmean(theta_max)

    # Set up size of nlms arrays
    rot_response_size = response.shape

    ny = rot_response_size[0]
    nx = rot_response_size[1]
    na = rot_response_size[2]
    period = na * angle_multiplier

    # Pad data with more periods
    if mask is None:
        if angle_multiplier != 1:
            new_response = resample(response, period, t=None, axis=2)
            assert isinstance(new_response, np.ndarray)
            response = new_response
    else:
        response = np.moveaxis(response, 2, 0)
        rotationResponseTemp = resample(response[:, mask], period, axis=0)
        response = np.zeros((period, mask.shape[0], mask.shape[1]))
        response[:, :, :] = np.nan
        response[:, mask] = rotationResponseTemp
        response = np.moveaxis(response, 0, 2)
        rotationResponseTemp = 0  # clear rotationResponseTemp;

    angleIdx = theta_max

    # Pad response
    response = np.pad(response, ((1, 1), (1, 1), (0, 0)), "symmetric")
    if angle_multiplier != 1:
        response = np.pad(response, ((0, 0), (0, 0), (1, 1)), "wrap")
        angleIdx[angleIdx < 0] = angleIdx[angleIdx < 0] + np.pi
        angleIdx = angleIdx / np.pi * period + 1

    x, y = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing="ij")

    x_offset = np.cos(offset_angle)
    y_offset = np.sin(offset_angle)

    # Get coordinates to sample along the strongest gradient
    Xplus, Yplus = x + x_offset, y + y_offset
    Xminus, Yminus = x - x_offset, y - y_offset

    Xstack = np.tile(x[:, :, None], nO)
    Ystack = np.tile(y[:, :, None], nO)

    # Stack together for use in the regular grid interpolator, yielding three layers
    x = np.block([Xminus[:, :, None], Xstack, Xplus[:, :, None]])
    y = np.block([Yminus[:, :, None], Ystack, Yplus[:, :, None]])
    angleIdx = np.tile(angleIdx[:, :, None], 3)

    Xminus = 0
    Xstack = 0
    Xplus = 0
    Yminus = 0
    Ystack = 0
    Yplus = 0
    x_offset = 0
    y_offset = 0

    x_ = np.arange(response.shape[0])
    y_ = np.arange(response.shape[1])
    z_ = np.arange(response.shape[2])

    # Sample with X-, X and X+ coordinates.
    # if angle_multiplier != 1:
    interpolator = RegularGridInterpolator(
        (x_, y_, z_),
        response,
        method=interp_method,
        fill_value=0,
        bounds_error=False,
    )
    A = interpolator((x, y, angleIdx))

    # If response value is highest in the middle, then there is a local maximum
    nlms = A[:, :, 1]
    suppress = np.logical_or(nlms < A[:, :, 0], nlms < A[:, :, 2])
    nlms[suppress] = suppression_value
    return nlms
