import numpy as np
import skimage as ski
from scipy.signal import resample
from scipy.interpolate import interpn, RegularGridInterpolator, LinearNDInterpolator


def nlms_precise(
    response: np.ndarray,
    theta_max: np.ndarray,
    suppression_value=0,
    interp_method="linear",
    mask=None,
    offset_angle=None,
    angle_multiplier=3,
):
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
        offset_angle[offset_angle == np.NaN] = np.nanmean(theta_max)
    print(theta_max.shape)

    # Set up size of nlms arrays
    rot_response_size = response.shape

    ny = rot_response_size[0]
    nx = rot_response_size[1]
    na = rot_response_size[2]
    period = na * angle_multiplier

    # Pad data with more periods
    if mask is None:
        if angle_multiplier != 1:
            response = resample(response, period, axis=2)
    else:
        response = np.moveaxis(response, 2, 0)
        rotationResponseTemp = resample(response[:, mask], period, axis=0)
        response = np.zeros((period, mask.shape[0], mask.shape[1]))
        response[:, :, :] = np.NaN
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
    print(x.shape)
    
    x_offset = np.cos(offset_angle)
    y_offset = np.sin(offset_angle)

    Xplus, Yplus = x + x_offset, y + y_offset
    Xminus, Yminus = x - x_offset, y - y_offset

    Xstack = np.tile(x[:, :, None], nO)
    Ystack = np.tile(y[:, :, None], nO)

    x = np.block([Xminus[:, :, None], Xstack, Xplus[:, :, None]])
    y = np.block([Yminus[:, :, None], Ystack, Yplus[:, :, None]])
    angleIdx = np.tile(angleIdx[:, :, None], 3)
    
    print(x.shape)
    print(angleIdx.shape)

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
    print(response.shape)

    if angle_multiplier != 1:
        interpMethod = "linear"
        interpolator = RegularGridInterpolator(
            (x_, y_, z_),
            response,
            method=interpMethod,
            fill_value=0,
            bounds_error=False,
        )
        A = interpolator((x, y, angleIdx))

    nlms = A[:, :, 1]
    suppress = np.logical_or(nlms < A[:, :, 0], nlms < A[:, :, 2])
    nlms[suppress] = suppression_value
    return nlms
