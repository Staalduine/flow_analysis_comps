import numpy as np
import cv2
from flow_analysis_comps.data_structs.GST_structs import GST_params
from flow_analysis_comps.data_structs.video_metadata_structs import videoDeltas


def calcGST(inputIMG: np.ndarray, window_size: int):
    """
    Calculates the Image orientation and the image coherency. Image orientation is merely a guess, and image coherency gives an idea how sure that guess is.
    inputIMG:   The input image
    w:          The window size of the various filters to use. Large boxes catch higher order structures.
    """

    # The idea here is to perceive any patch of the image as a transformation matrix.
    # Such a matrix will have some eigenvalues, which describe the direction of uniform transformation.
    # If the largest eigenvalue is much bigger than the smallest eigenvalue, that indicates a strong orientation.

    img = inputIMG.astype(np.float32)
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # TODO: check if this is correct
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    imgDiffXY = cv2.multiply(imgDiffX, imgDiffY)
    imgDiffXX = cv2.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv2.multiply(imgDiffY, imgDiffY)
    J11 = cv2.boxFilter(imgDiffXX, cv2.CV_32F, (window_size, window_size))
    J22 = cv2.boxFilter(imgDiffYY, cv2.CV_32F, (window_size, window_size))
    J12 = cv2.boxFilter(imgDiffXY, cv2.CV_32F, (window_size, window_size))
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv2.multiply(tmp2, tmp2)
    tmp3 = cv2.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
    lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue
    imgCoherencyOut = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
    imgOrientationOut = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    imgOrientationOut = 0.5 * imgOrientationOut
    return imgCoherencyOut, imgOrientationOut


def fourier_filter_right_diagonal(tiled_image):
    dft = np.fft.fftshift(
        np.fft.fft2(tiled_image)
    )  # Apply FFT and shift zero frequency to center
    # Zero out specific regions in the Fourier transform
    h, w = dft.shape
    # Horizontal line across the middle
    dft[h // 2 - 1 : h // 2 + 1, :] = 0
    # Top-left quadrant
    dft[: h // 2, : w // 2] = 0
    # botom-right quadrant
    dft[h // 2 :, w // 2 :] = 0

    # Inverse Fourier Transform
    inverse_dft = np.fft.ifft2(np.fft.ifftshift(dft)).real

    return inverse_dft


def tile_pad_image(kymo: np.ndarray):
    """
    Create a tiled version of the kymograph image by duplicating and flipping it in all four quadrants.

    Args:
        kymo (np.ndarray): The input kymograph image.

    Returns:
        _type_: A larger image with the original kymograph and its flipped versions arranged in a 3x3 grid.
    """
    height, width = kymo.shape

    # Duplicate and manipulate images
    subFourier1 = kymo.copy()
    subFourier2 = cv2.flip(kymo, 1)  # Flip horizontally
    subFourier3 = cv2.flip(subFourier2, 0)  # Flip vertically after horizontal
    subFourier4 = cv2.flip(kymo, 0)  # Flip vertically

    # Create a larger image and place manipulated images accordingly
    filter_forward = np.zeros((3 * height, 3 * width), dtype=kymo.dtype)
    filter_forward[height : 2 * height, width : 2 * width] = subFourier1
    filter_forward[height : 2 * height, 0:width] = subFourier2
    filter_forward[height : 2 * height, 2 * width : 3 * width] = subFourier2
    filter_forward[0:height, 0:width] = subFourier3
    filter_forward[0:height, 2 * width : 3 * width] = subFourier3
    filter_forward[2 * height : 3 * height, 0:width] = subFourier3
    filter_forward[2 * height : 3 * height, 2 * width : 3 * width] = subFourier3
    filter_forward[0:height, width : 2 * width] = subFourier4
    filter_forward[2 * height : 3 * height, width : 2 * width] = subFourier4
    return filter_forward


def filter_kymo_right(kymo: np.ndarray):
    # Apply Fourier transform to the kymograph image and filter out the right diagonal
    height, width = kymo.shape
    paved_kymo = tile_pad_image(kymo)
    paved_kymo_filter = fourier_filter_right_diagonal(paved_kymo)
    paved_kymo = tile_pad_image(kymo)
    filtered_kymo = paved_kymo_filter[height : 2 * height, width : 2 * width]
    filtered_kymo -= np.percentile(filtered_kymo, 10)

    return filtered_kymo


def GST_extract_orientations(image: np.ndarray, gst_params: GST_params):
    """
    Extracts the orientations from an image using the Generalized Structure Tensor (GST) method.
    This function calculates the GST for a range of window sizes and determines the orientation

    Args:
        image (np.ndarray): The input image from which to extract orientations.
        gst_params (GST_params): Parameters for the GST calculation, including window sizes and coherency thresholds.

    Returns:
        _type_: An array containing the maximum GST orientation for each pixel in the image.
    """
    coherency_stack = np.array(
        [
            calcGST(image, w)
            for w in range(
                gst_params.window_start,
                gst_params.window_amount * 2 + gst_params.window_start,
                2,
            )
        ]
    )
    imgCoherencySum = 1 * np.greater(
        coherency_stack[0][0], gst_params.coherency_threshold
    )
    imgCoherencySum = np.where(imgCoherencySum == 1, 0, np.nan)
    imgGSTMax = np.where(imgCoherencySum == 0, coherency_stack[0][1], np.nan)

    for w in range(1, gst_params.window_amount):
        C_thresh_current = (
            gst_params.coherency_threshold - gst_params.coherency_threshold_falloff * w
        )
        coherency_interest = np.where(
            coherency_stack[w][0] > C_thresh_current,
            coherency_stack[w][0],
            np.nan,
        )
        imgCoherencySum = np.where(
            coherency_interest > coherency_stack[w - 1][0], w, imgCoherencySum
        )
        newValues = np.isnan(imgCoherencySum) * (
            np.invert(np.isnan(coherency_interest))
        )
        imgCoherencySum = np.where(newValues, w, imgCoherencySum)
        imgGSTMax = np.where(imgCoherencySum == w, coherency_stack[w][1], imgGSTMax)
    return imgGSTMax


def speed_from_orientation_image(image: np.ndarray, deltas: videoDeltas):
    """
    Calculates the speed from an orientation image using the tangent of the angle.

    Parameters
    ----------
    image : np.ndarray
        The orientation image, where each pixel represents an angle in degrees. Range of orientation is 0-180 degrees.
    deltas : kymoDeltas
        An object containing the spatial (`delta_x`) and temporal (`delta_t`) resolution of the image.

    Returns
    -------
    np.ndarray
        An array of the same shape as `image`, containing the calculated speeds of interest. Values outside the threshold or not matching the desired sign are set to NaN.

    Line-by-line description
    -----------------------
    1. Computes the unthresholded speed for each pixel using the tangent of the angle (converted from degrees to radians), scaled by the spatial and temporal resolution.
    5. Returns the resulting speed array.
    """

    # Ensure orientation is in range (-pi/2, pi/2)
    assert np.all(image >= -np.pi / 2) and np.all(image <= np.pi / 2), (
        "Orientation image must be in range (-pi/2, pi/2) radians."
    )

    speed = np.tan(image) * deltas.delta_x / deltas.delta_t
    return speed
