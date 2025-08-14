import numpy as np
import cv2
from flow_analysis_comps.data_structs.GST_structs import GST_params
from flow_analysis_comps.data_structs.video_metadata_structs import videoDeltas
from flow_analysis_comps.processing.GSTSpeedExtract.image_util_elementary import calcGST


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
