import cv2
import numpy as np
from skimage.filters import threshold_yen

from flow_analysis_comps.processing.video_manipulation.threshold_methods import (
    RenyiEntropy_thresholding,
    find_histogram_edge,
)
from flow_analysis_comps.data_structs.video_info import videoMode


def _segment_brightfield_image(
    seg_thresh: float, mean_image: np.ndarray, std_image: np.ndarray
) -> np.ndarray:
    """
    Segments Bright-Field image based on mean and standard deviation image.

    Args:
        seg_thresh (float): Threshold on the threshold
        mean_image (np.ndarray): Mean image of image series
        std_image (np.ndarray): std image of image series

    Returns:
        np.ndarray: Segmented network of hyphae
    """
    smooth_im_blur = cv2.blur(std_image, (20, 20))
    smooth_im_blur_mean = cv2.blur(mean_image, (20, 20))

    CVs = smooth_im_blur / smooth_im_blur_mean
    CVs_blurr = cv2.blur(CVs, (20, 20))
    thresh = find_histogram_edge(CVs_blurr)

    segmented = (CVs_blurr >= thresh * seg_thresh).astype(np.uint8) * 255
    return segmented


def _segment_fluorescence_image(
    mean_img: np.ndarray, seg_thresh: float = 1.10, threshtype: str = "hist_edge"
) -> np.ndarray:
    """
    Segmentation method for brightfield video, uses vesselness filters to get result.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    threshtype:     Type of threshold to apply to segmentation. Can be hist_edge, Renyi or Yen

    """
    smooth_im_blur = cv2.blur(mean_img, (20, 20))
    match threshtype:
        case "hist_edge":
            # the biggest derivative in the hist is calculated and we multiply with a small number to sit just right of that.
            thresh = find_histogram_edge(smooth_im_blur)
            segmented = (smooth_im_blur >= thresh * seg_thresh).astype(np.uint8) * 255
        case "Renyi":
            # this version minimizes a secific entropy (phi)
            segmented = RenyiEntropy_thresholding(smooth_im_blur)
        case "Yen":
            # This maximizes the distance between the two means and probabilities, sigma^2 = p(1-p)(mu1-mu2)^2
            thresh = threshold_yen(smooth_im_blur)
            segmented = (smooth_im_blur >= thresh).astype(np.uint8) * 255
        case _:
            print("threshold type has a typo! pls fix.")
            raise ValueError

    return segmented


def segment_hyphae_w_mean_std(
    mean_image: np.ndarray, std_image: np.ndarray, seg_thresh: float, mode: videoMode
):
    match mode:
        case videoMode.BRIGHTFIELD:
            segmented = _segment_brightfield_image(seg_thresh, mean_image, std_image)
        case videoMode.FLUORESCENCE:
            segmented = _segment_fluorescence_image(mean_image, threshtype="Yen")
        case _:
            raise ValueError(f"Wrong mode, what is {mode}?")
    return segmented
