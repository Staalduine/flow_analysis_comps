import cv2
import scipy
import imageio
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize
import networkx as nx
from util.graph_util import generate_nx_graph, remove_spurs, from_sparse_to_graph
from skimage.filters import threshold_yen

from video_manipulation.img_util import RenyiEntropy_thresholding, find_histogram_edge


def mean_std_from_img_paths(
    image_addresses: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the mean and std pixel value of a stack of images

    Args:
        image_addresses (list[Path]): List of images

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean image and std_dev image
    """
    n = len(image_addresses)
    sum_images = None
    sum_sq_diff = None

    # Create mean image from all images

    for address in image_addresses:
        image = imageio.imread(address)
        if sum_images is None:
            sum_images = np.zeros_like(image, dtype=np.float32)
            sum_sq_diff = np.zeros_like(image, dtype=np.float32)

        sum_images += image

    mean_image = sum_images / n

    # Create standard deviation image

    for address in image_addresses:
        image = imageio.imread(address)
        sq_diff = (image - mean_image) ** 2
        sum_sq_diff += sq_diff

    variance_image = sum_sq_diff / n
    std_dev_image = np.sqrt(variance_image)

    return mean_image, std_dev_image


def segment_brightfield_image(
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



def skeletonize_segmented_im(segmented: np.ndarray) -> tuple[nx.Graph, dict]:
    """
    Take segmented image and skeletonize it

    Args:
        segmented (np.ndarray): Segmented image

    Returns:
        tuple[nx.Graph, dict]: networkx graph and positions
    """
    skeletonized = skeletonize(segmented > 0)

    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)

    return nx_graph_pruned, pos


def segment_ultimate(
    image_addresses: list[Path], seg_thresh: float = 1.15, mode="BRIGHTFIELD"
) -> np.ndarray:
    """
    Segmentation method for brightfield video.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    threshtype:     Type of threshold to apply to segmentation. Can be hist_edge, Renyi or Yen

    """
    mean_image, std_image = mean_std_from_img_paths(image_addresses)
    match mode:
        case "BRIGHTFIELD":
            segmented = segment_brightfield_image(seg_thresh, mean_image, std_image)
        case "FLUO":
            segmented = segment_fluorescence_image(mean_image, threshtype="hist_edge")
        case _:
            raise ValueError(f"Wrong mode, what is {mode}?")
    return segmented


def segment_fluorescence_image(
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
            print("threshold type has a typo! rito pls fix.")
            raise ValueError

    return segmented


