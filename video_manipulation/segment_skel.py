import cv2
import scipy
import imageio
import numpy as np
from pathlib import Path
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
import networkx as nx
from util.graph_util import generate_nx_graph, remove_spurs, from_sparse_to_graph


def incremental_mean_std_address(
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


def find_histogram_edge(image: np.ndarray) -> float:
    """
    Uses a sobel filter to find the point of highest gradient.

    Args:
        image (np.ndarray): Image

    Returns:
        float: Pixel value of highest image gradient
    """
    # Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 40)
    hist = hist.astype(float) / hist.max()  # Normalize the histogram

    # Sobel Kernel
    sobel_kernel = np.array([-1, 0, 1])

    # Apply Sobel edge detection to the histogram
    sobel_hist = convolve(hist, sobel_kernel)

    # Find the point with the highest gradient change
    threshold = np.argmax(sobel_hist)

    return bins[threshold]


def segment_meanstd_image(
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


def segment_brightfield_ultimate(
    image_addresses: list[Path],
    seg_thresh: float = 1.15,
) -> np.ndarray:
    """
    Segmentation method for brightfield video.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    threshtype:     Type of threshold to apply to segmentation. Can be hist_edge, Renyi or Yen

    """
    mean_image, std_image = incremental_mean_std_address(image_addresses)
    segmented = segment_meanstd_image(seg_thresh, mean_image, std_image)
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
