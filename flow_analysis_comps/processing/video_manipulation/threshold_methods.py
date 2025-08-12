from typing import Optional
import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import minimize_scalar
import cv2

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

def calculate_renyi_entropy(threshold: float, pixels: np.ndarray) -> np.ndarray:
    # Calculate probabilities and entropies
    Ps = np.mean(pixels <= threshold)
    
    assert pixels.dtype is not np.bool_, "Pixels should be a boolean array"
    Hs = -np.sum(
        pixels[pixels <= threshold] * np.log(pixels[pixels <= threshold] + 1e-10)
    ) # type: ignore
    Hn = -np.sum(pixels * np.log(pixels + 1e-10)) # type: ignore

    # Calculate phi(s)
    phi_s = np.log(Ps * (1 - Ps)) + Hs / Ps + (Hn - Hs) / (1 - Ps)

    return -phi_s


def RenyiEntropy_thresholding(image: np.ndarray) -> np.ndarray:
    # Flatten the image
    pixels = image.flatten()

    # Find the optimal threshold
    result = minimize_scalar(
        calculate_renyi_entropy, bounds=(0, 255), args=(pixels,), method="bounded"
    )

    # The image is rescaled to [0,255] and thresholded
    optimal_threshold = result.x
    _, thresholded = cv2.threshold(
        image / np.max(image) * 255, optimal_threshold, 255, cv2.THRESH_BINARY
    )

    return thresholded

def harmonic_mean(pixels: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    arr = pixels.flatten()
    if mask is not None:
        arr = np.array(
            [arr_val for arr_val, mask_val in zip(arr, mask.flatten()) if mask_val > 0]
        )
    return float(len(arr) / np.sum(1.0 / (arr[arr > 0])))


def harmonic_mean_thresh(
    img: np.ndarray, mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    img_inv = np.nanmax(img.flatten()) - img
    thresh_val = harmonic_mean(img_inv, mask)
    thresh, thresholded_image = cv2.threshold(
        img_inv, thresh_val, 255, cv2.THRESH_TOZERO_INV
    )
    return thresholded_image, thresh_val
