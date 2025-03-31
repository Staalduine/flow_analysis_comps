import numpy as np
import cv2


def calcGST(inputIMG: np.ndarray, w: int):
    """
    Calculates the Image orientation and the image coherency. Image orientation is merely a guess, and image coherency gives an idea how sure that guess is.
    inputIMG:   The input image
    w:          The window size of the various filters to use. Large boxes catch higher order structures.
    """

    # The idea here is to perceive any patch of the image as a transformation matrix.
    # Such a matrix will have some eigenvalues, which describe the direction of uniform transformation.
    # If the largest eigenvalue is much bigger than the smallest eigenvalue, that indicates a strong orientation.

    img = inputIMG.astype(np.float32)
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0, -1)
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1, -1)
    imgDiffXY = cv2.multiply(imgDiffX, imgDiffY)
    imgDiffXX = cv2.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv2.multiply(imgDiffY, imgDiffY)
    J11 = cv2.boxFilter(imgDiffXX, cv2.CV_32F, (w, w))
    J22 = cv2.boxFilter(imgDiffYY, cv2.CV_32F, (w, w))
    J12 = cv2.boxFilter(imgDiffXY, cv2.CV_32F, (w, w))
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
    height, width = kymo.shape
    paved_kymo = tile_pad_image(kymo)
    paved_kymo_filter = fourier_filter_right_diagonal(paved_kymo)
    paved_kymo = tile_pad_image(kymo)
    filtered_kymo = paved_kymo_filter[height : 2 * height, width : 2 * width]
    filtered_kymo -= np.percentile(filtered_kymo, 10)

    return filtered_kymo


class kymoAnalyser:
    def __init__(self, kymograph: np.ndarray):
        self.kymograph = kymograph
        self.window_start = 3
        self.window_amount = 15
        self.preblur = 0
        self.coherency_threshold = 0.95
        self.coherency_threhold_falloff = 0.05

    @property
    def kymograph_decomposed_directions(self) -> np.ndarray:
        kymo_filtered_left = filter_kymo_right(self.kymograph)
        kymo_filtered_right = np.flip(
            filter_kymo_right(np.flip(self.kymograph, axis=1)), axis=1
        )
        out = np.array([kymo_filtered_left, kymo_filtered_right])
        # if blur > 0:
        #     out = cv2.GaussianBlur(out, (blur, blur), 0)
        return out

    def process_fourier_images(self):
        fourier_imgs = self.kymograph_decomposed_directions

        orientation_field_left = self._orientation_field(fourier_imgs[0])
        orientation_field_right = self._orientation_field(fourier_imgs[1])
        return np.array([orientation_field_left, orientation_field_right])

    def _orientation_field(self, image):
        if self.preblur > 0:
            image = cv2.GaussianBlur(image, (self.preblur, self.preblur), 0)

        coherency_stack = np.array(
            [
                calcGST(image, w)
                for w in range(
                    self.window_start, self.window_amount * 2 + self.window_start, 2
                )
            ]
        )
        imgCoherencySum = 1 * np.greater(coherency_stack[0][0], self.coherency_threshold)
        imgCoherencySum = np.where(imgCoherencySum == 1, 0, np.nan)
        imgGSTMax = np.where(imgCoherencySum == 0, coherency_stack[0][1], np.nan)

        for w in range(1, self.window_amount):
            C_thresh_current = self.coherency_threshold - self.coherency_threhold_falloff * w
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
            imgGSTMax = np.where(
                imgCoherencySum == w, coherency_stack[w][1], imgGSTMax
            )

        return imgGSTMax
