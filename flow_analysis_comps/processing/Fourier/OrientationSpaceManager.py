import numpy as np
from scipy import fftpack
from flow_analysis_comps.data_structs.AOS_structs import (
    OSFilterParams,
)
from flow_analysis_comps.processing.AOSFilter.OrientationSpaceFilter import (
    OrientationSpaceFilter,
)
from flow_analysis_comps.processing.Fourier.OrientationSpaceResponse import (
    OrientationSpaceResponse,
    ThresholdMethods,
)
import numpy.typing as npt

from flow_analysis_comps.processing.Fourier.NLMSPrecise import nlms_precise
from flow_analysis_comps.processing.Fourier.findAllMaxima import interpft_extrema_fast
from flow_analysis_comps.util.coord_space_util import wraparoundN


class orientationSpaceManager:
    def __init__(self, params: OSFilterParams, image: np.ndarray, thresh_method = ThresholdMethods.OTSU):
        """
        Creates instance of OSManager, input with image to filter image. From there different functions can be called to retrieve

        Args:
            freq_central (float): Radial frequency corresponding to object size to search for.
            freq_width (Optional[float], optional): Breadth of frequency band to look in to. Will be set according to freq_central if not set. Defaults to None.
            K (float, optional): Orientation order, higher means more orientation space is sampled, at the cost of using more memory. Defaults to 5.
            radialOrder (bool, optional): Adjust radial width according to orientation order. Defaults to False.

        Returns:
            instance of object
        """
        self.filter_params = params
        self.image = image
        self.os_filter: OrientationSpaceFilter = OrientationSpaceFilter(params)
        self.filter_arrays, self.setup_imdims = self.init_filter_on_img(image)
        self.response = self.get_response(image)
        self.thresh_method = thresh_method
        self.mask = self.response.nlms_mask(thresh_method=thresh_method)

    def init_filter_on_img(self, img: np.ndarray):
        """Create filters for the specific image size

        Args:
            img (np.ndarray): Input image, only its shape is used
        """
        img_shape = img.shape
        filter_arrays = self.os_filter.calculate_numerical_filter(img_shape)
        filter_shape = img_shape
        return filter_arrays, filter_shape

    def get_response(self, img: np.ndarray):
        """Applies filter onto input image, first uses image shape to create filter arrays

        Args:
            img (np.ndarray): input image

        Returns:
            self: returns itself, but also creates a response object containing result
        """
        # Fourier transform image
        image_fft = fftpack.fftn(img)

        # Apply filters
        ridge_resp = self.apply_ridge_filter(image_fft)
        edge_resp = self.apply_edge_filter(image_fft)
        ang_resp = ridge_resp + edge_resp

        # Create response object capable of further processing results
        return OrientationSpaceResponse(ang_resp)

    def apply_ridge_filter(self, If: npt.NDArray[np.complex128]):
        ridge_response = fftpack.ifftn(
            If[:, :, None] * self.filter_arrays.real, axes=(0, 1)
        ).real
        return ridge_response

    def apply_edge_filter(self, If: npt.NDArray[np.complex128]):
        edge_response = 1j * (
            fftpack.ifftn(
                (If * -1j)[:, :, None] * (self.filter_arrays.imag), axes=(0, 1)
            ).real
        )
        return edge_response

    def __mul__(self, other: np.ndarray):
        return self.get_response(other)

    def update_response_at_order_FT(
        self, K_new: float, normalize: int = 2
    ) -> tuple[OrientationSpaceResponse, OrientationSpaceFilter]:
        """Adjust response of filter to a LOWER K-value

        Args:
            K_new (float): new K, must be lower than old K
            normalize (int, optional): Setting for how to normalize new result, can only be used as 2. Defaults to 2.

        Returns:
            tuple[OrientationSpaceResponse, OrientationSpaceFilter]: Updated response object instance, as well as a different filter for debug purposes.
        """
        # If same, just return
        if K_new == self.os_filter.params.orientation_accuracy:
            return self.response, self.os_filter
        else:
            n_new = 1 + 2 * K_new
            n_old = 1 + 2 * self.os_filter.params.orientation_accuracy
            s_inv = np.sqrt(n_old**2 * n_new**2 / (n_old**2 - n_new**2))
            s_hat = s_inv / (2 * np.pi)

            if normalize == 2:
                x = np.arange(1, self.response.number_of_angles + 1) - np.floor(
                    self.response.number_of_angles / 2 + 1
                )
            else:
                lower = -self.os_filter.params.sampling_factor * np.ceil(K_new)
                upper = np.ceil(K_new) * self.os_filter.params.sampling_factor
                x = np.arange(lower, upper + 1)

            if s_hat != 0:
                f_hat = np.exp(-0.5 * (x / s_hat) ** 2)
                f_hat = fftpack.ifftshift(f_hat)
            else:
                f_hat = np.array([1])

            f_hat = np.broadcast_to(f_hat, (1, 1, f_hat.shape[0]))
            # a_hat: np.ndarray = fftpack.fft(self.response.response_stack.real, axis=2)

            a_hat = self.response.response_stack_fft * f_hat

            filter_new = OrientationSpaceFilter(
                OSFilterParams(
                    space_frequency_center=self.os_filter.params.space_frequency_center,
                    space_frequency_width=self.os_filter.params.space_frequency_width,
                    orientation_accuracy=K_new,
                )
            )

            if normalize == 1:
                Response = OrientationSpaceResponse(
                    fftpack.ifft(
                        a_hat * a_hat.shape[2] / self.response.response_stack.shape[2],
                        axis=2,
                    ),
                )
            else:
                Response = OrientationSpaceResponse(fftpack.ifft(a_hat, axis=2))
            return Response, filter_new

    def get_max_angles(self):
        # mask out the non-nlms elements
        nlsm_mask = self.mask

        nanTemplate = np.zeros_like(nlsm_mask, dtype=np.float32)
        nanTemplate[:] = np.nan
        a_hat = np.rollaxis(self.response.response_stack_fft, 2, 0)
        a_hat = a_hat[:, nlsm_mask]

        maximum_single_angle = nanTemplate
        maximum_single_angle[nlsm_mask] = wraparoundN(
            -np.angle(a_hat[1, :]) / 2, 0, np.pi
        )
        return maximum_single_angle

    def get_all_angles(self):
        a_hat = np.rollaxis(self.response.response_stack_fft, 2, 0)
        response = interpft_extrema_fast(a_hat, dim=0)
        return response

    def nlms_simple_case(self, order=5):
        updated_response, filter = self.update_response_at_order_FT(order)
        maximum_single_angle = self.get_max_angles()
        nlms_single = nlms_precise(
            updated_response.response_stack.real,
            maximum_single_angle,
            mask=self.mask,
        )
        return nlms_single
