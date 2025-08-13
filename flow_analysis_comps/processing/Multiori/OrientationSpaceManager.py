import numpy as np
from scipy import fftpack
from flow_analysis_comps.data_structs.multiori_config_struct import (
    multiOriOutput,
    multiOriParams,
)
from flow_analysis_comps.data_structs.video_metadata_structs import videoInfo
from flow_analysis_comps.processing.Multiori.utils import (
    orientation_maxima_first_derivative,
    find_all_extrema_in_filter_response,
    find_regime_bifurcation,
    get_response_at_order_vec_hat,
    nlms_precise,
)
from flow_analysis_comps.processing.Multiori import (
    OrientationSpaceFilter,
    OrientationSpaceResponse,
    ThresholdMethods,
)
import numpy.typing as npt
from flow_analysis_comps.util import wraparoundN, mirror_pad_with_exponential_fade

def multiori_kymo_analysis(video_metadata: videoInfo, multi_ori_filter_params: multiOriParams, kymograph: np.ndarray):
    # Start with orientation manager
    os_manager = orientationSpaceManager(multi_ori_filter_params, kymograph)

    multi_ori_output = os_manager.get_all_angles()

    output_struct = multiOriOutput(
        metadata=video_metadata,
        angles_maxima=multi_ori_output["maxima"],
        angles_minima=multi_ori_output["minima"],
        values_maxima=multi_ori_output["values_max"],
        values_minima=multi_ori_output["values_min"],
    )

    return output_struct

class orientationSpaceManager:
    def __init__(
        self,
        params: multiOriParams,
        image: np.ndarray,
        thresh_method=ThresholdMethods.OTSU,
    ):
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
        self.thresh_method = thresh_method
        self.os_filter: OrientationSpaceFilter = OrientationSpaceFilter(params)

        # If image has even dimension sizes, pad to make it odd
        # TODO: Even-sized images lead to orientation offsets, possibly related to FFT artifacts for even dimensions.
        # This is a workaround, but should be fixed in the future.
        if image.ndim == 2 and (image.shape[0] % 2 == 0 or image.shape[1] % 2 == 0):
            image = np.pad(
                image,
                ((0, int(image.shape[0] % 2 == 0)), (0, int(image.shape[1] % 2 == 0))),
                mode="reflect",
            )

        # Pad image to reduce Fourier ringing artifacts
        self.image = (
            image
            if self.filter_params.padding <= 0
            else mirror_pad_with_exponential_fade(image, self.filter_params.padding)
        )

        self.filter_arrays, self.setup_imdims = self._init_filter_on_img(self.image)
        self.response = self.get_response(self.image, self.filter_params.padding)
        self.mask = self.response.nlms_mask(thresh_method=thresh_method)

    def _init_filter_on_img(self, img: np.ndarray):
        """Create filters for the specific image size

        Args:
            img (np.ndarray): Input image, only its shape is used
        """
        img_shape = img.shape
        filter_arrays = self.os_filter.calculate_numerical_filter(img_shape)
        filter_shape = img_shape
        return filter_arrays, filter_shape

    def get_response(self, img: np.ndarray, pad: int):
        """Applies filter onto input image, first uses image shape to create filter arrays

        Args:
            img (np.ndarray): input image

        Returns:
            self: returns itself, but also creates a response object containing result
        """
        # Fourier transform image
        image_fft = fftpack.fftn(img)

        # Apply filters
        ridge_resp = self._apply_ridge_filter(image_fft)
        edge_resp = self._apply_edge_filter(image_fft)
        ang_resp = ridge_resp + edge_resp  # Output array, for now still (x, y, n)
        if pad > 0:
            # Crop out the padded region
            ang_resp = ang_resp[pad:-pad, pad:-pad, ...]  # Handles 2D and 3D arrays

        # Create response object capable of further processing results
        return OrientationSpaceResponse(ang_resp)

    def _apply_ridge_filter(self, If: npt.NDArray[np.complex128]):
        ridge_response = fftpack.ifftn(
            If[:, :, None] * self.filter_arrays.real, axes=(0, 1)
        ).real
        return ridge_response

    def _apply_edge_filter(self, If: npt.NDArray[np.complex128]):
        edge_response = 1j * (
            fftpack.ifftn(
                (If * -1j)[:, :, None] * (self.filter_arrays.imag), axes=(0, 1)
            ).real
        )
        return edge_response

    def __mul__(self, other: np.ndarray):
        return self.get_response(other, pad=0)

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

            f_hat = np.broadcast_to(
                f_hat.reshape(-1, 1, 1), self.response.response_stack_fft.shape
            )

            a_hat = self.response.response_stack_fft * f_hat

            new_params = self.os_filter.params.model_copy()
            new_params.orientation_accuracy = K_new

            filter_new = OrientationSpaceFilter(new_params)

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
        a_hat = self.response.response_stack_fft.copy()
        a_hat = a_hat[:, nlsm_mask]

        maximum_single_angle = nanTemplate
        maximum_single_angle[nlsm_mask] = wraparoundN(
            -np.angle(a_hat[1, :]) / 2, 0, np.pi
        )
        return maximum_single_angle

    def get_all_angles(self) -> dict:
        interpolated_extrema_dict = find_all_extrema_in_filter_response(
            self.response.response_stack, self.filter_params.multires_filter_params
        )  # dims = (D, x, y), response should be in real space
        return interpolated_extrema_dict

    def refine_all_angles(self, lowest_response_order: float, all_angles_dict=None):
        if all_angles_dict is None:
            all_angles_dict = self.get_all_angles()
        maxima_highest_temporary, minima_highest_temporary = (
            all_angles_dict["angles_maxima"],
            all_angles_dict["angles_minima"],
        )
        n_maxima_highest_temp = maxima_highest_temporary.shape[0] - np.sum(
            np.isnan(maxima_highest_temporary), axis=0
        )
        K_high = self.filter_params.orientation_accuracy
        K_low = np.where(
            n_maxima_highest_temp - 1 > lowest_response_order,
            n_maxima_highest_temp - 1,
            lowest_response_order,
        )
        K_high, K_low = find_regime_bifurcation(
            self.response.response_stack_fft,
            self.filter_params.orientation_accuracy,
            K_high,
            K_low,
            maxima_highest_temporary,
            minima_highest_temporary,
            tolerance=0.1,
            freq=True,
        )
        # Compute initial best derivatives and related arrays
        best_derivs, _, maxima_highest_temp_refined = (
            orientation_maxima_first_derivative(
                self.response.response_stack_fft,
                self.filter_params.orientation_accuracy,
                maxima_highest_temporary,
            )
        )
        best_abs_derivs = np.abs(best_derivs)
        best_K = np.full(best_derivs.shape, self.filter_params.orientation_accuracy)
        best_maxima = np.copy(maxima_highest_temporary)
        maxima_working = np.copy(maxima_highest_temporary)

        # Loop over K values, decreasing from K_high to 1 with step K_sampling_delta
        K_sampling_delta = getattr(self.filter_params, "K_sampling_delta", 1)
        for K in np.arange(
            self.filter_params.orientation_accuracy, 0, -K_sampling_delta
        ):
            s = K > K_high
            if not np.any(s):
                continue
            lower_a_hat = get_response_at_order_vec_hat(
                self.response.response_stack_fft[:, s],
                self.filter_params.orientation_accuracy,
                K,
            )
            new_derivs, _, maxima_working_slice = orientation_maxima_first_derivative(
                lower_a_hat, K, maxima_working[:, s], period=None, refine=True
            )
            new_abs_derivs = np.abs(new_derivs)
            better = new_abs_derivs < best_abs_derivs[:, s]

            # Update bests where new is better
            best_abs_derivs[:, s][better] = new_abs_derivs[better]
            best_derivs[:, s][better] = new_derivs[better]
            best_K[:, s][better] = K
            best_maxima[:, s][better] = maxima_working_slice[better]
            maxima_working[:, s] = maxima_working_slice
            # Reconstruct maxima_highest array as in MATLAB code
        maxima_highest_temporary = best_maxima / 2

        nanTemplate = np.zeros_like(maxima_highest_temporary, dtype=np.float32)
        nanTemplate[:] = np.nan
        maxima_highest = np.full(
            (maxima_highest_temporary.shape[0],) + nanTemplate.shape,
            np.nan,
            dtype=np.float32,
        )
        for i in range(maxima_highest_temporary.shape[0]):
            maxima_highest[i][self.mask] = maxima_highest_temporary[i, :]
        # If you want to match the MATLAB shiftdim(maxima_highest,1) at the end:
        maxima_highest = np.moveaxis(maxima_highest, 0, -1)
        return maxima_highest

    def nlms_simple_case(self, order: int | None = None):
        if order:
            updated_response, filter = self.update_response_at_order_FT(order)
        else:
            updated_response = self.response
        maximum_single_angle = self.get_max_angles()
        nlms_single = nlms_precise(
            updated_response.response_stack.real,
            maximum_single_angle,
            mask=self.mask,
        )
        return nlms_single
