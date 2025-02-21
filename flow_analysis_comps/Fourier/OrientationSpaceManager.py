from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from scipy import fftpack
from flow_analysis_comps.Fourier.OrientationSpaceFilter import (
    OSFilterParams,
    OrientationSpaceFilter,
)
from flow_analysis_comps.Fourier.OrientationSpaceResponse import (
    OrientationSpaceResponse,
)
import numpy.typing as npt

from flow_analysis_comps.Fourier.NLMSPrecise import nlms_precise
from util.coord_transforms import wraparoundN


class orientationSpaceManager:
    def __init__(
        self,
        freq_central,
        freq_width: Optional[float] = None,
        K: float = 5,
        radialOrder=False,
    ):
        if radialOrder and freq_width is None:
            freq_width = freq_central / np.sqrt(K)

        params = OSFilterParams(freq_central=freq_central, freq_width=freq_width, K=K)
        self.filter: OrientationSpaceFilter = OrientationSpaceFilter(params)
        self.filter_arrays: Optional[npt.NDArray[np.complex128]] = None
        self.setup_imdims = None
        self.response: Optional[OrientationSpaceResponse] = None

    def init_filter_on_img(self, img: np.ndarray):
        """Create filters for the specific image size

        Args:
            img (np.ndarray): Input image, only its shape is used
        """
        img_shape = img.shape
        self.filter_arrays = self.filter.setup_filter(img_shape)
        self.setup_imdims = img_shape

    def get_response(self, img: np.ndarray):
        """Applies filter onto input image, first uses image shape to create filter arrays

        Args:
            img (np.ndarray): input image

        Returns:
            self: returns itself, but also creates a response object containing result
        """

        # Initiate filters based on image size. Scaling is just done on pixel level.
        if self.setup_imdims != img.shape:
            print(f"Initiating filters on new image with dims {img.shape}")
            self.init_filter_on_img(img)

        # Fourier transform image
        If = fftpack.fftn(img)

        # Apply filters
        ridge_resp = self.apply_ridge_filter(If)
        edge_resp = self.apply_edge_filter(If)
        ang_resp = ridge_resp + edge_resp

        # Create response object capable of further processing results
        self.response = OrientationSpaceResponse(ang_resp)
        return self

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

    def update_response_at_order_FT(self, K_new, normalize=2):
        # If same, just return
        if K_new == self.filter.params.K:
            return self.response, self.filter
        else:
            n_new = 1 + 2 * K_new
            n_old = 1 + 2 * self.filter.params.K
            s_inv = np.sqrt(n_old**2 * n_new**2 / (n_old**2 - n_new**2))
            s_hat = s_inv / (2 * np.pi)

            if normalize == 2:
                x = np.arange(1, self.response.n + 1) - np.floor(
                    self.response.n / 2 + 1
                )
            else:
                lower = -self.filter.params.sample_factor * np.ceil(K_new)
                upper = np.ceil(K_new) * self.filter.params.sample_factor
                x = np.arange(lower, upper + 1)

            if s_hat != 0:
                f_hat = np.exp(-0.5 * (x / s_hat) ** 2)
                f_hat = fftpack.ifftshift(f_hat)
            else:
                f_hat = 1

            f_hat = np.broadcast_to(f_hat, (1, 1, f_hat.shape[0]))
            a_hat: np.ndarray = fftpack.fft(self.response.response_array.real, axis=2)

            a_hat = a_hat * f_hat

            filter_new = OrientationSpaceFilter(
                OSFilterParams(
                    freq_central=self.filter.params.freq_central,
                    freq_width=self.filter.params.freq_width,
                    K=K_new,
                )
            )
            # self.filter = filter_new

            if normalize == 1:
                Response = OrientationSpaceResponse(
                    fftpack.ifft(
                        a_hat * a_hat.shape[2] / self.response.response_array.shape[2],
                        axis=2,
                    ),
                )
            else:
                Response = OrientationSpaceResponse(fftpack.ifft(a_hat, axis=2))
            return Response, filter_new

    def get_max_angles(self, order=5):
        # Set up arrays
        nlsm_mask = self.response.nlms_mask()

        nanTemplate = np.zeros_like(nlsm_mask, dtype=np.float32)
        nanTemplate[:] = np.NaN
        a_hat = np.rollaxis(self.response.a_hat, 2, 0)
        a_hat = a_hat[:, nlsm_mask]

        maximum_single_angle = nanTemplate
        maximum_single_angle[nlsm_mask] = wraparoundN(
            -np.angle(a_hat[1, :]) / 2, 0, np.pi
        )
        return maximum_single_angle

    def nlms_simple_case(self, order=5):
        updated_response, filter = self.update_response_at_order_FT(order)
        maximum_single_angle = self.get_max_angles(order)
        nlms_single = nlms_precise(
            updated_response.response_array.real,
            maximum_single_angle,
            mask=self.response.nlms_mask(),
        )
        return nlms_single

    ## Plotting functions
    def demo_image(self, img, order=5):
        fig, ax = plt.subplot_mosaic(
            [["img", "."], ["orient", "colorwheel"], ["nlms", "."]],
            width_ratios=[8, 2],
            figsize=(5, 8),
            dpi=200,
            layout='constrained'
        )

        self.get_response(img)
        simple_angles = self.get_max_angles(order)
        nlms_candidates = self.nlms_simple_case(order)
        self.response.visualize_orientation_wheel(ax=ax["colorwheel"])

        img_show = ax["img"].imshow(img, cmap="cet_CET_L20")
        ax["img"].set_aspect("equal")
        fig.colorbar(img_show)

        ax["orient"].imshow(simple_angles, cmap="cet_CET_C3_r", vmin=0, vmax=np.pi)
        nlms_show = ax["nlms"].imshow(
            nlms_candidates, cmap="cet_CET_L19", vmin=0, vmax=np.pi
        )
        fig.colorbar(nlms_show)
        # fig.tight_layout()

        # return fig
