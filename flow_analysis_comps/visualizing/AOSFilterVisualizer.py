from copy import copy
from typing import Optional
from flow_analysis_comps.data_structs.AOS_structs import OSFilterParams
from flow_analysis_comps.processing.Fourier.OrientationSpaceManager import orientationSpaceManager
from flow_analysis_comps.processing.Fourier.OrientationSpaceResponse import ThresholdMethods
from flow_analysis_comps.util.coord_space_util import cart2pol
from matplotlib import pyplot as plt
import numpy as np

class AOSVisualizer:
    def __init__(self, params: OSFilterParams) -> None:
        self.params = params
        
    
    ## Plotting functions
    def plot_mean_response(self, response_array:np.ndarray, mean_response):
        largest_value = max(
            abs(response_array.min()), abs(response_array.max())
        )
        fig, ax = plt.subplots(dpi=200)
        ax.imshow(
            mean_response.__abs__(),
            cmap="cet_CET_D1A",
            vmin=-largest_value,
            vmax=largest_value,
        )
        return fig

    def plot_lineof_point(self, response_array, angles, response_fft, coord):
        """
        Plot the filter response of a point in space with orientation units

        Args:
            coord (tuple[int, int, Optional[int]]): input coordinate, can be 2d, or 3d (not implemented yets)
        """
        fig, ax = plt.subplots(1, 2)

        match len(coord):
            case 2:
                # Assume img comes in as array (so y,x or z, x axes), so flip the coords
                ax[0].plot(angles, response_array[coord[1], coord[0], :].real)
                ax[0].set_xlim(0, np.pi)
                ax[0].set_xticks(np.linspace(0, np.pi, 3, endpoint=True))
                ax[0].set_xticklabels(["0", r"$\pi$/2", r"$\pi$"])
                ax[1].plot(response_fft[coord[1], coord[0], :].real)

    def visualize_point_response(self, response_array, angles, coord, mesh_size=128):
        match len(coord):
            case 2:
                line_plot = response_array[coord[1], coord[0], :].real
                xs, ys = np.meshgrid(
                    np.linspace(-1, 1, mesh_size, endpoint=True),
                    np.linspace(-1, 1, mesh_size, endpoint=True),
                )
                rs, thetas_2 = cart2pol(xs, ys)
                theta_response_2 = np.interp(
                    (thetas_2 + np.pi / 2) % np.pi, angles, line_plot, period=np.pi
                )
                theta_response_2 *= rs < 1
                theta_response_2 *= rs > 0.5

                fig, ax = plt.subplots()
                ax.imshow(theta_response_2)

    def visualize_orientation_wheel(self, mesh_size = 128, cmap = 'cet_CET_C3_r', ax = None):
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, mesh_size, endpoint=True),
            np.linspace(-1, 1, mesh_size, endpoint=True),
        )
        rs, thetas_2 = cart2pol(xs, ys)

        thetas_2 *= rs < 1
        thetas_2 *= rs > 0.5
        thetas_2 = (thetas_2 + np.pi/2) % np.pi
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(thetas_2, cmap= cmap)
        ax.set_xticks([])
        ax.set_yticks([])


    ## Plotting functions
    def demo_image(
        self,
        AOSManager : orientationSpaceManager,
        order=5,
        invert=False,
        histo_thresh=0.5,
        speed_extent=10,
        inner_pad=5,
    ):
        fig, ax = plt.subplot_mosaic(
            [
                ["img", "img"],
                ["nlms", "nlms"],
                # ["overlay", "overlay"],
                ["total_histo", "temporal_histo"],
            ],
            # width_ratios=[8, 2],
            figsize=(8, 6),
            dpi=200,
            layout="constrained",
        )
        kymo_extent = (
            AOSManager.filter_params.x_spacing,
            AOSManager.filter_params.x_spacing * AOSManager.image.shape[1],
            AOSManager.filter_params.y_spacing * (AOSManager.image.shape[0] - inner_pad),
            inner_pad * AOSManager.filter_params.y_spacing,
        )

        if invert:
            AOSManager.image = AOSManager.image.max() - AOSManager.image

        simple_angles = AOSManager.get_max_angles()
        simple_speeds = (
            np.tan(simple_angles) / AOSManager.filter_params.y_spacing * AOSManager.filter_params.x_spacing
        )  # um.s-1
        nlms_candidates = AOSManager.nlms_simple_case(order)
        nlms_candidates = np.where(np.isnan(nlms_candidates), 0, nlms_candidates)

        if inner_pad > 0:
            nlms_candidates = nlms_candidates[inner_pad:-inner_pad]
            simple_speeds = simple_speeds[inner_pad:-inner_pad]

        palette = copy(plt.get_cmap("cet_CET_L16"))
        palette.set_under("white", 1.0)

        ax["img"].imshow(
            AOSManager.image,
            cmap=palette,
            extent=kymo_extent
        )

        ax["nlms"].imshow(
            nlms_candidates,
            cmap=palette,
            vmin=histo_thresh,
            vmax=nlms_candidates.max(),
            extent=kymo_extent,
        )

        ax["nlms"].set_ylabel("time (s)")
        ax["nlms"].set_xlabel(r"Curvilinear position ($\mu m$)")

        time_histo = []
        for speed_row, mask_row in zip(simple_speeds, nlms_candidates):
            speed_row = np.where(mask_row > histo_thresh, speed_row, np.nan)
            histo_moment = np.histogram(speed_row, 500, (-speed_extent, speed_extent))[
                0
            ]
            time_histo.append(histo_moment)
        time_histo = np.array(time_histo)

        ax["total_histo"].hist(
            simple_speeds[nlms_candidates > histo_thresh],
            bins=150,
            range=(-speed_extent, speed_extent),
        )
        ax["total_histo"].set_ylabel("frequency")
        ax["total_histo"].set_xlabel(r"velocity ($\mu m / s$)")
        ax["total_histo"].axvline(0, c="black", alpha=0.4)
        ax["temporal_histo"].imshow(
            time_histo.T,
            cmap="cet_CET_L8",
            extent=(0, len(time_histo) * AOSManager.filter_params.y_spacing, -speed_extent, speed_extent),
        )
        ax["temporal_histo"].set_ylabel(r"velocity ($\mu m / s$)")
        ax["temporal_histo"].set_xlabel("time (s)")
        for ax_title in ax:
            ax[ax_title].set_aspect("auto")

        return fig, ax, time_histo