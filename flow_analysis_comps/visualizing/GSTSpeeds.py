from flow_analysis_comps.data_structs.GST_structs import GSTSpeedOutputs
from flow_analysis_comps.data_structs.kymograph_structs import kymoOutputs
import numpy as np
import matplotlib.pyplot as plt


class GSTSpeedVizualizer:
    def __init__(self, flow_analysis: GSTSpeedOutputs):
        assert isinstance(flow_analysis.speed_left, np.ndarray), (
            "flow_analysis must be a GSTSpeedOutputs instance"
        )
        self.flow_analysis = flow_analysis
        assert isinstance(self.flow_analysis.speed_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

    def show_speed_fields(self):
        assert isinstance(self.flow_analysis.speed_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )
        assert isinstance(self.flow_analysis.speed_right, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

        kymo_extent: tuple[float, float, float, float] = (
            0,
            self.flow_analysis.deltas.delta_x * len(self.flow_analysis.speed_left[0]),
            self.flow_analysis.deltas.delta_t * len(self.flow_analysis.speed_left),
            0,
        )

        fig, ax = plt.subplot_mosaic(
            [["speedLeft", "speedRight"]],
            layout="constrained",
        )

        speed_abs = self._max_speed()

        ax["speedLeft"].imshow(
            self.flow_analysis.speed_left,
            vmin=-speed_abs,
            vmax=speed_abs,
            cmap="cet_CET_D1A",
            extent=kymo_extent,
        )
        rightIm = ax["speedRight"].imshow(
            self.flow_analysis.speed_right,
            vmin=-speed_abs,
            vmax=speed_abs,
            cmap="cet_CET_D1A",
            extent=kymo_extent,
        )

        for ax_title in ax:
            ax[ax_title].set_title(ax_title)
            ax[ax_title].set_aspect("auto")
            ax[ax_title].set_xlabel(r"Curvilinear distance ($\mu m$)")
            ax[ax_title].set_ylabel("time (s)")

        # add speed label to colorbar
        fig.colorbar(
            rightIm, ax=[ax["speedLeft"], ax["speedRight"]], aspect=40
        ).set_label(r"Speed $(\mu m/s)$")
        return fig, ax

    def plot_summary(self, fourier_images: kymoOutputs):
        name = self.flow_analysis.name
        speedmax = self._max_speed()
        kymo_extent = self._get_kymo_extent(fourier_images)
        time_axis_points = self._get_time_axis_points(fourier_images)
        speed_histo = self._get_speed_histogram(speedmax)
        speed_mean_over_time, speed_std_over_time, speed_mean_max = (
            self._get_speed_means()
        )

        fig, ax = plt.subplot_mosaic(
            [["kymograph", "temporal histogram"], ["speed plot", "temporal histogram"]],
            layout="constrained",
            dpi=300,
            figsize=(12, 8),
        )

        self._plot_kymograph(ax["kymograph"], fourier_images, kymo_extent)
        self._plot_speed(
            ax["speed plot"],
            time_axis_points,
            speed_mean_over_time,
            speed_std_over_time,
            speed_mean_max,
        )
        self._plot_histogram(
            ax["temporal histogram"], speed_histo, fourier_images, speedmax
        )

        for ax_title in ax:
            ax[ax_title].set_title(ax_title)
            ax[ax_title].set_aspect("auto")

        fig.suptitle(f"{name} - Kymograph and Speed Analysis")
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        return fig, ax

    def _get_kymo_extent(self, fourier_images: kymoOutputs):
        assert isinstance(fourier_images.kymo_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

        return (
            0,
            fourier_images.deltas.delta_x * len(fourier_images.kymo_left[0]),
            fourier_images.deltas.delta_t * len(fourier_images.kymo_left),
            0,
        )

    def _get_time_axis_points(self, fourier_images: kymoOutputs):
        assert isinstance(fourier_images.kymo_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

        return np.linspace(
            0,
            fourier_images.deltas.delta_t * len(fourier_images.kymo_left),
            len(fourier_images.kymo_left),
        )

    def _get_speed_histogram(self, speedmax):
        assert isinstance(self.flow_analysis.speed_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )
        assert isinstance(self.flow_analysis.speed_right, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

        speed_bins = np.linspace(-speedmax, speedmax, 1001)
        speed_histo_left = np.array(
            [np.histogram(row, speed_bins)[0] for row in self.flow_analysis.speed_left]
        )
        speed_histo_right = np.array(
            [np.histogram(row, speed_bins)[0] for row in self.flow_analysis.speed_right]
        )
        speed_histo = (speed_histo_left + speed_histo_right) / (
            2 * len(self.flow_analysis.speed_left[0])
        )
        return speed_histo

    def _get_speed_means(self):
        assert isinstance(self.flow_analysis.speed_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )
        assert isinstance(self.flow_analysis.speed_right, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

        speed_mean_over_time = [
            np.nanmean(self.flow_analysis.speed_left, axis=1),
            np.nanmean(self.flow_analysis.speed_right, axis=1),
        ]
        speed_std_over_time = [
            np.nanstd(self.flow_analysis.speed_left, axis=1),
            np.nanstd(self.flow_analysis.speed_right, axis=1),
        ]
        speed_mean_max = (
            np.nanmax([abs(speed_mean_over_time[0]), abs(speed_mean_over_time[1])])
            * 1.1
        )
        return speed_mean_over_time, speed_std_over_time, speed_mean_max

    def _plot_kymograph(self, ax, fourier_images, kymo_extent):
        ax.imshow(
            fourier_images.kymo_left + fourier_images.kymo_right,
            cmap="cet_CET_L20",
            extent=kymo_extent,
        )
        ax.set_xlabel(r"Curvilinear distance ($\mu m$)")
        ax.set_ylabel("time (s)")

    def _plot_speed(
        self,
        ax,
        time_axis_points,
        speed_mean_over_time,
        speed_std_over_time,
        speed_mean_max,
    ):
        ax.plot(
            time_axis_points,
            speed_mean_over_time[0],
            c="tab:orange",
            label="speed left",
        )
        ax.fill_between(
            time_axis_points,
            speed_mean_over_time[0] - speed_std_over_time[0],
            speed_mean_over_time[0] + speed_std_over_time[0],
            color="tab:orange",
            alpha=0.3,
        )
        ax.plot(
            time_axis_points, speed_mean_over_time[1], c="tab:blue", label="speed right"
        )
        ax.fill_between(
            time_axis_points,
            speed_mean_over_time[1] - speed_std_over_time[1],
            speed_mean_over_time[1] + speed_std_over_time[1],
            color="tab:blue",
            alpha=0.3,
        )
        ax.axhline(0, linestyle="--", c="black")
        ax.legend()
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"Speed ($\mu m / s$)")
        ax.set_ylim(-speed_mean_max, speed_mean_max)

    def _plot_histogram(self, ax, speed_histo, fourier_images, speedmax):
        ax.imshow(
            speed_histo.T,
            extent=(
                0,
                len(speed_histo) * fourier_images.deltas.delta_t,
                -speedmax,
                speedmax,
            ),
            origin="lower",
            cmap="cet_CET_L16",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"Speed ($\mu m / s$)")

    def _max_speed(self):
        assert isinstance(self.flow_analysis.speed_left, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )
        assert isinstance(self.flow_analysis.speed_right, np.ndarray), (
            "flow_analysis must be contain np arrays"
        )

        speed_images = [self.flow_analysis.speed_left, self.flow_analysis.speed_right]
        speed_min = np.nanmin(speed_images)
        speed_max = np.nanmax(speed_images)

        speed_abs = np.max([abs(speed_min), speed_max])
        return speed_abs
