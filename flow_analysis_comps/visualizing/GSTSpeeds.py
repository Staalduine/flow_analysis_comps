from flow_analysis_comps.data_structs.kymographs import GSTSpeedOutputs
import numpy as np
import matplotlib.pyplot as plt


class GSTSpeedVizualizer:
    def __init__(self, flow_analysis: GSTSpeedOutputs):
        self.flow_analysis = flow_analysis

    def show_speed_fields(self):
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

        speed_abs = self.max_speed()

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
        fig.colorbar(rightIm, ax=[ax["speedLeft"], ax["speedRight"]], aspect=40).set_label(r"Speed $(\mu m/s)$")

    def max_speed(self):
        speed_images = [self.flow_analysis.speed_left, self.flow_analysis.speed_right]
        speed_min = np.nanmin(speed_images)
        speed_max = np.nanmax(speed_images)

        print(speed_max, speed_min)
        speed_abs = np.max([abs(speed_min), speed_max])
        return speed_abs
