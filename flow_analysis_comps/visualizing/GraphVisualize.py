from flow_analysis_comps.data_structs.kymographs import (
    KymoCoordinates,
    VideoGraphExtraction,
    kymoDeltas,
    kymoExtractConfig,
)
from flow_analysis_comps.processing.kymographing.kymo_utils import (
    extract_kymo_coordinates,
)
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import colorcet


class GraphVisualizer:
    def __init__(
        self, graph: VideoGraphExtraction, extract_properties: kymoExtractConfig
    ):
        self.graph = graph
        self.image: np.ndarray = graph.io.video_array[0].compute()
        self.metadata = graph.io.metadata
        deltas = graph.io.get_deltas()
        self.deltas = kymoDeltas(
            delta_x=deltas[0],
            delta_t=deltas[1],
        )
        self.extract_properties = extract_properties
        self.segment_coords = [
            extract_kymo_coordinates(
                edge,
                self.extract_properties.step,
                self.extract_properties.resolution,
                self.extract_properties.target_length,
            )
            for edge in graph.edges
        ]

    def plot_extraction(self):
        fig1, ax1 = plt.subplots()
        ax1.imshow(
            self.image,
            extent=(
                0,
                self.deltas.delta_x * self.image.shape[1],
                self.deltas.delta_x * self.image.shape[0],
                0,
            ),
            cmap="cet_CET_L20",
        )
        ax1.set_xlabel(r"x $\mu m$")
        ax1.set_ylabel(r"y $\mu m$")

        for edge in self.segment_coords:
            if edge is not None:
                self._plot_segments(edge, self.deltas.delta_x, ax1)
        # Add 10 um scalebar
        scalebar_length_um = 10  # Uses extent imshow
        x_start = self.deltas.delta_x * self.image.shape[1] * 0.05
        y_start = self.deltas.delta_x * self.image.shape[0] * 0.95
        ax1.plot(
            [x_start, x_start + scalebar_length_um],
            [y_start, y_start],
            color="white",
            linewidth=3,
            solid_capstyle="butt",
        )
        ax1.text(
            x_start + scalebar_length_um / 2,
            y_start - self.deltas.delta_x * self.image.shape[0] * 0.02,
            r"$10 \mu m$",
            color="white",
            ha="center",
            va="bottom",
            fontsize=10,
            path_effects=[
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ],
        )
        ax1.set_axis_off()
        fig1.tight_layout()
        return fig1

    def _plot_segments(self, edge: KymoCoordinates, adjust_val, ax):
        weight = 0.05
        for point_1, point_2 in edge.segment_coords:
            ax.plot(
                [point_1[1] * adjust_val, point_2[1] * adjust_val],
                [point_1[0] * adjust_val, point_2[0] * adjust_val],
                color="white",
                alpha=0.2,
            )
            ax.text(
                *np.flip(
                    (1 - weight) * edge.edge_info.pixel_list[0]
                    + weight * edge.edge_info.pixel_list[-1]
                )
                * self.deltas.delta_x,
                str(edge.edge_info.edge[0]),
                color="white",
                path_effects=[
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ],
            )
            ax.text(
                *np.flip(
                    (1 - weight) * edge.edge_info.pixel_list[-1]
                    + weight * edge.edge_info.pixel_list[0]
                )
                * self.deltas.delta_x,
                str(edge.edge_info.edge[1]),
                color="white",
                path_effects=[
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ],
            )
