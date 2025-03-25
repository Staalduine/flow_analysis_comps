from pathlib import Path
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import colorcet

import pandas as pd
from pydantic import BaseModel
import scipy.interpolate
import tifffile


class visualizerParams(BaseModel):
    frames_per_second: float = 1.0
    pixels_per_micrometer: float = 1.0
    use_mask: bool = True
    use_flags: bool = True
    output_pattern: str = "PIV_output"


class PIV_visualize:
    def __init__(self, txt_folder: Path, parameters: Optional[visualizerParams] = None):
        self.folder_address = txt_folder
        self.txt_folder = next(
            self.folder_address.glob(f"*{parameters.output_pattern}")
        )
        self.img_folder = next(self.folder_address.glob("Img"))
        self.txt_files = sorted(list(self.txt_folder.glob("*.txt")))
        self.img_files = sorted(list(self.img_folder.glob("*.ti*")))
        self.params = parameters
        self.pixel_extent: tuple[int, int] = (1, 1)
        self.graph_ratio = 1

        if parameters is None:
            self.params = visualizerParams()

        self.current_frame_index: int = 0
        self.current_frame_data: Optional[pd.DataFrame] = self.import_txt(
            self.current_frame_index
        )

    def set_image_index(self, index: int):
        self.current_frame_index = index
        self.current_frame_data = self.import_txt(index)

    def import_txt(self, path_index: int):
        PIV_data = pd.read_table(self.txt_files[path_index])
        # Txt data comes out this way
        PIV_data.rename({"# x": "x"}, axis=1, inplace=True)

        self._set_pixel_extent(PIV_data)

        if self.params.use_mask:
            PIV_data = PIV_data[PIV_data["mask"] == 0]
        if self.params.use_flags:
            PIV_data = PIV_data[PIV_data["flags"] == 0]

        PIV_data["abs"] = np.sqrt(PIV_data["u"] ** 2 + PIV_data["v"] ** 2)
        PIV_data["speed"] = (
            PIV_data["abs"]
            / self.params.pixels_per_micrometer
            * self.params.frames_per_second
        )
        PIV_data["speed_dir"] = np.arctan2(PIV_data["v"], PIV_data["u"])
        PIV_data["vel_x"], PIV_data["vel_y"] = (
            PIV_data["u"]
            / self.params.pixels_per_micrometer
            * self.params.frames_per_second,
            PIV_data["v"]
            / self.params.pixels_per_micrometer
            * self.params.frames_per_second,
        )

        return PIV_data

    def _set_pixel_extent(self, PIV_data):
        self.pixel_extent = (int(PIV_data["x"].max()), int(PIV_data["y"].max()))
        self.graph_ratio = self.pixel_extent[1] / self.pixel_extent[0] * 0.8

    def calculate_vortex_potential(
        self, dist: float, index_stop=-1, speed_threshold=1e-5
    ):
        data = self.current_frame_data
        data["vortex"] = 0.0
        data["speed_present"] = data["abs"] > speed_threshold
        for index, rows in self.current_frame_data.iterrows():
            data["diff_x"] = data["x"] - data["x"][index]
            data["diff_y"] = data["y"] - data["y"][index]
            data["diff_dist"] = np.linalg.norm([data["diff_x"], data["diff_y"]], axis=0)
            data["dist_bool"] = data["diff_dist"] < dist
            data["diff_theta"] = (np.arctan2(data["diff_y"], data["diff_x"])) - data[
                "speed_dir"
            ]
            data.loc[index, "vortex"] = np.sum(
                np.sin(data["diff_theta"]) * data["dist_bool"] * data["speed_present"]
            ) / np.sum(data["dist_bool"])
            if index == index_stop:
                break
        self.current_frame_data = data

    def plot_vortex_potential(
        self,
        dist: float = 30,
        data_shown="vortex",
        fig=None,
        ax=None,
        index_stop=-1,
        speed_threshold=1e-5,
    ) -> tuple[Axes, AxesImage]:
        self.calculate_vortex_potential(
            dist, index_stop=index_stop, speed_threshold=speed_threshold
        )

        if ax is None:
            fig, ax = plt.subplots()
        color_data = self.current_frame_data[data_shown]

        linspace_x = np.arange(4, self.current_frame_data["x"].max(), 2)
        linspace_y = np.arange(4, self.current_frame_data["y"].max(), 2)
        mx, my = np.meshgrid(linspace_x, linspace_y)

        points = np.array(
            [self.current_frame_data["x"], self.current_frame_data["y"]]
        ).T
        values = np.array(color_data)
        xi = np.array([mx.flatten(), my.flatten()]).T

        grid_interpolation = scipy.interpolate.griddata(
            points, values, xi, method="cubic", fill_value=0.0
        )
        grid_interpolation = grid_interpolation.reshape(
            (len(linspace_y), len(linspace_x))
        )

        # data_max = float(abs(grid_interpolation.flatten()).max())
        data_max = 0.5

        vortex_image = ax.imshow(
            grid_interpolation,
            cmap="cet_CET_D1A",
            origin="lower",
            vmin=-data_max,
            vmax=data_max,
            extent=(
                0,
                self.current_frame_data["x"].max(),
                0,
                self.current_frame_data["y"].max(),
            ),
        )

        ax.set_title(f"{data_shown} frame {self.current_frame_index}")
        ax.set_aspect("equal")
        return vortex_image

    def show_quiver_plot(self, dpi=200, scale=20, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5 * self.graph_ratio), dpi=dpi)
        ax.quiver(
            self.current_frame_data["x"] / self.params.pixels_per_micrometer,
            self.current_frame_data["y"] / self.params.pixels_per_micrometer,
            self.current_frame_data["u"],
            self.current_frame_data["v"],
            self.current_frame_data["abs"],
            scale=scale,
            cmap="cet_CET_L8",
        )
        ax.set_title(f"PIV results frame {self.current_frame_index}")
        ax.set_xlim(0, self.pixel_extent[0])
        ax.set_ylim(0, self.pixel_extent[1])
        ax.set_xlabel(r"x $(\mu m)$")
        ax.set_ylabel(r"y $(\mu m)$")
        ax.set_aspect("equal")

    def show_raw_img(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        img = tifffile.imread(self.img_files[self.current_frame_index])[::-1]
        ax.imshow(img, origin="lower", cmap="cet_CET_L20")
        ax.set_title(f"Raw image frame {self.current_frame_index}")

    def plot_full_figure(self, dpi=300):
        fig, ax = plt.subplots(1, 3, figsize=(5 * 2.7, 5 * self.graph_ratio), dpi=dpi, layout="constrained")
        self.show_raw_img(ax[0])
        self.show_quiver_plot(ax=ax[1], scale=18)
        vortex_image = self.plot_vortex_potential(ax=ax[2], speed_threshold=1e-1)

        fig.colorbar(vortex_image)
        # fig.tight_layout()
        return fig