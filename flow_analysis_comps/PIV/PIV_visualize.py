from pathlib import Path
from typing import Callable, Optional
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import colorcet

import pandas as pd
from pydantic import BaseModel
import scipy.interpolate
import scipy.stats
import tifffile
from tqdm import tqdm


class visualizerParams(BaseModel):
    frames_per_second: float = 1.0
    pixels_per_micrometer: float = 1.0
    use_mask: bool = True
    use_flags: bool = True
    output_pattern: str = "PIV_output"


class PIV_visualize:
    def __init__(
        self,
        root_folder: Path,
        txt_folder: Path,
        parameters: Optional[visualizerParams] = None,
        limit_data=True,
    ):
        if parameters is None:
            self.params = visualizerParams(use_mask=limit_data, use_flags=limit_data)
        else:
            self.params = parameters

        self._get_info_from_folder(root_folder, txt_folder=txt_folder)
        self.pixel_extent: tuple[int, int] = (1, 1)
        self.graph_ratio = 1

        self.current_frame_index: int = 0
        self.current_frame_data: Optional[pd.DataFrame] = self.import_txt(
            self.current_frame_index
        )
        self.mean_frame_data = self.current_frame_data.copy(deep=True)

    def _get_info_from_folder(
        self,
        root_folder: Path,
        txt_folder: Optional[Path] = None,
    ):
        self.root_folder = root_folder
        if txt_folder is None:
            self.txt_folder = next(
                self.root_folder.glob(f"*{self.params.output_pattern}")
            )
        else:
            self.txt_folder = txt_folder
        self.img_folder = next(self.root_folder.glob("Img"))
        self.txt_files = sorted(list(self.txt_folder.glob("*.txt")))
        self.img_files = sorted(list(self.img_folder.glob("*.ti*")))

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

    def get_mean_direction(self):
        speed_dir_array = self.collect_data_over_time("speed_dir")
        speed_dir_array = scipy.stats.circmean(
            speed_dir_array + np.pi, axis=0, nan_policy="omit"
        )
        speeds_interpolated = self.interpolate_from_dataframe(speed_dir_array)
        return speeds_interpolated

    def get_mean_speed(self):
        speed_abs_array = self.collect_data_over_time("abs")
        speed_abs_array = np.nanmean(speed_abs_array, axis=0)
        speed_abs_interpolated = self.interpolate_from_dataframe(speed_abs_array)
        return speed_abs_interpolated

    def get_mean_generic(self, array_name, IS_MEAN_CIRCULAR: bool = False):
        def partial_circ_mean(data):
            return scipy.stats.circmean(data + np.pi, axis=0, nan_policy="omit")

        def partial_linear_mean(data):
            return np.nanmean(data, axis=0)

        match IS_MEAN_CIRCULAR:
            case True:
                mean_method = partial_circ_mean
            case False:
                mean_method = partial_linear_mean
            case _:
                raise TypeError("IS_MEAN_CIRCULAR is meant to be bool")

        total_array = self.collect_data_over_time(array_name)
        total_array = mean_method(total_array)
        total_array_interpolated = self.interpolate_from_dataframe(total_array)
        return total_array_interpolated

    def collect_data_over_time(self, data_name):
        sample_data = self.current_frame_data[data_name].to_numpy()
        data_time_array = np.zeros((len(self.txt_files), *sample_data.shape))
        for i in tqdm(range(len(self.txt_files))):
            self.set_image_index(i)
            data_time_array[i] = self.current_frame_data[data_name].to_numpy()
        return data_time_array

    def _set_pixel_extent(self, PIV_data):
        self.pixel_extent = (int(PIV_data["x"].max()), int(PIV_data["y"].max()))
        self.graph_ratio = self.pixel_extent[1] / self.pixel_extent[0] * 0.8

    def calculate_vortex_potential(
        self, dist: float = 30, index_stop=-1, speed_threshold=1e-5
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

    def plot_grid_interp_frame_data(
        self,
        data_shown="vortex",
        fig=None,
        ax=None,
        pull_from_mean_data=False,
        cmap="cet_CET_D1A",
    ) -> tuple[Axes, AxesImage]:
        if data_shown == "vortex":
            self.calculate_vortex_potential()

        if ax is None:
            fig, ax = plt.subplots()

        if pull_from_mean_data:
            color_data = self.mean_frame_data[data_shown]
        else:
            color_data = self.current_frame_data[data_shown]

        values = np.array(color_data)

        frame_input = [self.current_frame_data, self.mean_frame_data][
            pull_from_mean_data
        ]

        grid_interpolation = self.interpolate_from_dataframe(values, frame_input)

        data_max = float(abs(grid_interpolation.flatten()).max())
        # data_max = 0.5

        vortex_image = ax.imshow(
            grid_interpolation,
            cmap=cmap,
            origin="lower",
            vmin=-data_max,
            vmax=data_max,
            extent=[
                0,
                self.current_frame_data["x"].max(),
                0,
                self.current_frame_data["y"].max(),
            ],
        )

        ax.set_title(f"{data_shown} frame {self.current_frame_index}")
        ax.set_aspect("equal")
        return vortex_image

    def interpolate_from_dataframe(
        self, values, dataframe: Optional[pd.DataFrame] = None
    ):
        if dataframe is None:
            dataframe = self.current_frame_data
        linspace_x, linspace_y, xi, points = self.create_interp_grid(dataframe)

        grid_interpolation = scipy.interpolate.griddata(
            points, values, xi, method="cubic", fill_value=0.0
        )
        grid_interpolation = grid_interpolation.reshape(
            (len(linspace_y), len(linspace_x))
        )

        return grid_interpolation

    def speed_against_point_distance(self, point: tuple[float, float]):
        mesh_x = self.current_frame_data["x"]
        print(mesh_x)
        print(mesh_x.shape)

        distance = np.linalg.norm(
            [
                (self.current_frame_data["x"] - point[0]).to_numpy(),
                (self.current_frame_data["y"] - point[1]).to_numpy(),
            ],
            axis=0,
        )
        return distance, self.current_frame_data["abs"]

    def create_interp_grid(self, frame_used):
        linspace_x = np.arange(4, frame_used["x"].max(), 4)
        linspace_y = np.arange(4, frame_used["y"].max(), 4)
        mx, my = np.meshgrid(linspace_x, linspace_y)

        points = np.array([frame_used["x"], frame_used["y"]]).T

        xi = np.array([mx.flatten(), my.flatten()]).T

        return linspace_x, linspace_y, xi, points

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
        fig, ax = plt.subplots(
            1, 3, figsize=(5 * 2.7, 5 * self.graph_ratio), dpi=dpi, layout="constrained"
        )
        self.show_raw_img(ax[0])
        self.show_quiver_plot(ax=ax[1], scale=18)
        vortex_image = self.plot_grid_interp_frame_data(ax=ax[2])

        fig.colorbar(vortex_image)
        # fig.tight_layout()
        return fig
