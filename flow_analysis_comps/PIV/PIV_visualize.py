from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import colorcet

import pandas as pd
from pydantic import BaseModel


class visualizerParams(BaseModel):
    frames_per_second: float = 1.0
    pixels_per_micrometer: float = 1.0
    use_mask: bool = True
    use_flags: bool = True


class PIV_visualize:
    def __init__(self, txt_folder: Path, parameters: Optional[visualizerParams] = None):
        self.folder_address = txt_folder
        self.txt_files = sorted(list(self.folder_address.glob("*.txt")))
        self.params = parameters
        self.pixel_extent: tuple[int, int] = (1, 1)

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

        self.pixel_extent = (int(PIV_data["x"].max()), int(PIV_data["y"].max()))

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

    def calculate_vortex_potential(self, dist:float):
        data = self.current_frame_data
        data["vortex"] = 0.0
        for index, rows in self.current_frame_data.iterrows():
            data["diff_x"] = data["x"] - data["x"][index]
            data["diff_y"] = data["y"] - data["y"][index]
            data["diff_dist"] = np.linalg.norm([data["diff_x"], data["diff_y"]], axis=0)
            data["dist_bool"] = -np.arctan(data["diff_dist"]) + np.pi /2
            data["diff_theta"] = (np.arctan2(data["diff_y"], data["diff_x"])) - data[
                "speed_dir"
            ]
            data.loc[index, "vortex"] = np.mean(
                np.sin(data["diff_theta"]) * data["dist_bool"] * data["abs"] ** 2
            )
        self.current_frame_data = data

    def plot_vortex_potential(self, dist:float=30, fig=None, ax=None):
        self.calculate_vortex_potential(dist)

        if ax is None:
            fig, ax = plt.subplots()
        color_data = self.current_frame_data["vortex"]
        data_max = color_data.abs().max()

        scatter_plot = ax.scatter(
            self.current_frame_data["x"],
            self.current_frame_data["y"],
            c=color_data,
            cmap="cet_CET_D1A",
            vmin=-data_max,
            vmax=data_max,
        )
        ax.set_title(f"Vortex potential (frame {self.current_frame_index})")
        ax.set_aspect("equal")
        fig.colorbar(scatter_plot)

    def show_current_frame(self, dpi=200, scale=20, ax=None):
        graph_ratio = self.pixel_extent[1] / self.pixel_extent[0] * 0.8
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5 * graph_ratio), dpi=dpi)
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
