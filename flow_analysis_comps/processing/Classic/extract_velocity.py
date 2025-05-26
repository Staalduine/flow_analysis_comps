from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from flow_analysis_comps.processing.Classic.classic_image_util import (
    extract_orientations,
    speed_from_orientation_image,
)
from flow_analysis_comps.data_structs.kymographs import (
    GSTSpeedOutputs,
    kymoOutputs,
)
from flow_analysis_comps.data_structs.kymographs import (
    GSTConfig,
)
from flow_analysis_comps.processing.Classic.plot_classic import (
    plot_fields,
    plot_summary,
)


class kymoAnalyser:
    def __init__(
        self,
        kymograph: kymoOutputs,
        gst_params: GSTConfig,
        name: str = "Unnamed",
    ):
        self.kymograph = kymograph
        self.name = name
        self.video_deltas = self.kymograph.deltas
        self.config = gst_params
        self.GST_params = gst_params.gst_params

    @property
    def orientation_images(self):
        fourier_imgs = self.kymograph

        orientation_field_left = self._orientation_field(fourier_imgs.kymo_left)
        orientation_field_right = self._orientation_field(fourier_imgs.kymo_right)
        return np.array([orientation_field_left, orientation_field_right])

    @property
    def speed_images(self):
        orientation_images = self.orientation_images
        speed_field_left = speed_from_orientation_image(
            orientation_images[0], self.video_deltas, self.config.speed_limit, True
        )
        speed_field_right = speed_from_orientation_image(
            orientation_images[1], self.video_deltas, self.config.speed_limit, False
        )
        return np.array([speed_field_left, speed_field_right])

    def _orientation_field(self, image):
        imgGSTMax = extract_orientations(image, self.GST_params)

        return imgGSTMax

    def output_speeds(self) -> GSTSpeedOutputs:
        speed_images = self.speed_images
        dataframes = self.return_summary_frames()
        return GSTSpeedOutputs(
            deltas=self.video_deltas,
            name=self.name,
            speed_left=speed_images[0],
            speed_right=speed_images[1],
            speed_mean_time_series=dataframes[0],
            speed_mean_overall=dataframes[1],
        )

    def plot_summary(self) -> tuple[Figure, dict[str, Axes]]:
        return plot_summary(
            self.kymograph,
            self.speed_images,
            self.kymograph.name,
        )

    def compute_speed_time_series(self) -> pd.DataFrame:
        # Returns a DataFrame with time series of speed.
        speed_fields = self.speed_images
        speeds_mean_over_time = [
            np.nanmean(speed_fields[0], axis=1),
            np.nanmean(speed_fields[1], axis=1),
        ]
        speed_fields_coverage = ~np.isnan(speed_fields)
        speed_fields_coverage = np.sum(speed_fields_coverage, axis=2)
        speed_fields_coverage = speed_fields_coverage / speed_fields.shape[2]
        speed_fields_coverage_ratio = speed_fields_coverage / np.sum(
            speed_fields_coverage, axis=0
        )
        speeds_mean_nan_weighted = np.nansum(
            speeds_mean_over_time * speed_fields_coverage_ratio, axis=0
        )

        time_series_df = pd.DataFrame(
            {
                "time": np.arange(speed_fields.shape[1]) * self.video_deltas.delta_t,
                "speed_left": speeds_mean_over_time[0],
                "speed_right": speeds_mean_over_time[1],
                "speed_mean": speeds_mean_nan_weighted,
            }
        )
        return time_series_df

    def compute_speed_means(self) -> pd.DataFrame:
        # Returns a DataFrame with mean speeds.
        time_series_df = self.compute_speed_time_series()
        speed_mean_left = np.nanmean(time_series_df["speed_left"])
        speed_mean_right = np.nanmean(time_series_df["speed_right"])
        speed_mean = np.nanmean(time_series_df["speed_mean"])

        mean_speeds_df = pd.DataFrame(
            {
                "speed_left": speed_mean_left,
                "speed_right": speed_mean_right,
                "speed_mean": speed_mean,
            },
            index=[0],
        )
        return mean_speeds_df

    def return_summary_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Returns a DataFrame with time series of speed and a DataFrame with mean speeds.
        time_series_df = self.compute_speed_time_series()
        mean_speeds_df = self.compute_speed_means()
        return time_series_df, mean_speeds_df
