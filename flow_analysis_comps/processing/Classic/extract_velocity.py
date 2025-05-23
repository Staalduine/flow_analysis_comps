from typing import Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import cv2
import pandas as pd

from flow_analysis_comps.processing.Classic.classic_image_util import (
    filter_kymo_right,
    extract_orientations,
    speed_from_orientation_image,
)
from flow_analysis_comps.data_structs.kymographs import kymoDeltas
from flow_analysis_comps.processing.Classic.model_parameters import (
    GST_params,
)
from flow_analysis_comps.processing.Classic.plot_classic import (
    plot_fields,
    plot_summary,
)


class kymoAnalyser:
    def __init__(
        self,
        kymograph: np.ndarray,
        video_deltas: kymoDeltas,
        preblur=0,
        speed_threshold=10.0,
        gst_params: Optional[GST_params] = None,
        name: str = "Unnamed",
    ):
        self.kymograph = kymograph
        self.preblur = preblur
        self.name = name
        self.video_deltas = video_deltas
        self.speed_threshold = speed_threshold

        if gst_params is not None:
            self.GST_params = gst_params
        else:
            self.GST_params = GST_params()

    @property
    def kymograph_decomposed_directions(self) -> np.ndarray:
        kymo_filtered_left = filter_kymo_right(self.kymograph)
        kymo_filtered_right = np.flip(
            filter_kymo_right(np.flip(self.kymograph, axis=1)), axis=1
        )
        out = np.array([kymo_filtered_left, kymo_filtered_right])

        return out

    @property
    def orientation_images(self):
        fourier_imgs = self.kymograph_decomposed_directions

        orientation_field_left = self._orientation_field(fourier_imgs[0])
        orientation_field_right = self._orientation_field(fourier_imgs[1])
        return np.array([orientation_field_left, orientation_field_right])

    @property
    def speed_images(self):
        orientation_images = self.orientation_images
        speed_field_left = speed_from_orientation_image(
            orientation_images[0], self.video_deltas, self.speed_threshold, True
        )
        speed_field_right = speed_from_orientation_image(
            orientation_images[1], self.video_deltas, self.speed_threshold, False
        )
        return np.array([speed_field_left, speed_field_right])

    def _orientation_field(self, image):
        if self.preblur > 0:
            image = cv2.GaussianBlur(image, (self.preblur, self.preblur), 0)

        imgGSTMax = extract_orientations(image, self.GST_params)

        return imgGSTMax

    def plot_kymo_fields(self):
        plot_fields(
            self.kymograph_decomposed_directions, self.speed_images, self.video_deltas
        )

    def plot_summary(self) -> tuple[Figure, dict[str, Axes]]:
        return plot_summary(
            self.kymograph_decomposed_directions,
            self.speed_images,
            self.video_deltas,
            self.name,
        )

    def return_summary_frames(self):
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

        speed_mean_left, speed_mean_right, speed_mean = (
            np.nanmean(speeds_mean_over_time[0]),
            np.nanmean(speeds_mean_over_time[1]),
            np.nanmean(speeds_mean_nan_weighted),
        )

        time_series_df = pd.DataFrame(
            {
                "time": np.arange(speed_fields.shape[1]) * self.video_deltas.delta_t,
                "speed_left": speeds_mean_over_time[0],
                "speed_right": speeds_mean_over_time[1],
                "speed_mean": speeds_mean_nan_weighted,
            }
        )
        mean_speeds_df = pd.DataFrame(
            {
                "speed_left": speed_mean_left,
                "speed_right": speed_mean_right,
                "speed_mean": speed_mean,
            },
            index=[0],
        )
        return time_series_df, mean_speeds_df
