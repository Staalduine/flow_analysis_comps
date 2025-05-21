from pathlib import Path
from typing import Optional
from openpiv import tools, pyprocess, validation, filters, scaling, windef
import tifffile
from tqdm import tqdm
from flow_analysis_comps.data_structs.video_info import videoMode
from flow_analysis_comps.video_manipulation.threshold_methods import (
    harmonic_mean_thresh,
)
from flow_analysis_comps.video_manipulation.segmentation_methods import (
    segment_hyphae_general,
)
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from flow_analysis_comps.PIV.definitions import PIV_params, segmentMode

from flow_analysis_comps.PIV.PIV_visualize import PIV_visualize, visualizerParams


class AMF_PIV:
    def __init__(
        self,
        parameters: PIV_params,
        # visualize_params: Optional[visualizerParams] = None,
    ):
        self.parameters = parameters

        self.frame_paths = sorted(
            glob(str(self.parameters.root_path / "Img") + os.sep + "Img*")
        )
        if len(self.frame_paths) == 0:
            raise FileNotFoundError("No images in given folder")

        self.segmented_img = segment_hyphae_general(
            self.frame_paths[:200:10],
            mode=self.parameters.segment_mode,
        )

        window_sizes = tuple(
            int(self.parameters.window_size_start / (2**i))
            for i in range(self.parameters.number_of_passes)
        )
        overlap_sizes = tuple(window_size // 2 for window_size in window_sizes)


        speed_thresholds = (
            -self.parameters.max_speed_px_per_frame,
            self.parameters.max_speed_px_per_frame,
        )

        self.windef_settings = windef.PIVSettings()

        self.windef_settings.num_iterations = self.parameters.number_of_passes
        self.windef_settings.filepath_images = self.parameters.root_path / "Img"
        self.windef_settings.frame_pattern_a = "Img*.tif"
        self.windef_settings.frame_pattern_b = "(1+2),(2+3)"
        self.windef_settings.min_max_u_disp = speed_thresholds
        self.windef_settings.min_max_v_disp = speed_thresholds
        self.windef_settings.windowsizes = window_sizes
        self.windef_settings.overlap = overlap_sizes
        self.windef_settings.static_mask = ~self.segmented_img.astype(np.bool_)
        self.windef_settings.save_path = self.parameters.root_path
        self.windef_settings.save_folder_suffix = "PIV_output"
        self.windef_settings.save_plot = False
        # self.windef_settings.sig2noise_validate=True
        self.windef_settings.sig2noise_threshold = self.parameters.stn_threshold
        self.windef_settings.show_all_plots = False
        self.windef_settings.show_plot = False
        self.windef_settings.scale_plot = 8

        self.visualizer = None

    def set_target_frames(self, frame_name_A: str, frame_name_B: str):
        self.windef_settings.frame_pattern_a = frame_name_A
        self.windef_settings.frame_pattern_b = frame_name_B

    def run_full_video(
        self, video_name_pattern: str = "Img*.tif", video_sequence_pattern="(1+2),(2+3)"
    ):
        self.windef_settings.frame_pattern_a = video_name_pattern
        self.windef_settings.frame_pattern_b = video_sequence_pattern

        windef.piv(self.windef_settings)

        output_folder_number = self.windef_settings.windowsizes[
            self.windef_settings.num_iterations - 1
        ]
        output_folder_name = f"OpenPIV_results_{output_folder_number}_PIV_output"

        self.visualizer = PIV_visualize(
            self.windef_settings.save_path,
            self.windef_settings.save_path / output_folder_name,

        )

    def run_single_frame(self, frame_idxs: Optional[tuple[int, int]] = None):
        if frame_idxs is None:
            video_names = [Path(frame).name for frame in self.frame_paths[:2]]
        else:
            video_names = [
                Path(self.frame_paths[frame_idxs[0]]).name,
                Path(self.frame_paths[frame_idxs[1]]).name,
            ]

        self.set_target_frames(*video_names)
        self.windef_settings.show_plot = True
        windef.piv(self.windef_settings)
        self.windef_settings.show_plot = False

    def plot_raw_images(self, frames: tuple[int, int], segment=False):
        img1 = tifffile.imread(self.frame_paths[frames[0]])
        img2 = tifffile.imread(self.frame_paths[frames[1]])

        if segment:
            img1, thresh = harmonic_mean_thresh(img1)
            img2, thresh = harmonic_mean_thresh(img2)

        fig, ax = plt.subplot_mosaic(
            [["img1", "img2"], ["img1", "img2"]], figsize=(10, 6)
        )
        ax["img1"].imshow(img1 * self.segmented_img, cmap=plt.cm.gray)
        ax["img2"].imshow(img2, cmap=plt.cm.gray)

    def start_visualizer(self, output_file, limit_data=True):
        self.visualizer = PIV_visualize(self.parameters.root_path, output_file, limit_data=limit_data)

