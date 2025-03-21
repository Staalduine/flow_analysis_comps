from pathlib import Path
from typing import Optional
from openpiv import tools, pyprocess, validation, filters, scaling, windef
import tifffile
from tqdm import tqdm
from flow_analysis_comps.data_structs.video_info import videoMode
from flow_analysis_comps.video_manipulation.segment_skel import (
    segment_hyphae_general,
    harmonic_mean_thresh,
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
        visualize_folder: Optional[Path] = None,
        visualize_params: Optional[visualizerParams] = None,
    ):
        self.parameters = parameters
        self.frame_paths = sorted(
            glob(str(self.parameters.video_path) + os.sep + "Img*")
        )
        self.segmented_img = segment_hyphae_general(
            self.frame_paths[:200:10],
            mode=videoMode.BRIGHTFIELD,
            # seg_thresh=1000
        )

        self.parameters.video_path = self.select_segment_data()
        # print(f"Found {len(self.frame_paths)} images in target directory.")
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.piv_mask = None
        self.harm_mean_thresh = None

        self.visualizer = None
        if visualize_folder is not None:
            self.visualizer = PIV_visualize(visualize_folder, visualize_params)

    def select_segment_data(self) -> Path:
        match self.parameters.segment_mode:
            case videoMode.FLUORESCENCE:
                return self.parameters.video_path
            case videoMode.BRIGHTFIELD:
                out_adr = self.parameters.video_path.parent / "Harm_mean_thresh"
                out_adr.mkdir(exist_ok=True)
                _, thresh = harmonic_mean_thresh(
                    tifffile.imread(self.frame_paths[0]), self.segmented_img
                )
                print(f"Threshold used: {thresh}")
                self.segment_data(thresh, out_adr)
                return out_adr
                # return self.parameters.video_path
            case _:
                print("Data is not pre-thresholded")
                return self.parameters.video_path

    def segment_data(
        self,
        threshold_val: float,
        out_adr: Path,
    ):
        for frame_adr in tqdm(self.frame_paths):
            frame = tifffile.imread(frame_adr)
            _, frame_threshed = cv2.threshold(
                (frame.max() - frame), threshold_val, 255, cv2.THRESH_TOZERO_INV
            )
            tifffile.imwrite(
                out_adr / Path(frame_adr).name, frame_threshed, imagej=True
            )

    def plot_raw_images(self, frames: tuple[int, int]):
        img1 = tifffile.imread(self.frame_paths[frames[0]])
        img2 = tifffile.imread(self.frame_paths[frames[1]])

        img1, thresh = harmonic_mean_thresh(img1)
        img2, thresh = harmonic_mean_thresh(img2)

        fig, ax = plt.subplot_mosaic(
            [["img1", "img2"], ["img1", "img2"]], figsize=(10, 6)
        )
        ax["img1"].imshow(img1, cmap=plt.cm.gray)
        ax["img2"].imshow(img2, cmap=plt.cm.gray)

    def plot_segmentation(self):
        fig, ax = plt.subplots()
        ax.imshow(self.segmented_img)

    def make_hsv_img(self) -> tuple[np.ndarray, np.ndarray]:
        if self.x is None:
            raise ValueError("Process has not ran yet!")

        hue = np.arctan2(self.v, self.u)
        val = np.linalg.norm([self.v, self.u], axis=0) * ~self.piv_mask
        val_im = val / np.nanmax(val)
        val_im = np.sqrt(val_im)
        sat = np.ones_like(val_im)

        hsv_im = np.array([hue / (2 * np.pi) + 0.5, sat, val_im])
        hsv_im = np.moveaxis(hsv_im, 0, -1)
        rgb_im = hsv_to_rgb(hsv_im)
        return rgb_im, hue, val

    def plot_results_color(self):
        if self.x is None:
            raise ValueError("Process has not ran yet!")

        rgb_im, hue, val = self.make_hsv_img()

        fig, ax = plt.subplot_mosaic(
            [["result", "hue"], ["result", "hist"]], figsize=(10, 5)
        )
        ax["result"].imshow(rgb_im)
        ax["hue"].set_title("Hue")
        ax["hue"].imshow(np.arctan2(self.y - 1, self.x - 1), cmap="hsv")
        ax["hist"].hist(val.flatten(), bins=50, range=(0.001, 0.01))

        fig.tight_layout()

    def plot_results_arrows(self, scale=4, width=0.0015):
        fig, ax = plt.subplots(figsize=(10, 10))
        tools.display_vector_field(
            Path("exp1_001.txt"),
            ax=ax,
            scaling_factor=self.parameters.px_per_mm,
            scale=scale,  # scale defines here the arrow length
            width=width,  # width is the thickness of the arrow
            on_img=True,  # overlay on the image
            image_name=self.frame_paths[0],
        )

    def piv_process_windef(self, frames: tuple[int, int]):
        settings = windef.PIVSettings()

        settings.filepath_images = self.parameters.video_path
        settings.frame_pattern_a = "Img*.tif"
        settings.frame_pattern_b = "(1+2),(2+3)"
        settings.min_max_u_disp = (-5, 5)
        settings.min_max_v_disp = (-5, 5)

        settings.windowsizes = (32, 16, 8, 4)
        settings.overlap = (16, 8, 4, 2)

        settings.static_mask = ~self.segmented_img.astype(np.bool_)
        settings.save_path = self.parameters.video_path.parent
        settings.save_folder_suffix = "PIV_output"
        settings.save_plot = False
        settings.sig2noise_threshold = self.parameters.stn_threshold
        # settings.dt = 1/20
        settings.show_all_plots = False
        settings.show_plot = False

        windef.piv(settings)

        self.visualizer = PIV_visualize(
            settings.save_path / "OpenPIV_results_8_PIV_output"
        )

    def piv_process(
        self, frames: tuple[int, int], USE_SEGMENTATION=True, FAKE_OUTLIERS=False
    ):
        # Determine time difference between frames
        frame_difference = frames[1] - frames[0]
        t_dt = 1 / (self.parameters.fps / frame_difference)

        # Load frames
        img1 = tifffile.imread(self.frame_paths[frames[0]])
        img2 = tifffile.imread(self.frame_paths[frames[1]])

        img1 = harmonic_mean_thresh(img1, self.segmented_img)
        img2 = harmonic_mean_thresh(img2, self.segmented_img)

        # Run PIV analysis
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(
            img1.astype(np.int32),
            img2.astype(np.int32),
            window_size=self.parameters.window_size,
            overlap=self.parameters.overlap_size,
            dt=t_dt,
            search_area_size=self.parameters.search_size,
            sig2noise_method="peak2peak",
        )

        # Get MeshGrid style coordinates
        x, y = pyprocess.get_coordinates(
            image_size=img1.shape,
            search_area_size=self.parameters.search_size,
            overlap=self.parameters.overlap_size,
        )

        # Filter out low signal-to-noise arrows
        invalid_mask = validation.sig2noise_val(
            sig2noise,
            threshold=self.parameters.stn_threshold,
        )

        # Apply own segmentation
        if USE_SEGMENTATION:
            destined_shape = invalid_mask.shape

            segmented_im = cv2.resize(self.segmented_img, destined_shape[::-1])
            segmented_im = np.where(segmented_im > 1, True, False)

            invalid_mask = ~(invalid_mask * segmented_im)

        # Replace outlier arrows by interpolating between non-outlier arrows
        if FAKE_OUTLIERS:
            u0, v0 = filters.replace_outliers(
                u0,
                v0,
                invalid_mask,
                method="localmean",
                max_iter=3,
                kernel_size=3,
            )

        # Scale results based on pixels per millimeter
        x, y, u3, v3 = scaling.uniform(
            x, y, u0, v0, scaling_factor=self.parameters.px_per_mm
        )

        # Change coordinates so plt plots look good
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

        self.x, self.y, self.u, self.v = x, y, u3, v3
        self.piv_mask = invalid_mask
        tools.save("exp1_001.txt", x, y, u3, v3, invalid_mask)

        return
