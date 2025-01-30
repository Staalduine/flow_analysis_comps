from pathlib import Path
from openpiv import tools, pyprocess, validation, filters, scaling
import tifffile
from video_manipulation.segment_skel import segment_brightfield_ultimate
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from flow_analysis_comps.PIV.definitions import PIV_params, segmentMode


class AMF_PIV:
    def __init__(self, parameters: PIV_params):
        self.parameters = parameters
        self.frame_paths = glob(
            str(self.parameters.video_path / "Img") + os.sep + "Img*"
        )
        self.segmented_img = segment_brightfield_ultimate(
            self.frame_paths[:20],
            mode=self.parameters.segment_mode,
        )
        print(f"Found {len(self.frame_paths)} images in target directory.")
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.piv_mask = None

    def plot_raw_images(self, frames: tuple[int, int]):
        img1 = tifffile.imread(self.frame_paths[frames[0]])
        img2 = tifffile.imread(self.frame_paths[frames[1]])

        fig, ax = plt.subplot_mosaic(
            [["img1", "img2"], ["img1", "img2"]], figsize=(20, 20)
        )
        ax["img1"].imshow(img1, cmap=plt.cm.gray)
        ax["img2"].imshow(img2, cmap=plt.cm.gray)

    def plot_segmentation(self):
        fig, ax = plt.subplots()
        ax.imshow(self.segmented_img)
        
    def make_hsv_img(self):
        if self.x is None:
            raise ValueError("Process has not ran yet!")
        
        hue = np.arctan2(self.v, self.u) 
        val = np.linalg.norm([self.v, self.u], axis=0)
        val /= np.nanmax(val)
        val = np.sqrt(val)
        sat = np.ones_like(val)

        hsv_im = np.array([hue/ (2*np.pi) + 0.5, sat, val])
        hsv_im = np.moveaxis(hsv_im, 0, -1)
        rgb_im = hsv_to_rgb(hsv_im)
        return rgb_im, hue
    
    
    
    def plot_results_color(self):
        if self.x is None:
            raise ValueError("Process has not ran yet!")
        
        rgb_im, hue = self.make_hsv_img()
        
        fig, ax = plt.subplot_mosaic([["result", "hue"], ["result","hist"]], figsize=(20,10))
        ax["result"].imshow(rgb_im)
        ax["hue"].set_title("Hue")
        ax["hue"].imshow(np.arctan2(self.y - 1, self.x - 1), cmap="hsv")
        ax["hist"].hist(hue.flatten(), bins=50)

        fig.tight_layout()
    
    def plot_results_arrows(self, scale=4, width=0.0015):
        fig, ax = plt.subplots(figsize=(20,20))
        tools.display_vector_field(
            Path('exp1_001.txt'),
            ax=ax, scaling_factor=self.parameters.px_per_mm,
            scale=scale, # scale defines here the arrow length
            width=width, # width is the thickness of the arrow
            on_img=True, # overlay on the image
            image_name= self.frame_paths[0],
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
            x,
            y,
            u0,
            v0,
            scaling_factor=self.parameters.px_per_mm
        )

        # Change coordinates so plt plots look good
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

        self.x, self.y, self.u, self.v = x, y, u3, v3
        self.piv_mask = invalid_mask
        return
