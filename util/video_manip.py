from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
import os
import tifffile
import numpy as np
import matplotlib
import cv2

def read_video_info_txt(txt_address: Path):
    pass

def tif_folder_to_mp4(folder_path, output_file, fps=10, cmap=None):
    """
    Converts a folder of single-channel .tif images to an MP4 video.

    Parameters:
        folder_path (str): Path to the folder containing the .tif images.
        output_file (str): Path to save the output MP4 file.
        fps (int): Frames per second for the video.
        cmap (str): Optional Matplotlib colormap name to apply.
    """
    # Get sorted list of .tif files
    tif_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")],
    )

    if not tif_files:
        raise ValueError("No .tif files found in the specified folder.")

    frames = []

    for tif_file in tqdm(tif_files, desc="Reading files"):
        # Read the .tif image
        img_path = os.path.join(folder_path, tif_file)
        image = tifffile.imread(img_path)

        if cmap:
            # Apply colormap if specified
            norm_image = (image - np.min(image)) / (
                np.max(image) - np.min(image)
            )  # Normalize to [0, 1]
            colormap = matplotlib.colormaps.get_cmap(cmap)
            colored_image = (colormap(norm_image)[:, :, :3] * 255).astype(
                np.uint8
            )  # Apply colormap and convert to RGB
            # colored_image = cv2.cvtColor(
            #     colored_image, cv2.COLOR_RGB2BGR
            # )  # Convert RGB to BGR for OpenCV
            frames.append(colored_image)
        else:
            # Convert single-channel image to 3-channel grayscale
            frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    # Get frame dimensions
    height, width, _ = frames[0].shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write frames to video
    for frame in tqdm(frames, desc="Making video"):
        video_writer.write(frame)

    video_writer.release()

    print(f"Video saved to {output_file}")
    return
