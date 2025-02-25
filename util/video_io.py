from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
import os
import tifffile
import numpy as np
import matplotlib
import cv2
import pandas as pd
from data_structs.video_info import videoInfo, plateInfo
from datetime import datetime, date


def read_video_info_txt(address: Path) -> videoInfo:
    if not address.exists():
        print(f"Could not find {address}, skipping for now")
        return

    raw_data = pd.read_csv(
        address, sep=": ", engine="python", header=0, names=["Info"], index_col=0
    )["Info"]
    # Drop all columns with no data
    raw_data = raw_data.dropna(how="all")
    raw_data
    time_info = " ".join(raw_data["DateTime"].split(", ")[1:])
    time_obj = datetime.strptime(time_info, "%d %B %Y %X")
    crossing_date = date.fromisoformat(raw_data["CrossDate"].strip())

    plate_info_obj = plateInfo(
        plate_nr=raw_data["Plate"],
        root=raw_data["Root"].strip(),
        strain=raw_data["Strain"].strip(),
        treatment=raw_data["Treatment"].strip(),
        crossing_date=crossing_date,
    )

    info_obj = videoInfo(plate_info=plate_info_obj, datetime=time_obj)
    return raw_data

    # raw_data["unique_id"] = [f"{address.parts[-3]}_{address.parts[-2]}"]
    # raw_data["tot_path"] = (
    #     address.relative_to(analysis_folder).parent / "Img"
    # ).as_posix()
    # raw_data["tot_path_drop"] = ["DATA/" + raw_data["tot_path"][0]]
    # if raw_data["Operation"].to_string().split(" ")[-1] == "Undetermined":
    #     print(
    #         f"Undetermined operation in {raw_data['unique_id'].to_string().split(' ')[-1]}, please amend. Assuming 50x BF."
    #     )
    #     raw_data["Operation"] = "  50x Brightfield"
    # try:
    #     txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)
    # except:
    #     print(f"Weird concatenation with {address}, trying to reset index")
    #     print(raw_data.columns)
    #     txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)


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
