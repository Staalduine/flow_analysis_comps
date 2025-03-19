from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
import os
import tifffile
import numpy as np
import matplotlib
import cv2
import pandas as pd
from flow_analysis_comps.data_structs.video_info import (
    cameraPosition,
    cameraSettings,
    videoInfo,
    plateInfo,
)
from datetime import datetime, date, timedelta
import json
import dask.array as da
from dask import delayed
import numpy.typing as npt


def load_tif_series_to_dask(folder_path) -> npt.ArrayLike:
    """
    Loads a series of .tif images from a folder into a Dask array.

    Parameters:
        folder_path (str): Path to the folder containing the .tif images.

    Returns:
        dask.array.Array: A Dask array representing the .tif series.
    """
    # Get sorted list of .tif files
    tif_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
        ],
        # key=lambda x: int(os.path.basename(x)[3:].split('.')[0])  # Extract number from 'Img<nr>.tif'
    )

    if not tif_files:
        raise ValueError("No .tif files found in the specified folder.")

    # Use Dask to stack images lazily
    sample_image = tifffile.imread(tif_files[0])
    dtype = sample_image.dtype
    shape = (len(tif_files),) + sample_image.shape

    def lazy_reader(filename):
        return tifffile.imread(filename)

    dask_array = da.stack(
        [
            da.from_delayed(
                delayed(lazy_reader)(file), shape=sample_image.shape, dtype=dtype
            )
            for file in tif_files
        ]
    )

    return dask_array


def read_video_metadata(address: Path) -> videoInfo:
    match address.suffix:
        case ".txt":
            return read_video_info_txt(address)
        case ".json":
            return read_video_info_json(address)


def read_video_info_json(address: Path) -> videoInfo:
    with open(str(address), encoding="utf-8-sig") as json_data:
        print(json_data)
        json_data.seek(0)
        video_json = json.load(json_data)

    if video_json["camera"]["intensity"][0] == 0:
        image_mode = "fluorescence"
    elif video_json["camera"]["intensity"][1] == 0:
        image_mode = "brightfield"
    else:
        image_mode = "brightfield"

    position = cameraPosition(
        x=video_json["video"]["location"][0],
        y=video_json["video"]["location"][1],
        z=video_json["video"]["location"][2],
    )

    camera_settings = cameraSettings(
        model=video_json["camera"]["model"],
        exposure_us=video_json["camera"]["exposure_time"] * 1e6,
        frame_rate=video_json["camera"]["frame_rate"],
        frame_size=video_json["camera"]["frame_size"],
        binning=video_json["camera"]["binning"].split("x")[0],
        gain=video_json["camera"]["gain"],
        gamma=video_json["camera"]["gamma"],
    )

    info_obj = videoInfo(
        duration=video_json["video"]["duration"],
        frame_nr=int(
            video_json["video"]["duration"] * video_json["camera"]["frame_rate"]
        ),
        mode=image_mode,
        magnification=50.0,
        position=position,
        camera_settings=camera_settings,
    )
    return info_obj


def read_video_info_txt(address: Path) -> videoInfo:
    if not address.exists():
        print(f"Could not find {address}, skipping for now")
        return

    raw_data = pd.read_csv(
        address, sep=": ", engine="python", header=0, names=["Info"], index_col=0
    )["Info"]
    # Drop all columns with no data
    raw_data = raw_data.dropna(how="all")
    for col in raw_data.index:
        raw_data[col] = raw_data[col].strip()
    time_info = " ".join(raw_data["DateTime"].split(", ")[1:])
    time_obj = datetime.strptime(time_info, "%d %B %Y %X")
    crossing_date = date.fromisoformat(raw_data["CrossDate"])

    plate_info_obj = plateInfo(
        plate_nr=raw_data["Plate"],
        root=raw_data["Root"],
        strain=raw_data["Strain"],
        treatment=raw_data["Treatment"],
        crossing_date=crossing_date,
    )

    camera_settings = cameraSettings(
        model=raw_data["Model"],
        exposure_us=float(raw_data["ExposureTime"].split(" ")[0]),
        frame_rate=float(raw_data["FrameRate"].split(" ")[0]),
        frame_size=(raw_data["FrameSize"].split(" ")[0].split("x")),
        binning=raw_data["Binning"].split("x")[0],
        gain=raw_data["Gain"],
        gamma=raw_data["Gamma"],
    )

    position = cameraPosition(
        x=raw_data["X"].split(" ")[0],
        y=raw_data["Y"].split(" ")[0],
        z=raw_data["Z"].split(" ")[0],
    )

    info_obj = videoInfo(
        plate_info=plate_info_obj,
        date_time=time_obj,
        storage_path=raw_data["StoragePath"],
        run_nr=raw_data["Run"],
        duration=timedelta(seconds=int(raw_data["Time"].strip().split(" ")[0])),
        frame_nr=int(raw_data["Frames Recorded"].strip().split("/")[0]),
        mode=raw_data["Operation"].strip().split(" ")[1].lower(),
        magnification=float(raw_data["Operation"].strip().split()[0][:-1]),
        camera_settings=camera_settings,
        position=position,
    )
    return info_obj

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
