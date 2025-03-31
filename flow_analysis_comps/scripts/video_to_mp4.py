import os
import json

import cv2
import tifffile


def process(run_info_index, process_args):
    row = run_info_index
    path = row["total_path"]
    video_folder = os.path.join(path, "Img")
    metadata_folder = os.path.join(path, "metadata.json")

    with open(str(metadata_folder), encoding="utf-8-sig") as json_data:
        print(json_data)
        json_data.seek(0)
        video_json = json.load(json_data)
    fps = int(video_json["metadata"]["camera"]["frame_rate"])

    # Get sorted list of .tif files
    tif_files = sorted(
        [f for f in os.listdir(video_folder) if f.lower().endswith(".tif")],
    )

    if not tif_files:
        raise ValueError("No .tif files found in the specified folder.")

    frames = []

    for tif_file in tif_files:
        # Read the .tif image
        img_path = os.path.join(video_folder, tif_file)
        image = tifffile.imread(img_path)

        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    # Get frame dimensions
    height, width, _ = frames[0].shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        os.path.join(path, "Video.mp4"), fourcc, fps, (width, height)
    )

    # Write frames to video
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
