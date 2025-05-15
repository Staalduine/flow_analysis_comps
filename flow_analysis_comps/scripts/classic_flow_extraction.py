import os
import json

import cv2
import tifffile

from flow_analysis_comps.video_manipulation.control_class import videoControl
from flow_analysis_comps.Classic.extract_velocity import kymoAnalyser, videoDeltas
from flow_analysis_comps.Fourier.OrientationSpaceManager import orientationSpaceManager


def process(run_info_index, process_args):
    # expecting a float in um/s
    speed_limit = process_args[0]

    row = run_info_index
    path = row["total_path"]
    video_folder = os.path.join(path, "Img")
    metadata_folder = os.path.join(path, "metadata.json")

    video_operator = videoControl(video_folder, metadata_folder, resolution=1)
    kymographs = video_operator.get_kymographs()

    deltas = videoDeltas(
        delta_x = video_operator.space_pixel_size,
        delta_t = video_operator.time_pixel_size
    )

    for key, kymo in kymographs.items():
        analyser = kymoAnalyser(kymo, video_deltas=deltas, speed_threshold=speed_limit)
        analyser.plot_kymo_fields()
        fig, ax = analyser.plot_summary()
        fig.savefig(video_folder / f"{key}_summary.png")
