import cv2
import flow_analysis_comps.io as io
from pathlib import Path
import pandas as pd
import numpy as np

from flow_analysis_comps.data_structs.kymograph_structs import (
    kymoExtractConfig,
    kymoOutputs,
)
from flow_analysis_comps.data_structs.GST_structs import GST_params

from flow_analysis_comps.data_structs.video_metadata_structs import videoInfo
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.processing.graph_extraction.graph_extract import (
    VideoGraphExtractor,
)
from flow_analysis_comps.processing.GSTSpeedExtract.extract_velocity import kymoAnalyser
from flow_analysis_comps.processing.kymographing.kymographer import KymographExtractor
from flow_analysis_comps.visualizing.GraphVisualize import (
    GraphVisualizer,
)
from flow_analysis_comps.visualizing.GSTSpeeds import GSTSpeedVizualizer
from flow_analysis_comps.util.logging import setup_logger
import imageio.v3


def coord_to_folder(x: float, y: float, precision: int = 2):
    def fmt(val):
        val = round(val, precision)
        if val < 0:
            prefix = "n"
            val = -val
        else:
            prefix = ""
        return prefix + str(val).replace(".", "_")

    return f"x_{fmt(x)}_y_{fmt(y)}"


def process(run_info_index, process_args):
    # expecting a float in um/s
    separate_positions = bool(process_args[1]) if len(process_args) > 1 else False
    speed_config = GST_params()

    row = run_info_index
    path = Path(row["total_path"])

    video_io = videoIO(path)

    metadata_json_path = video_io.root_folder / "video_metadata.json"
    with open(metadata_json_path, "w", encoding="utf-8-sig") as f:
        f.write(video_io.metadata.model_dump_json())

    if video_io.metadata.position:
        pos_x, pos_y = video_io.metadata.position.x, video_io.metadata.position.y
    else:
        pos_x, pos_y = 0.0, 0.0
    video_position = coord_to_folder(pos_x, pos_y, precision=3)

    timeformat = "%Y%m%d_%H%M%S"
    formatted_timestamp = (
        video_io.metadata.date_time.strftime(timeformat)
        if video_io.metadata.date_time
        else "unknown_timestamp"
    )

    if separate_positions:
        out_folder: Path = path / "flow_analysis" / video_position / formatted_timestamp
    else:
        out_folder: Path = path / "flow_analysis" / formatted_timestamp

    process_video(path, out_folder, speed_config, video_position, formatted_timestamp,user_metadata=video_io.metadata)


def process_video(
    root_folder: Path,
    out_folder: Path,
    speed_config: GST_params,
    video_position: str | None = None,
    formatted_timestamp: str | None = None,
    kymo_extract_config: kymoExtractConfig | None = None,
    user_metadata: videoInfo | None = None,
):
    video_process_logger = setup_logger(name="flow_analysis_comps.video_processing")
    out_folder.mkdir(exist_ok=True, parents=True)

    if kymo_extract_config is None:
        kymo_extract_config = kymoExtractConfig()
    if video_position is None:
        video_position = "vid"
    if formatted_timestamp is None:
        formatted_timestamp = "extract"
    if not user_metadata:
        user_metadata = io.read_video_metadata(root_folder)

    graph_data = VideoGraphExtractor(user_metadata).edge_data

    kymo_extractor = KymographExtractor(user_metadata, graph_data, kymo_extract_config)

    kymograph_list = kymo_extractor.processed_kymographs
    kymograph_videos = kymo_extractor.hyphal_videos
    edges = kymo_extractor.edges

    edge_extraction_fig = GraphVisualizer(
        graph_data, kymo_extract_config
    ).plot_extraction()
    edge_extraction_fig.savefig(out_folder / "edges_map.png")
    video_process_logger.info(
        f"Extracted edges from {root_folder} and saved edge map to {out_folder}"
    )

    averages_list = []
    for kymo, edge in zip(kymograph_list,edges):

        kymo_averages = process_kymo(
            kymo, out_folder, speed_config, video_position, formatted_timestamp
        )
        # save hyphal video with video settings
        hyphal_video = kymograph_videos[kymo.name]
        hyphal_video_path = out_folder / kymo.name / f"{video_position}_{formatted_timestamp}_{kymo.name}_hyphal_video.mp4"
        hyphal_edge_path = out_folder / kymo.name / "edge_pixels.npy"
        metadata_path = out_folder / kymo.name / "metadata.json"
        framerate = user_metadata.camera.frame_rate if user_metadata.camera.frame_rate else 30.0

        # Convert to uint8 if needed
        if hyphal_video.dtype != np.uint8:
            hyphal_video = (255 * (hyphal_video - hyphal_video.min()) / 
                          (hyphal_video.max() - hyphal_video.min())).astype(np.uint8)

        # Get video dimensions
        height, width = hyphal_video[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(str(hyphal_video_path), fourcc, framerate, (width, height), isColor=False)
        
        # Write frames
        for frame in hyphal_video:
            out.write(frame)
        
        # Release video writer
        out.release()

        video_json = user_metadata.model_dump_json()

        # Save to file
        with open(metadata_path, "w") as f:
            f.write(video_json)
        np.save(hyphal_edge_path,edge.pixel_list) # type: ignore
        kymo_averages["kymo_name"] = kymo.name  # Add name as a column
        kymo_averages.set_index("kymo_name", inplace=True)  # Set as index
        averages_list.append(kymo_averages)

    # Merge averages into a single dataframe with kymo.name as index
    all_averages_df = pd.concat(averages_list)
    all_averages_df.to_json(out_folder / "all_kymograph_averages.json")

    video_process_logger.info(
        f"Processed {len(kymograph_list)} kymographs from {root_folder} and saved to {out_folder}"
    )
    return


def process_kymo(
    kymo: kymoOutputs,
    out_folder: Path,
    speed_config: GST_params,
    video_position: str | None = None,
    formatted_timestamp: str | None = None,
):
    if video_position is None:
        video_position = "vid"
    if formatted_timestamp is None:
        formatted_timestamp = "extract"

    kymo_speeds = kymoAnalyser(kymo, speed_config).output_speeds()
    edge_out_folder = out_folder / f"{kymo.name}"
    edge_out_folder.mkdir(exist_ok=True)
    analyser = kymoAnalyser(kymo, speed_config)
    fig, ax = GSTSpeedVizualizer(kymo_speeds).plot_summary(kymo)
    fig.savefig(
        edge_out_folder
        / f"{video_position}_{formatted_timestamp}_{kymo.name}_summary.png"
    )
    time_series, averages = analyser.return_summary_frames()
    time_series.to_json(edge_out_folder / f"{kymo.name}_time_series.json")
    averages.to_csv(edge_out_folder / f"{kymo.name}_averages.csv")
    return averages
