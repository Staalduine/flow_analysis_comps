from pathlib import Path
import pandas as pd

from flow_analysis_comps.data_structs.kymographs import (
    GSTConfig,
    graphExtractConfig,
    kymoExtractConfig,
    kymoOutputs,
)
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.processing.graph_extraction.graph_extract import (
    VideoGraphExtractor,
)
from flow_analysis_comps.processing.GSTSpeedExtract.extract_velocity import kymoAnalyser
from flow_analysis_comps.processing.kymographing.kymographer import KymographExtractor
from flow_analysis_comps.util.video_io import coord_to_folder
from flow_analysis_comps.visualizing.GraphVisualize import (
    GraphVisualizer,
)
from flow_analysis_comps.visualizing.GSTSpeeds import GSTSpeedVizualizer
from flow_analysis_comps.util.logging import setup_logger
import imageio



def process(run_info_index, process_args):
    # expecting a float in um/s
    speed_limit = float(process_args[0])
    separate_positions = bool(process_args[1]) if len(process_args) > 1 else False
    speed_config = GSTConfig(speed_limit=speed_limit)

    row = run_info_index
    path = Path(row["total_path"])

    video_io = videoIO(path)
    pos_x, pos_y = video_io.metadata.position.x, video_io.metadata.position.y
    video_position = coord_to_folder(pos_x, pos_y, precision=3)

    timeformat = "%Y%m%d_%H%M%S"
    formatted_timestamp = video_io.metadata.date_time.strftime(timeformat)

    if separate_positions:
        out_folder: Path = path / "flow_analysis" / video_position / formatted_timestamp
    else:
        out_folder: Path = path / "flow_analysis" / formatted_timestamp

    process_video(path, out_folder, speed_config, video_position, formatted_timestamp)


def process_video(
    root_folder: Path,
    out_folder: Path,
    speed_config: GSTConfig,
    kymo_extract_config: kymoExtractConfig = kymoExtractConfig(),
    video_position: str | None = None,
    formatted_timestamp: str | None = None,
    user_metadata: videoInfo | None = None,
):
    video_process_logger = setup_logger(name="flow_analysis_comps.video_processing")
    out_folder.mkdir(exist_ok=True, parents=True)

    if video_position is None:
        video_position = "vid"
    if formatted_timestamp is None:
        formatted_timestamp = "extract"

    graph_data = VideoGraphExtractor(
        root_folder, graphExtractConfig(), user_metadata=user_metadata
    ).edge_data

    kymo_extractor = KymographExtractor(
        graph_data, kymo_extract_config
    )

    kymograph_list = kymo_extractor.processed_kymographs
    kymograph_videos = kymo_extractor.hyphal_videos

    edge_extraction_fig = GraphVisualizer(
        graph_data, kymo_extract_config
    ).plot_extraction()
    edge_extraction_fig.savefig(out_folder / "edges_map.png")
    video_process_logger.info(
        f"Extracted edges from {root_folder} and saved edge map to {out_folder}"
    )

    averages_list = []
    for kymo in kymograph_list:
        kymo_averages = process_kymo(
            kymo, out_folder, speed_config, video_position, formatted_timestamp
        )
        # save hyphal video with video settings
        hyphal_video = kymograph_videos[kymo.name]
        hyphal_video_path = out_folder / kymo.name / f"{video_position}_{formatted_timestamp}_{kymo.name}_hyphal_video.mp4"

        imageio.mimsave(
            str(hyphal_video_path),
            hyphal_video,
            fps=kymo_extractor.metadata.camera.frame_rate
        )

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
    speed_config: GSTConfig,
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
