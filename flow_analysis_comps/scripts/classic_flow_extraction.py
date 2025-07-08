from pathlib import Path
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.data_structs.kymographs import (
    GSTConfig,
    graphExtractConfig,
    kymoExtractConfig,
)
from flow_analysis_comps.processing.graph_extraction.graph_extract import (
    VideoGraphExtractor,
)
from flow_analysis_comps.visualizing.GSTSpeeds import GSTSpeedVizualizer
from flow_analysis_comps.visualizing.GraphVisualize import (
    GraphVisualizer,
)
from flow_analysis_comps.processing.GSTSpeedExtract.extract_velocity import kymoAnalyser
from flow_analysis_comps.processing.kymographing.kymographer import KymographExtractor
from flow_analysis_comps.util.video_io import coord_to_folder


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

    out_folder.mkdir(exist_ok=True, parents=True)

    graph_data = VideoGraphExtractor(path, graphExtractConfig()).edge_data
    kymograph_list = KymographExtractor(
        graph_data, kymoExtractConfig()
    ).processed_kymographs
    edge_extraction_fig = GraphVisualizer(graph_data).plot_extraction()
    edge_extraction_fig.savefig(out_folder / "edges_map.png")

    for kymo in kymograph_list:
        kymo_speeds = kymoAnalyser(kymo, speed_config).output_speeds()
        edge_out_folder = out_folder / f"{kymo.name}"
        edge_out_folder.mkdir(exist_ok=True)
        analyser = kymoAnalyser(kymo, speed_config)
        fig, ax = GSTSpeedVizualizer(kymo_speeds).plot_summary(kymo)
        fig.savefig(edge_out_folder / f"{video_position}_{formatted_timestamp}_{kymo.name}_summary.png")
        time_series, averages = analyser.return_summary_frames()
        time_series.to_csv(edge_out_folder / f"{kymo.name}_time_series.csv")
        averages.to_csv(edge_out_folder / f"{kymo.name}_averages.csv")
