from pathlib import Path

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


def process(run_info_index, process_args):
    # expecting a float in um/s
    speed_limit = process_args[0]
    speed_config = GSTConfig(speed_limit=speed_limit)

    row = run_info_index
    path = Path(row["total_path"])

    out_folder = path / "flow_analysis"
    out_folder.mkdir(exist_ok=True)

    graph_data = VideoGraphExtractor(path, graphExtractConfig()).edge_data
    kymograph_list = KymographExtractor(
        graph_data, kymoExtractConfig()
    ).processed_kymographs
    edge_extraction_fig = GraphVisualizer(graph_data).plot_extraction()
    edge_extraction_fig.savefig(out_folder / "edges_map.png")

    for kymo in kymograph_list:
        kymo_speeds = kymoAnalyser(kymograph_list[0], speed_config).output_speeds()
        edge_out_folder = out_folder / f"edge_{kymo.name}"
        edge_out_folder.mkdir(exist_ok=True)
        analyser = kymoAnalyser(kymo, speed_config)
        fig, ax = GSTSpeedVizualizer(kymo_speeds).plot_summary(kymo)
        fig.savefig(edge_out_folder / f"{kymo.name}_summary.png")
        time_series, averages = analyser.return_summary_frames()
        time_series.to_csv(edge_out_folder / f"{kymo.name}_time_series.csv")
        averages.to_csv(edge_out_folder / f"{kymo.name}_averages.csv")
