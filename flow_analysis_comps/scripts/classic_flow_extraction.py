from pathlib import Path

from flow_analysis_comps.data_structs.kymographs import kymoExtractConfig
from flow_analysis_comps.processing.video_manipulation.control_class import videoControl
from flow_analysis_comps.processing.Classic.extract_velocity import kymoAnalyser
from flow_analysis_comps.processing.kymographing.kymographer import KymographCollection


def process(run_info_index, process_args):
    # expecting a float in um/s
    speed_limit = process_args[0]

    row = run_info_index
    path = Path(row["total_path"])
    # video_folder = path / "Img"

    kymographer = KymographCollection(path, kymoExtractConfig())

    kymographs = kymographer.kymographs
    deltas = kymographer.deltas

    # metadata_folder = path / "metadata.json"
    out_folder = path / "flow_analysis"
    # out_folder.mkdir(exist_ok=True)

    video_operator = videoControl(path, metadata_folder, resolution=1)
    kymographs = video_operator.get_kymographs()
    edge_extraction_fig = video_operator.plot_edge_extraction()
    edge_extraction_fig.savefig(out_folder / "edges_map.png")

    for key, kymo in kymographs.items():
        edge_out_folder = out_folder / f"edge_{key}"
        edge_out_folder.mkdir(exist_ok=True)
        analyser = kymoAnalyser(
            kymo, video_deltas=deltas, speed_threshold=speed_limit, name=key
        )
        # analyser.plot_kymo_fields()
        fig, ax = analyser.plot_summary()
        fig.savefig(edge_out_folder / f"{key}_summary.png")
        time_series, averages = analyser.return_summary_frames()
        time_series.to_csv(edge_out_folder / "time_series.csv")
        averages.to_csv(edge_out_folder / "averages.csv")
