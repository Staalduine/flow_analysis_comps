from pathlib import Path
from flow_analysis_comps.data_structs.kymographs import (
    edgeOutput,
    graphOutput,
    graphExtractConfig,
)
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.processing.graph_extraction.segmentation_utils import (
    segment_hyphae_w_mean_std,
)
from flow_analysis_comps.processing.graph_extraction.graph_utils import orient, skeletonize_segmented_im
import dask.array as da


class VideoGraphExtractor:
    """
    A class to extract graphs from a video file.
    """

    video_path: Path

    def __init__(self, video_path: Path, extract_properties: graphExtractConfig):
        self.video_path = video_path
        self.extract_properties = extract_properties
        self.io = videoIO(self.video_path)
        self.video_array: da.Array = self.io.video_array
        self.metadata: videoInfo = self.io.metadata

    @property
    def mean_img(self):
        return self.video_array.mean(axis=0)

    @property
    def std_img(self):
        return self.video_array.std(axis=0)

    @property
    def mask(self):
        return segment_hyphae_w_mean_std(
            self.mean_img,
            self.std_img,
            self.extract_properties.segmentation_threshold,
            self.metadata.mode,
        )

    @property
    def graph(self):
        graph, positions = skeletonize_segmented_im(self.mask)
        graph_output = graphOutput(
            graph=graph,
            positions=positions,
        )
        return graph_output

    @property
    def edge_graphs(self) -> list[tuple[int, int]]:
        return list(self.graph.graph.edges)

    @property
    def edge_data(self) -> list[edgeOutput]:
        edges = []
        for edge_graph in self.edge_graphs:
            edge_pixels = orient(
                self.graph.graph.get_edge_data(*edge_graph)["pixel_list"],
                self.graph.positions[edge_graph[0]],
            )

            if len(edge_pixels) < self.extract_properties.edge_length_threshold:
                continue

            name = f"edge_{edge_graph[0]}_{edge_graph[1]}"

            edges.append(edgeOutput(name=name, edge=edge_graph, pixel_list=edge_pixels))
        return edges
