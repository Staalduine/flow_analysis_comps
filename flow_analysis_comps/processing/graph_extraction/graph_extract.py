from pathlib import Path
from flow_analysis_comps.data_structs.kymographs import (
    VideoGraphExtraction,
    VideoGraphEdge,
    graphOutput,
    graphExtractConfig,
)
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.processing.graph_extraction.edge_utils import (
    low_pass_filter,
    resample_trail,
)
from flow_analysis_comps.processing.graph_extraction.segmentation_utils import (
    segment_hyphae_w_mean_std,
)
from flow_analysis_comps.processing.graph_extraction.graph_utils import (
    orient,
    skeletonize_segmented_im,
)
import dask.array as da
import numpy as np


class VideoGraphExtractor:
    """
    A class to extract graphs from a video file.
    """

    video_path: Path

    def __init__(self, video_path: Path, extract_properties: graphExtractConfig, user_metadata : videoInfo | None = None):
        self.video_path = video_path
        self.extract_properties = extract_properties
        self.io = videoIO(self.video_path, user_metadata=user_metadata)
        self.video_array: da.Array = self.io.video_array[:20].compute()
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
    def edge_data(self) -> VideoGraphExtraction:
        output = VideoGraphExtraction(io=self.io, edges=[])
        for edge_graph in self.edge_graphs:
            edge_pixels = orient(
                self.graph.graph.get_edge_data(*edge_graph)["pixel_list"],
                self.graph.positions[edge_graph[0]],
            )

            edge_pixels = np.array(edge_pixels)

            # Filter edges on length
            if len(edge_pixels) < self.extract_properties.edge_length_threshold:
                continue

            name = f"edge_{edge_graph[0]}_{edge_graph[1]}"
            output_edge = VideoGraphEdge(
                name=name,
                edge=edge_graph,
                pixel_list=edge_pixels,
            )
            self._smooth_pixel_trail(output_edge)

            output.edges.append(output_edge)
        return output

    @staticmethod
    def _smooth_pixel_trail(edge: VideoGraphEdge):
        edge_pixels = edge.pixel_list
        edge_pixels = resample_trail(low_pass_filter(edge_pixels))
        edge.pixel_list = edge_pixels
