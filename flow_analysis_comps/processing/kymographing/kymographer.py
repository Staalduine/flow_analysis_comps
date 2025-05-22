from pathlib import Path
from flow_analysis_comps.data_structs.kymographs import kymoExtractProperties
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.processing.graph_extraction.graph_extract import (
    VideoGraphExtractor,
)
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.processing.Classic.model_parameters import KymoDeltas
from flow_analysis_comps.util.coord_space_util import (
    validate_interpolation_order,
)
import numpy as np
from scipy import ndimage as ndi
from flow_analysis_comps.util.logging import setup_logger


class Kymographer:
    """
    A class to extract kymographs from a video file.
    """

    video_path: Path

    def __init__(self, video_path: Path, extract_properties: kymoExtractProperties):
        self.video_path = video_path
        self.extract_properties = extract_properties
        self.io = videoIO(self.video_path)
        self.video_array = self.io.video_array
        self.metadata: videoInfo = self.io.metadata
        self.logger = setup_logger(name="flow_analysis_comps.kymographer")

        self.deltas = KymoDeltas(
            delta_t=1 / self.metadata.camera_settings.frame_rate,
            delta_x=1.725
            * 2
            / (self.metadata.magnification)
            * self.metadata.camera_settings.binning
            * self.extract_properties.resolution,
        )

        self.logger.info(f"Extracting kymographs from {self.video_path}")
        self.edges = self._extract_graph_edges()
        self.edge_dict = {
            edge.name: edge for edge in self.edges if edge.name is not None
        }
        self.logger.info(f"Extracted {len(self.edges)} edges from {self.video_path}")

        self.edge_coords = self._prepare_coordinates()
        self.logger.info(f"Extracted edge coordinates from {self.video_path}")

    @property
    def hyphal_videos(self) -> dict[str, np.ndarray]:
        hyphal_videos = {}
        video_array: np.ndarray = self.video_array.compute()

        order = validate_interpolation_order(video_array[0].dtype, None)

        for im in video_array:
            for name, edge_coords in self.edge_coords.items():
                edge_coords = np.array(edge_coords)
                perp_lines = np.moveaxis(edge_coords, 2, 0)
                edge_im = ndi.map_coordinates(
                    im,
                    perp_lines,
                    prefilter=order > 1,
                    order=order,
                    mode="reflect",
                    cval=0,
                )

                # Store the kymograph in the dictionary
                hyphal_videos[name] = edge_im
        return hyphal_videos

    @property
    def kymographs(self):
        return {name: video.mean(axis=2) for name, video in self.hyphal_videos.items()}

    def _extract_graph_edges(self):
        """
        Extracts the edges from the video file.
        """
        graph_extractor = VideoGraphExtractor(
            self.video_path, self.extract_properties.graph_extraction
        )

        return graph_extractor.edge_data

    def _prepare_coordinates(self) -> dict:
        edge_coords = {}
        for name, edge in self.edge_dict.items():
            edge_coords[name] = []
            start, end, step_size = (
                self.extract_properties.step,
                len(edge.pixel_list) - self.extract_properties.step,
                self.extract_properties.resolution,
            )
            if end < start:
                continue
            segment_pixel_list = edge.pixel_list[start:end:step_size]
            prev_segment_pixel_list = edge.pixel_list[0 : end - start : step_size]
            next_segment_pixel_list = edge.pixel_list[start * 2 :: step_size]
            orientations = np.array(prev_segment_pixel_list) - np.array(
                next_segment_pixel_list
            )

            perpendicular = np.array([orientations[:, 1], -orientations[:, 0]]).T
            perpendicular_norm = (
                (perpendicular.T / np.linalg.norm(perpendicular, axis=1)).T
                * self.extract_properties.target_length
                / 2
            )

            edge_coords[name] = np.array(
                [
                    segment_pixel_list + perpendicular_norm,
                    segment_pixel_list - perpendicular_norm,
                ]
            )
            edge_coords[name] = np.moveaxis(edge_coords[name], 0, 1)
            edge_coords[name] = np.array(
                [
                    [pivot + perp, pivot - perp]
                    for pivot, perp in zip(segment_pixel_list, perpendicular_norm)
                ]
            )
        return edge_coords
