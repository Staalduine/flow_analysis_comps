from pathlib import Path
from flow_analysis_comps.data_structs.kymographs import (
    KymoCoordinates,
    VideoGraphExtraction,
    VideoGraphEdge,
    kymoExtractConfig,
)
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.processing.graph_extraction.edge_utils import (
    low_pass_filter,
    resample_trail,
)
from flow_analysis_comps.processing.graph_extraction.graph_extract import (
    VideoGraphExtractor,
)
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.processing.Classic.model_parameters import KymoDeltas
from flow_analysis_comps.util.coord_space_util import (
    extract_perp_lines,
    validate_interpolation_order,
)
import numpy as np
from scipy import ndimage as ndi
from flow_analysis_comps.util.logging import setup_logger


class Kymograph:
    def __init__(self, kymodata) -> None:
        pass


class KymographCollection:
    """
    A class to extract kymographs from a video file.
    """

    video_path: Path

    def __init__(
        self,
        graph_extraction: VideoGraphExtraction,
        extract_properties: kymoExtractConfig,
    ):
        self.extract_properties = extract_properties
        self.io = graph_extraction.io
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

        self.edges = graph_extraction.edges

        self.edge_coords = self._prepare_coordinates()
        self.logger.info(f"Extracted edge coordinates from {self.video_path}")

    def _preprocess_pixel_trails(self):
        for edge in self.edges:
            edge_pixels = edge.pixel_list
            edge_pixels = resample_trail(low_pass_filter(edge_pixels))
            edge.pixel_list = edge_pixels

    @property
    def hyphal_videos(self) -> dict[str, np.ndarray]:
        hyphal_videos = {edge.name: [] for edge in self.edges}
        video_array: np.ndarray = self.video_array.compute()

        order = validate_interpolation_order(video_array[0].dtype, None)

        for im in video_array:
            for edge in self.edge_coords:
                edge_coords = np.array(edge.perp_lines)
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
                hyphal_videos[edge.edge_info.name].append(edge_im)
        hyphal_videos_np = {
            name: np.array(edge_im) for name, edge_im in hyphal_videos.items()
        }

        return hyphal_videos_np

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

    def _prepare_coordinates(self) -> list[KymoCoordinates]:
        edge_coords = []
        for edge in self.edges:
            # smooth the edge pixel list

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

            segment_coords = np.array(
                [
                    segment_pixel_list + perpendicular_norm,
                    segment_pixel_list - perpendicular_norm,
                ]
            )
            segment_coords = np.moveaxis(segment_coords, 0, 1)
            segment_coords = np.array(
                [
                    [pivot + perp, pivot - perp]
                    for pivot, perp in zip(segment_pixel_list, perpendicular_norm)
                ]
            )
            edge_perp_lines = [
                extract_perp_lines(segment[0], segment[1])[
                    : self.extract_properties.target_length
                ]
                for segment in segment_coords
            ]
            edge_perp_lines = np.array(edge_perp_lines)
            kymo_coords = KymoCoordinates(
                segment_coords=segment_coords,
                perp_lines=edge_perp_lines,
                edge_info=edge,
            )
            edge_coords.append(kymo_coords)
        return edge_coords
