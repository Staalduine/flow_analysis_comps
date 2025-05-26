from flow_analysis_comps.data_structs.kymographs import (
    KymoCoordinates,
    VideoGraphEdge,
    VideoGraphExtraction,
    kymoDeltas,
    kymoExtractConfig,
    kymoOutputs,
)
from flow_analysis_comps.processing.Classic.classic_image_util import filter_kymo_right
from flow_analysis_comps.processing.graph_extraction.edge_utils import (
    low_pass_filter,
    resample_trail,
)
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.util.coord_space_util import (
    extract_perp_lines,
    validate_interpolation_order,
)
import numpy as np
from scipy import ndimage as ndi
from flow_analysis_comps.util.logging import setup_logger
import dask.array as da


class Kymograph:
    def __init__(self, kymodata) -> None:
        pass


class KymographExtractor:
    """
    A class to extract kymographs from a video file.
    """

    def __init__(
        self,
        graph_extraction: VideoGraphExtraction,
        extract_properties: kymoExtractConfig,
    ):
        self.extract_properties = extract_properties
        self.io = graph_extraction.io
        self.video_array: da.Array = self.io.video_array
        self.metadata: videoInfo = self.io.metadata
        self.logger = setup_logger(name="flow_analysis_comps.kymographer")

        self.deltas = kymoDeltas(
            delta_t=1 / self.metadata.camera_settings.frame_rate,
            delta_x=1.725
            * 2
            / (self.metadata.magnification)
            * self.metadata.camera_settings.binning
            * self.extract_properties.resolution,
        )

        self.edges = graph_extraction.edges

        self.edge_coords = self._prepare_coordinates()
        self.logger.info(f"Extracted edge coordinates from {self.io.root_folder}")

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
    def simple_kymographs(self):
        return {name: video.mean(axis=2) for name, video in self.hyphal_videos.items()}

    @staticmethod
    def _decompose_kymograph(kymograph: np.ndarray) -> np.ndarray:
        kymo_filtered_left = filter_kymo_right(kymograph)
        kymo_filtered_right = np.flip(
            filter_kymo_right(np.flip(kymograph, axis=1)), axis=1
        )
        out = np.array([kymo_filtered_left, kymo_filtered_right])
        return out

    @property
    def processed_kymographs(self) -> list[kymoOutputs]:
        """
        Returns a list of kymographs with their deltas and decomposed directions.
        """
        kymographs = self.simple_kymographs
        processed_kymographs = []

        for name, kymo in kymographs.items():
            decomposed_kymo = self._decompose_kymograph(kymo)
            processed_kymo = kymoOutputs(
                deltas=self.deltas,
                name=name,
                kymograph=kymo,
                kymo_left=decomposed_kymo[0],
                kymo_right=decomposed_kymo[1],
                kymo_no_static=decomposed_kymo[0] + decomposed_kymo[1],
            )
            processed_kymographs.append(processed_kymo)

        return processed_kymographs

    def _prepare_coordinates(self) -> list[KymoCoordinates]:
        # Prepares kymograph coordinates from the edges of the video graph.
        edge_coords = []
        for edge in self.edges:
            kymo_coords = self._extract_kymo_coordinates(edge)
            if kymo_coords is not None:
                edge_coords.append(kymo_coords)
        return edge_coords

    def _extract_kymo_coordinates(self, edge: VideoGraphEdge) -> KymoCoordinates | None:
        """
        Extracts kymograph coordinates from a VideoGraphEdge.
        """
        start, end, step_size = (
            self.extract_properties.step,
            len(edge.pixel_list) - self.extract_properties.step,
            self.extract_properties.resolution,
        )
        if end < start:
            return None
        segment_pixel_list = edge.pixel_list[start:end:step_size]
        prev_segment_pixel_list = edge.pixel_list[
            0 : ((end - start) // step_size) * step_size : step_size
        ]
        next_segment_pixel_list = (
            edge.pixel_list[start * 2 :: step_size]
            if start * 2 < len(edge.pixel_list)
            else []
        )
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
        return kymo_coords
