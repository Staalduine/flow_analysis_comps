from flow_analysis_comps.data_structs.process_configs import kymoExtractConfig
from flow_analysis_comps.data_structs.kymographs import (
    KymoCoordinates,
    VideoGraphExtraction,
    kymoOutputs,
)
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.processing.GSTSpeedExtract.classic_image_util import filter_kymo_right
from flow_analysis_comps.processing.kymographing.kymo_utils import (
    extract_kymo_coordinates,
)
from flow_analysis_comps.util.coord_space_util import (
    validate_interpolation_order,
)
import numpy as np
from scipy import ndimage as ndi
from flow_analysis_comps.util.logging import setup_logger
import dask.array as da
import flow_analysis_comps.io as io


def extract_kymographs_from_video(
    video_metadata: videoInfo,
    extracted_graph: VideoGraphExtraction,
    extract_params: kymoExtractConfig,
) -> list[kymoOutputs]:
    """
    Extracts kymographs from a video file using the specified extraction parameters.
    """
    extractor = KymographExtractor(video_metadata, extracted_graph, extract_params)
    return extractor.processed_kymographs


class KymographExtractor:
    """
    A class to extract kymographs from a video file.
    """

    def __init__(
        self,
        video_metadata: videoInfo,
        graph_extraction: VideoGraphExtraction,
        extract_params: kymoExtractConfig,
    ):
        self.metadata = video_metadata
        self.extract_properties = extract_params
        self.io = graph_extraction.io
        self.video_array: da.Array = io.read_video_array(self.metadata)
        self.logger = setup_logger(name="flow_analysis_comps.kymographer")

        self.deltas = self.metadata.deltas.model_copy()
        self.deltas.delta_x *= self.extract_properties.resolution

        self.edges = graph_extraction.edges

        self.edge_coords = self._prepare_coordinates()
        self.logger.info(f"Extracted edge coordinates from {self.metadata.root_path}")

    @property
    def hyphal_videos(self) -> dict[str, np.ndarray]:
        hyphal_videos = {edge.name: [] for edge in self.edges}
        video_array: np.ndarray = self.video_array.compute()

        order = validate_interpolation_order(video_array[0].dtype, None)
        order = 3

        for im in video_array:
            for edge in self.edge_coords:
                edge_coords = np.array(edge.perp_lines)
                perp_lines = np.moveaxis(edge_coords, 2, 0)
                edge_im = ndi.map_coordinates(
                    im,
                    perp_lines,
                    prefilter=order > 1,
                    order=order,
                    mode="constant",
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
            # decomposed_kymo = self._decompose_kymograph(kymo)
            processed_kymo = kymoOutputs(
                deltas=self.deltas,
                name=name,
                kymograph=kymo,
            )
            processed_kymographs.append(processed_kymo)

        return processed_kymographs

    def _prepare_coordinates(self) -> list[KymoCoordinates]:
        # Prepares kymograph coordinates from the edges of the video graph.
        edge_coords = []
        for edge in self.edges:
            kymo_coords = extract_kymo_coordinates(
                edge,
                self.extract_properties.step,
                self.extract_properties.resolution,
                self.extract_properties.target_length,
            )
            if kymo_coords is not None:
                edge_coords.append(kymo_coords)
        return edge_coords
