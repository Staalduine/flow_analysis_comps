from matplotlib import pyplot as plt
from pydantic import BaseModel
from data_structs.video_info import videoInfo
from util.video_io import load_tif_series_to_dask, read_video_metadata
from video_manipulation.segment_skel import (
    _segment_hyphae_w_mean_std,
    skeletonize_segmented_im,
)
from util.graph_util import orient, generate_index_along_sequence
import numpy as np
import networkx as nx


class videoControl:
    ## Initiate object
    def __init__(self, video_folder_adr, video_info_adr):
        self.video_info = read_video_metadata(video_info_adr)
        self.array: np.ndarray = load_tif_series_to_dask(
            video_folder_adr
        )  # Dims are t, y, x
        self.time_pixel_size = 1 / self.video_info.camera_settings.frame_rate
        self.space_pixel_size = (
            2
            * 1.725
            / (self.video_info.magnification)
            * self.video_info.camera_settings.binning
        )  # um.pixel

        self._mean_img = None
        self._std_img = None
        self._mask = None
        self._skeleton_graph = None
        self._edge_graphs = None
        self._node_positions = None
        self._edges = None

    ## array properties, gets calculated on a needs basis

    @property
    def mean_img(self):
        "Image stack averaged across time"
        if self._mean_img is None:
            self._mean_img = self.array.mean(axis=0).compute()
        return self._mean_img

    @property
    def std_img(self):
        "Standard deviation of pixels across time"
        if self._std_img is None:
            self._std_img = self.array.std(axis=0).compute()
        return self._std_img

    @property
    def mask(self):
        if self._mask is None:
            self._mask = _segment_hyphae_w_mean_std(
                self.mean_img, self.std_img, seg_thresh=1.15, mode=self.video_info.mode
            )
        return self._mask

    @property
    def skeleton_graph(self):
        if self._skeleton_graph is None:
            self._skeleton_graph, self._node_positions = skeletonize_segmented_im(
                self.mask
            )
        return self._skeleton_graph

    @property
    def node_positions(self):
        if self._node_positions is None:
            self._skeleton_graph, self._node_positions = skeletonize_segmented_im(
                self.mask
            )
        return self._node_positions

    @property
    def edge_graphs(self):
        if self._edge_graphs is None:
            self._edge_graphs = list(self.skeleton_graph.edges)
        return self._edge_graphs

    @property
    def edges(self):
        if self._edges is None:
            self._edges = []
            for edge_graph in self.edge_graphs:
                node_positions = (
                    self.node_positions[edge_graph[0]],
                    self.node_positions[edge_graph[1]],
                )
                edge_pixels = orient(
                    self.skeleton_graph.get_edge_data(*edge_graph)["pixel_list"],
                    self.node_positions[edge_graph[0]],
                )
                self.edges.append(edgeControl(self.video_info, edge_graph, edge_pixels))
        return self._edges

    ## Plotting functions
    def plot_edge_extraction(self):
        image = self.array[0, :, :]

        fig1, ax1 = plt.subplots()
        ax1.imshow(
            image,
            extent=[
                0,
                self.space_pixel_size * image.shape[1],
                self.space_pixel_size * image.shape[0],
                0,
            ],
            cmap="cet_CET_L20",
        )
        ax1.set_xlabel(r"x $(\mu m)$")
        ax1.set_ylabel(r"y $(\mu m)$")


class kymoExtractProperties(BaseModel):
    resolution: int = 1
    step: int = 15
    target_length: int = 120
    bounds: tuple[float, float] = (0.0, 1.0)


class edgeControl:
    """
    Class object controlling data and actions on edges in a video.
    Videos can contain multiple edges, each one containing different properties.
    """

    def __init__(
        self,
        video_info: videoInfo,
        edge_graph: tuple[int, int],
        # node_positions: tuple[int, int],
        pixel_list: list[tuple[int, int]],
    ):
        self.video_info = video_info
        self.edge_info = edge_graph
        # self.offset = int(np.linalg.norm(node_positions[0] - node_positions[1])) // 4
        self.pixel_list = pixel_list

        self.kymo_extract_settings = kymoExtractProperties()

        self._pixel_indices = None

    @property
    def pixel_indices(self):
        if self._pixel_indices is None:
            self._pixel_indices = generate_index_along_sequence(
                len(self.pixel_list),
                self.kymo_extract_settings.resolution,
                # self.offset,
            )
        return self._pixel_indices

    @property
    def segment_coords(self):
        if self._segment_coords is None:
            self._segment_coords = []
            for i in range(
                self.kymo_extract_settings.step,
                len(self.pixel_list) - self.kymo_extract_settings.step,
                self.kymo_extract_settings.resolution,
            ):
                pivot = self.pixel_list[i]
                orientation = np.array(self.pixel_list[i - self.kymo_extract_settings.step]) - np.array(self.pixel_list[i + self.kymo_extract_settings.step])
                perpendicular = (
                    [1, -orientation[0] / orientation[1]] if orientation[1] != 0 else [0, 1]
                )
                perpendicular_norm = np.array(perpendicular) / np.sqrt(
                    perpendicular[0] ** 2 + perpendicular[1] ** 2
                )
                point1 = np.array(pivot) + self.kymo_extract_settings.target_length * perpendicular_norm / 2
                point2 = np.array(pivot) - self.kymo_extract_settings.target_length * perpendicular_norm / 2
                self._segment_coords.append((point1, point2))
        return self._segment_coords
