from pathlib import Path
import cv2
from matplotlib import pyplot as plt
from pydantic import BaseModel
from data_structs.video_info import videoInfo
from util.coord_transforms import extract_perp_lines, validate_interpolation_order
from util.video_io import load_tif_series_to_dask, read_video_metadata
from video_manipulation.segment_skel import (
    _segment_hyphae_w_mean_std,
    skeletonize_segmented_im,
)
from util.graph_util import orient, generate_index_along_sequence
import numpy as np
import networkx as nx
from skimage.measure import profile_line
from scipy import ndimage as ndi


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
        kymo_extract_settings=None,
    ):
        self.video_info = video_info
        self.edge_info = edge_graph
        # self.offset = int(np.linalg.norm(node_positions[0] - node_positions[1])) // 4
        self.pixel_list = np.array(pixel_list)

        if kymo_extract_settings is None:
            self.kymo_extract_settings = kymoExtractProperties()

        self._pixel_indices = None
        self._segment_coords = None

    @property
    def segment_coords(self):
        if self._segment_coords is None:
            self._segment_coords = []
            start, end, step_size = (
                self.kymo_extract_settings.step,
                len(self.pixel_list) - self.kymo_extract_settings.step,
                self.kymo_extract_settings.resolution,
            )
            if end < start:
                return None
            segment_pixel_list = self.pixel_list[start:end:step_size]
            prev_segment_pixel_list = self.pixel_list[0 : end - start : step_size]
            next_segment_pixel_list = self.pixel_list[start * 2 :: step_size]
            orientations = prev_segment_pixel_list - next_segment_pixel_list

            perpendicular = np.array([orientations[:, 1], -orientations[:, 0]]).T
            perpendicular_norm = (
                (perpendicular.T / np.linalg.norm(perpendicular, axis=1)).T
                * self.kymo_extract_settings.target_length
                / 2
            )

            self._segment_coords = np.array(
                [
                    segment_pixel_list + perpendicular_norm,
                    segment_pixel_list - perpendicular_norm,
                ]
            )
            self._segment_coords = np.moveaxis(self._segment_coords, 0, 1)
            self._segment_coords = np.array(
                [
                    [pivot + perp, pivot - perp]
                    for pivot, perp in zip(segment_pixel_list, perpendicular_norm)
                ]
            )
        return self._segment_coords

    def extract_edge_image(self, image):
        edge_image = [
            profile_line(image, segment[0], segment[1], mode="constant")
            for segment in self.segment_coords
        ]
        len_min = np.min([len(line) for line in edge_image])
        edge_image = [line[:len_min] for line in edge_image][::-1]
        return np.array(edge_image)

    def plot_segments(self, adjust_val, ax):
        for point_1, point_2 in self.segment_coords:
            ax.plot(
                [point_1[1] * adjust_val, point_2[1] * adjust_val],
                [point_1[0] * adjust_val, point_2[0] * adjust_val],
                color="white",
                alpha=0.1,
            )


class videoControl:
    ## Initiate object
    def __init__(self, video_folder_adr, video_info_adr, edge_length_threshold = 200):
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
        self.edge_len_thresh = edge_length_threshold

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
    def edges(self) -> list[edgeControl]:
        if self._edges is None:
            self._edges = []
            for edge_graph in self.edge_graphs:
                edge_pixels = orient(
                    self.skeleton_graph.get_edge_data(*edge_graph)["pixel_list"],
                    self.node_positions[edge_graph[0]],
                )
                if len(edge_pixels) > self.edge_len_thresh:
                    self.edges.append(edgeControl(self.video_info, edge_graph, edge_pixels))
        return self._edges

    def get_edge_images(self) -> dict[str, np.ndarray]:
        """
        Creates videos of each found edge in the video.
        Edge video axes are [frame, length, width] along the hyphae.

        Returns:
            dict[str, np.ndarray]: Dictionary with edge names and 3d video arrays
        """

        edge_images = {}
        edge_coords = {}

        if "compute" in dir(self.array):
            self.array = self.array.compute()

        for edge in self.edges:
            edge_name = str(edge.edge_info)

            # Get coordinates for each line, truncate to target length
            edge_perp_lines = [
                extract_perp_lines(segment[0], segment[1])[
                    : edge.kymo_extract_settings.target_length
                ]
                for segment in edge.segment_coords
            ]  # dims are [length, width, (x, y)]

            edge_coords[edge_name] = np.array(edge_perp_lines)

        # Set up dict indices
        for key in edge_coords.keys():
            edge_images[key] = []

        order = validate_interpolation_order(self.array[0].dtype, None)

        # Load image, take data, add data to arrays
        for im in self.array:
            for key, perp_lines in edge_coords.items():
                # Rearrange axes for map_coordinates to:
                # [coord, width, length]
                perp_lines = np.moveaxis(perp_lines, 2, 0)
                edge_im = ndi.map_coordinates(
                    im,
                    perp_lines,
                    prefilter=order > 1,
                    order=order,
                    mode="reflect",
                    cval=0.0,
                )
                edge_images[key].append(edge_im)
        
        for key, video in edge_images.items():
            edge_images[key] = np.array(video)
        return edge_images
    
    def save_edge_videos(self, out_adr_folder:Path):
        videos_dict = self.get_edge_images()
        for title, array in videos_dict.items():
            filename = out_adr_folder / (title[1:-1] + ".mp4")
            t_max, x_max, y_max = array.shape
            min_val, max_val = np.min(array), np.max(array)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.video_info.camera_settings.frame_rate, (y_max, x_max))
            
            for t in range(t_max):
                frame = array[t]
                frame = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                frame_colored = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
                out.write(frame_colored)
            
            out.release()
            print(f'Video saved as {filename}')


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
        ax1.set_xlabel(r'x $\mu m$')
        ax1.set_ylabel(r'y $\mu m$')

        for edge in self.edges:
            edge.plot_segments(self.space_pixel_size, ax1)


class kymoExtractProperties(BaseModel):
    resolution: int = 1
    step: int = 15
    target_length: int = 70
    bounds: tuple[float, float] = (0.0, 1.0)
