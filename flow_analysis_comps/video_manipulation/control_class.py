from pathlib import Path
import cv2
from matplotlib import pyplot as plt
from pydantic import BaseModel
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.util.coord_space_util import (
    extract_perp_lines,
    validate_interpolation_order,
)
from flow_analysis_comps.util.video_io import (
    load_tif_series_to_dask,
    read_video_metadata,
)
from flow_analysis_comps.video_manipulation.segmentation_methods import (
    _segment_hyphae_w_mean_std,
    skeletonize_segmented_im,
)
from flow_analysis_comps.util.graph_util import orient
import numpy as np
from skimage.measure import profile_line
from scipy import ndimage as ndi
from scipy.signal import butter, sosfiltfilt


def low_pass_filter(coords, cutoff_freq=0.01, order=2):
    """
    Applies a low-pass Butterworth filter to (x, y) coordinates.

    Parameters:
    - coords: np.ndarray of shape (N, 2), where N is the number of points [(x1, y1), (x2, y2), ...]
    - cutoff_freq: Cutoff frequency (0 < f < 0.5, relative to Nyquist frequency)
    - order: Order of the Butterworth filter (higher means sharper cutoff)

    Returns:
    - Filtered np.ndarray of shape (N, 2)
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "Input must be an array of shape (N, 2) representing (x, y) coordinates."
        )

    # Create Butterworth low-pass filter
    b = butter(N=order, Wn=cutoff_freq, btype="lowpass", analog=False, output="sos")

    # Apply the filter to both x and y coordinates separately
    x_filtered = sosfiltfilt(b, coords[:, 0], padlen=100)
    y_filtered = sosfiltfilt(b, coords[:, 1], padlen=100)

    return np.column_stack((x_filtered, y_filtered))


def resample_trail(trail):
    trail = np.array(trail)  # Ensure it's an array
    distances = np.sqrt(np.sum(np.diff(trail, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(
        np.cumsum(distances), 0, 0
    )  # Insert 0 at the start

    new_distances = np.arange(
        0, cumulative_distances[-1], 1
    )  # New samples at distance 1
    new_trail = np.array(
        [
            np.interp(new_distances, cumulative_distances, trail[:, dim])
            for dim in range(trail.shape[1])
        ]
    ).T  # Interpolate each coordinate separately

    return new_trail  # Ensure integer pixel coordinates


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
        self.pixel_list = np.array(pixel_list)
        self.pixel_list = resample_trail(low_pass_filter(self.pixel_list))

        if kymo_extract_settings is None:
            self.kymo_extract_settings = kymoExtractProperties()
        else:
            self.kymo_extract_settings = kymo_extract_settings

        self._pixel_indices = None
        self._segment_coords = None

        # These values get assigned by video object
        self.edge_coords = None
        self.edge_video: list | np.ndarray = []

    @property
    def kymograph(self):
        if len(self.edge_video) > 0:
            return self.edge_video.mean(axis=2)
        else:
            raise InterruptedError(
                "Attempted to get kymograhps without edge extraction. Please get kymographs using the .get_kymographs() function in the video object."
            )

    @property
    def segment_coords(self):
        if self._segment_coords is not None:
            return self._segment_coords

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
    def __init__(
        self,
        video_folder_adr,
        video_info_adr,
        edge_length_threshold=200,
        resolution=1,
        video_folder_add="Img",
    ):
        self.video_info = read_video_metadata(video_info_adr)
        self.array: np.ndarray = load_tif_series_to_dask(
            video_folder_adr / video_folder_add
        )  # Dims are t, y, x
        self.time_pixel_size = 1 / self.video_info.camera_settings.frame_rate
        self.space_pixel_size = (
            1.725
            * 2
            / (self.video_info.magnification)
            * self.video_info.camera_settings.binning
            * resolution
        )  # um/pixel
        self.edge_len_thresh = edge_length_threshold
        self.kymo_extract_settings = kymoExtractProperties(resolution=resolution)

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
            self._mean_img = self.array[:20].mean(axis=0).compute()
        return self._mean_img

    @property
    def std_img(self):
        "Standard deviation of pixels across time"
        if self._std_img is None:
            self._std_img = self.array[:20].std(axis=0).compute()
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
            self.get_edge_objects_from_graphs()
        return self._edges

    def get_edge_objects_from_graphs(self):
        self._edges = []
        for edge_graph in self.edge_graphs:
            edge_pixels = orient(
                self.skeleton_graph.get_edge_data(*edge_graph)["pixel_list"],
                self.node_positions[edge_graph[0]],
            )

            if len(edge_pixels) < self.edge_len_thresh:
                continue

            edge_obj = edgeControl(
                self.video_info,
                edge_graph,
                edge_pixels,
                self.kymo_extract_settings,
            )

            self._edges.append(edge_obj)

    def get_edge_images(self) -> dict[str, np.ndarray]:
        """
        Creates videos of each found edge in the video.
        Edge video axes are [frame, length, width] along the hyphae.

        Returns:
            dict[str, np.ndarray]: Dictionary with edge names and 3d video arrays
        """

        edge_images = {}

        # if "compute" in dir(self.array):
        #     self.array = self.array.compute()

        # Number of edges is assumed small (<20)
        for edge in self.edges:
            # Get coordinates for each line, truncate to target length
            edge_perp_lines = [
                extract_perp_lines(segment[0], segment[1])[
                    : edge.kymo_extract_settings.target_length
                ]
                for segment in edge.segment_coords
            ]  # dims are [length, width, (x, y)]

            edge.edge_coords = np.array(edge_perp_lines)

        order = validate_interpolation_order(self.array[0].dtype, None)

        # Load image, take data, add data to arrays
        # Best to do this in video object, so it only has to iterate over image stack once
        for im in self.array:
            for edge in self.edges:
                perp_lines = np.moveaxis(edge.edge_coords, 2, 0)
                edge_im = ndi.map_coordinates(
                    im,
                    perp_lines,
                    prefilter=order > 1,
                    order=order,
                    mode="reflect",
                    cval=0.0,
                )
                edge.edge_video.append(edge_im)

        # TODO: Redo this with initializing the array first. No appends needed.
        for edge in self.edges:
            edge.edge_video = np.array(edge.edge_video)
        return edge_images

    def get_kymographs(self):
        if len(self.edges) == 0:
            raise ValueError("No edges found")

        if len(self.edges[0].edge_video) == 0:
            self.get_edge_images()

        kymographs = {}
        for edge in self.edges:
            kymographs[edge.edge_info] = edge.kymograph
        return kymographs

    def save_edge_videos(self, out_adr_folder: Path):
        videos_dict = self.get_edge_images()
        for title, array in videos_dict.items():
            filename = out_adr_folder / (title[1:-1] + ".mp4")
            t_max, x_max, y_max = array.shape
            min_val, max_val = np.min(array), np.max(array)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                filename,
                fourcc,
                self.video_info.camera_settings.frame_rate,
                (y_max, x_max),
            )

            for t in range(t_max):
                frame = array[t]
                frame = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                frame_colored = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
                out.write(frame_colored)

            out.release()
            print(f"Video saved as {filename}")

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
        ax1.set_xlabel(r"x $\mu m$")
        ax1.set_ylabel(r"y $\mu m$")

        for edge in self.edges:
            edge.plot_segments(self.space_pixel_size, ax1)


class kymoExtractProperties(BaseModel):
    resolution: int = 1
    step: int = 15
    target_length: int = 70
    bounds: tuple[float, float] = (0.0, 1.0)
