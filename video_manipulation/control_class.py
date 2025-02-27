from matplotlib import pyplot as plt
from util.video_io import load_tif_series_to_dask, read_video_metadata
from video_manipulation.segment_skel import _segment_hyphae_w_mean_std, skeletonize_segmented_im
import numpy.typing as npt
import numpy as np

class videoControl:

    ## Initiate object
    def __init__(self, video_folder_adr, video_info_adr):
        self.video_info = read_video_metadata(video_info_adr)
        self.array: np.ndarray = load_tif_series_to_dask(video_folder_adr)  # Dims are t, y, x
        self.time_pixel_size = 1 / self.video_info.camera_settings.frame_rate
        self.space_pixel_size = (
            2 * 1.725 / (self.video_info.magnification) * self.video_info.camera_settings.binning
        )  # um.pixel

        self._mean_img = None
        self._std_img = None
        self._mask = None
        self._skeleton_graph = None
        self._edge_graphs = None
        self._node_positions = None

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
            self._mask = _segment_hyphae_w_mean_std(self.mean_img, self.std_img, seg_thresh=1.15, mode=self.video_info.mode)
        return self._mask
    
    @property
    def skeleton_graph(self):
        if self._skeleton_graph is None:
            self._skeleton_graph, self._node_positions = skeletonize_segmented_im(self.mask)
        return self._skeleton_graph
    
    @property
    def node_positions(self):
        if self._node_positions is None:
            self._skeleton_graph, self._node_positions = skeletonize_segmented_im(self.mask)
        return self._node_positions
    
    @property
    def edge_graphs(self):
        if self._edge_graphs is None:
            self._edge_graphs = list(self.skeleton_graph.edges)
        return self._edge_graphs
    
    ## Plotting functions
    def plot_edge_extraction(self):
        image = self.array[0,:,:]

        fig1, ax1 = plt.subplots()
        ax1.imshow(
            image,
            extent=[
                0,
                self.space_pixel_size * image.shape[1],
                self.space_pixel_size * image.shape[0],
                0,
            ],
            cmap = "cet_CET_L20"
        )
        ax1.set_xlabel(r"x $(\mu m)$")
        ax1.set_ylabel(r"y $(\mu m)$")