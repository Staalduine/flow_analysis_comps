from pathlib import Path
import cv2
from flow_analysis_comps.data_structs.kymographs import kymoExtractProperties
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.processing.graph_extraction.graph_extract import (
    VideoGraphExtractor,
)
import imageio
from matplotlib import pyplot as plt
from pydantic import BaseModel
from flow_analysis_comps.data_structs.video_info import videoInfo
from flow_analysis_comps.processing.Classic.model_parameters import KymoDeltas
from flow_analysis_comps.util.coord_space_util import (
    extract_perp_lines,
    validate_interpolation_order,
)
from flow_analysis_comps.util.video_io import (
    load_tif_series_to_dask,
    read_video_metadata,
)
from flow_analysis_comps.processing.video_manipulation.segmentation_methods import (
    _segment_hyphae_w_mean_std,
    skeletonize_segmented_im,
)
from flow_analysis_comps.util.graph_util import orient
import numpy as np
from skimage.measure import profile_line
from scipy import ndimage as ndi
from scipy.signal import butter, sosfiltfilt
import matplotlib.patheffects as path_effects


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

        self.deltas = KymoDeltas(
            delta_t=1 / self.metadata.camera_settings.frame_rate,
            delta_x=1.725
            * 2
            / (self.metadata.magnification)
            * self.metadata.camera_settings.binning
            * self.extract_properties.resolution,
        )

        self.edges = self.exract_graph_edges()

    def exract_graph_edges(self):
        """
        Extracts the edges from the video file.
        """
        graph_extractor = VideoGraphExtractor(
            self.video_path, self.extract_properties.graph_extraction
        )

        return graph_extractor.edge_data
    

