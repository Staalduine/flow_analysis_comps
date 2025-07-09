from flow_analysis_comps.data_structs.kymographs import KymoCoordinates, VideoGraphEdge
from flow_analysis_comps.util.coord_space_util import extract_perp_lines
import numpy as np


def extract_kymo_coordinates(
    edge: VideoGraphEdge, step, resolution, target_length
) -> KymoCoordinates | None:
    """
    Extracts kymograph coordinates from a VideoGraphEdge.
    """
    start, end, step_size = (
        step,
        len(edge.pixel_list) - step,
        resolution,
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
    orientations = np.array(prev_segment_pixel_list) - np.array(next_segment_pixel_list)

    perpendicular = np.array([orientations[:, 1], -orientations[:, 0]]).T
    perpendicular_norm = (
        (perpendicular.T / np.linalg.norm(perpendicular, axis=1)).T * target_length / 2
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
        extract_perp_lines(segment[0], segment[1])[:target_length]
        for segment in segment_coords
    ]
    edge_perp_lines = np.array(edge_perp_lines)
    kymo_coords = KymoCoordinates(
        segment_coords=segment_coords,
        perp_lines=edge_perp_lines,
        edge_info=edge,
    )
    return kymo_coords
