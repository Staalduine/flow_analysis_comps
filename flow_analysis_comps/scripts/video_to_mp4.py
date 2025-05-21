from flow_analysis_comps.visualizing.video import VideoVisualizer


def process(run_info_index, process_args=None):
    """
    Processes a video file by converting it to MP4 format.

    Parameters:
    run_info_index (dict): A dictionary containing information about the video file, 
                           including the "total_path" key for the file path.

    Returns:
    None
    """
    row = run_info_index
    if "total_path" not in row:
        raise KeyError("The key 'total_path' is missing in the provided run_info_index dictionary.")
    path = row["total_path"]

    vizi = VideoVisualizer(path)
    vizi.save_mp4_video()
    return