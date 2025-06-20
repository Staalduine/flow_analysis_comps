from flow_analysis_comps.visualizing.video import VideoVisualizer
import os


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
    separate_into_positions = process_args[0] if process_args else False
    if "total_path" not in row:
        raise KeyError("The key 'total_path' is missing in the provided run_info_index dictionary.")
    path = row["total_path"]
    rename_json_to_video_metadata(path)

    vizi = VideoVisualizer(path)
    vizi.save_mp4_video(separate_into_positions)
    return


def rename_json_to_video_metadata(path):
    for file in os.listdir(path):
        if file.endswith(".json"):
            old_path = os.path.join(path, file)
            new_path = os.path.join(path, "video_metadata.json")
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")
            break
