from pathlib import Path
import imageio
from flow_analysis_comps.io.video import videoIO
from flow_analysis_comps.util.video_io import coord_to_folder
import dask.array as da

class VideoVisualizer:
    def __init__(self, video_path):
        self.video_info = videoIO(video_path)
        self.metadata = self.video_info.metadata
        self.array: da.Array = self.video_info.video_array

    def save_mp4_video(self, separate_into_positions=False):
        video_array = self.array.compute()

        folder_name = coord_to_folder(
            self.metadata.position.x, self.metadata.position.y, precision=3
        )

        timestamp_str = self.metadata.date_time
        # timestamp_obj = datetime.fromisoformat(timestamp_str)
        timeformat = "%Y%m%d_%H%M%S"
        formatted_timestamp = timestamp_str.strftime(timeformat)

        if separate_into_positions:
            output_path = Path(self.metadata.storage_path) / "video" / folder_name
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(self.metadata.storage_path)

        writer = imageio.get_writer(
            output_path / f"{formatted_timestamp}_Video.mp4",
            fps=int(self.metadata.camera_settings.frame_rate),
            codec="libx264",
        )
        for frame in video_array:
            writer.append_data(frame)
        writer.close()
