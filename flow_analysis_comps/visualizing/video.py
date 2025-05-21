import imageio
from flow_analysis_comps.io.video import videoIO
import dask.array as da


class videoVisualizer:
    def __init__(self, video_path):
        self.video_info = videoIO(video_path)
        self.metadata = self.video_info.metadata
        self.array: da.Array = self.video_info.video_array

    def save_mp4_video(self):
        video_array = self.array.compute()
        writer = imageio.get_writer(
            self.metadata.storage_path / "Video.mp4",
            fps=int(self.metadata.camera_settings.frame_rate),
            codec="libx264",
        )
        for frame in video_array:
            writer.append_data(frame)
        writer.close()
