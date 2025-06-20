from pathlib import Path
import imageio
from flow_analysis_comps.io.video import videoIO
import dask.array as da


def coord_to_folder(x, y, precision=2):
    def fmt(val):
        val = round(val, precision)
        if val < 0:
            prefix = "n"
            val = -val
        else:
            prefix = ""
        return prefix + str(val).replace(".", "_")

    return f"x_{fmt(x)}_y_{fmt(y)}"


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

        if separate_into_positions:
            output_path = Path(self.metadata.storage_path) / "video" / folder_name
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(self.metadata.storage_path)

        writer = imageio.get_writer(
            output_path / "Video.mp4",
            fps=int(self.metadata.camera_settings.frame_rate),
            codec="libx264",
        )
        for frame in video_array:
            writer.append_data(frame)
        writer.close()
