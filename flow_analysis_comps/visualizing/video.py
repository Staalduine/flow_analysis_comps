import imageio
from flow_analysis_comps.data_structs.video_info import videoInfo


class videoVisualizer:
    def __init__(self, video_info: videoInfo):
        self.video_info: videoInfo = video_info
        self.video_array = 

    def save_mp4_video(self):
        video_array = self.array.compute()
        writer = imageio.get_writer(
            self.root_folder / "Video.mp4",
            fps=int(self.video_info.camera_settings.frame_rate),
            codec="libx264",
        )
        for frame in video_array:
            writer.append_data(frame)
        writer.close()