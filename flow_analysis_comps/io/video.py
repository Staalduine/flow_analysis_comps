from datetime import date, datetime, timedelta
import json
from pathlib import Path
import dask.array as da
from dask.delayed import delayed
import tifffile
import pandas as pd

from flow_analysis_comps.data_structs.video_info import (
    cameraPosition,
    cameraSettings,
    videoInfo,
    videoMode,
)
from flow_analysis_comps.data_structs.plate_info import (
    plateInfo,
    rootTypes,
    strainTypes,
    treatmentTypes,
)


class videoIO:
    def __init__(self, root_folder):
        self.root_folder = Path(root_folder)
        self.metadata_file_path = self._find_metadata()
        self.metadata: videoInfo = self._read_video_metadata()
        self.video_array: da.Array = self._load_tif_series_to_dask()

    def _find_metadata(self):
        metadata_file_path = next(self.root_folder.glob("*.txt"), None)
        if metadata_file_path is None:
            metadata_file_path = next(self.root_folder.glob("*.json"), None)
        if metadata_file_path is None:
            raise ValueError(
                f"No metadata file found in {self.root_folder}. Expected .txt or .json file."
            )
        return metadata_file_path

    def _read_video_metadata(self) -> videoInfo:
        match self.metadata_file_path.suffix:
            case ".txt":
                return self._read_video_info_txt()
            case ".json":
                return self._read_video_info_json()
            case _:
                raise ValueError(
                    f"Unsupported metadata file format: {self.metadata_file_path.suffix}. Expected .txt or .json."
                )

    def _read_video_info_json(self) -> videoInfo:
        with open(str(self.metadata_file_path), encoding="utf-8-sig") as json_data:
            # print(json_data)
            json_data.seek(0)
            video_json = json.load(json_data)
            if "metadata" in video_json:
                video_json = video_json["metadata"]

        if video_json["camera"]["intensity"][0] == 0:
            image_mode = "fluorescence"
        elif video_json["camera"]["intensity"][1] == 0:
            image_mode = "brightfield"
        else:
            image_mode = "brightfield"

        position = cameraPosition(
            x=video_json["video"]["location"][0],
            y=video_json["video"]["location"][1],
            z=video_json["video"]["location"][2],
        )

        camera_settings = cameraSettings(
            model=video_json["camera"]["model"],
            exposure_us=video_json["camera"]["exposure_time"] * 1e6,
            frame_rate=video_json["camera"]["frame_rate"],
            frame_size=video_json["camera"]["frame_size"],
            binning=video_json["camera"]["binning"].split("x")[0],
            gain=video_json["camera"]["gain"],
            gamma=video_json["camera"]["gamma"],
        )

        info_obj = videoInfo(
            duration=video_json["video"]["duration"],
            frame_nr=int(
                video_json["video"]["duration"] * video_json["camera"]["frame_rate"]
            ),
            mode=videoMode(image_mode),
            magnification=50.0,
            position=position,
            camera_settings=camera_settings,
            storage_path=self.root_folder,
        )

        return info_obj

    def _read_video_info_txt(self) -> videoInfo:
        raw_data = pd.read_csv(
            self.metadata_file_path,
            sep=": ",
            engine="python",
            header=0,
            names=["Info"],
            index_col=0,
        )["Info"]
        # Drop all columns with no data
        raw_data = raw_data.dropna(how="all")
        for col in raw_data.index:
            raw_data[col] = raw_data[col].strip()
        time_info = " ".join(raw_data["DateTime"].split(", ")[1:])
        time_obj = datetime.strptime(time_info, "%d %B %Y %X")
        crossing_date = date.fromisoformat(raw_data["CrossDate"])

        plate_info_obj = plateInfo(
            plate_nr=raw_data["Plate"],
            root=rootTypes(raw_data["Root"]),
            strain=strainTypes(raw_data["Strain"]),
            treatment=treatmentTypes(raw_data["Treatment"]),
            crossing_date=crossing_date,
        )

        camera_settings = cameraSettings(
            model=raw_data["Model"],
            exposure_us=float(raw_data["ExposureTime"].split(" ")[0]),
            frame_rate=float(raw_data["FrameRate"].split(" ")[0]),
            frame_size=(
                int(raw_data["FrameSize"].split(" ")[0].split("x")[0]),
                int(raw_data["FrameSize"].split(" ")[0].split("x")[1]),
            ),
            binning=int(raw_data["Binning"].split("x")[0]),
            gain=float(raw_data["Gain"]),
            gamma=float(raw_data["Gamma"]),
        )

        position = cameraPosition(
            x=float(raw_data["X"].split(" ")[0]),
            y=float(raw_data["Y"].split(" ")[0]),
            z=float(raw_data["Z"].split(" ")[0]),
        )

        info_obj = videoInfo(
            plate_info=plate_info_obj,
            date_time=time_obj,
            storage_path=self.root_folder,
            run_nr=int(raw_data["Run"]),
            duration=timedelta(seconds=int(raw_data["Time"].strip().split(" ")[0])),
            frame_nr=int(raw_data["Frames Recorded"].strip().split("/")[0]),
            mode=videoMode(raw_data["Operation"].strip().split(" ")[1].lower()),
            magnification=float(raw_data["Operation"].strip().split()[0][:-1]),
            camera_settings=camera_settings,
            position=position,
        )
        return info_obj

    def _load_tif_series_to_dask(self) -> da.Array:
        """
        Loads a series of .tif images from a folder into a Dask array.

        Parameters:
            folder_path (str or Path): Path to the folder containing the .tif images.

        Returns:
            dask.array.Array: A Dask array representing the .tif series.
        """
        folder = Path(self.root_folder) / "Img"
        tif_files = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in [".tif", ".tiff"]]
        )

        if not tif_files:
            raise ValueError("No .tif files found in the specified folder.")

        sample_image = tifffile.imread(str(tif_files[0]))
        dtype = sample_image.dtype

        def lazy_reader(filename):
            return tifffile.imread(str(filename))

        dask_array = da.stack(
            [
                da.from_delayed(
                    delayed(lazy_reader)(file), shape=sample_image.shape, dtype=dtype
                )
                for file in tif_files
            ]
        )

        return dask_array

    def get_deltas(self) -> tuple[float, float]:
        """
        Returns the time and spatial deltas for the video.
        """
        delta_t = 1 / self.metadata.camera_settings.frame_rate
        delta_x = (
            1.725
            * 2
            / self.metadata.magnification
            * self.metadata.camera_settings.binning
        )
        return delta_x, delta_t
