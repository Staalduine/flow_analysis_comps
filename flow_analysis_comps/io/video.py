from datetime import datetime, timedelta
import json
from pathlib import Path
import dask.array as da
from dask import delayed
import tifffile
import numpy.typing as npt

import pandas as pd

from flow_analysis_comps.data_structs.video_info import (
    cameraPosition,
    cameraSettings,
    videoInfo,
)
from flow_analysis_comps.flow_analysis_comps.data_structs.plate_info import plateInfo


class videoIO:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.metadata = self._read_video_metadata()
        self.video_array = self._load_tif_series_to_dask()

    def _read_video_metadata(self) -> videoInfo:
        match self.video_path.suffix:
            case ".txt":
                return self._read_video_info_txt()
            case ".json":
                return self._read_video_info_json()

    def _read_video_info_json(self) -> videoInfo:
        with open(str(self.video_path), encoding="utf-8-sig") as json_data:
            # print(json_data)
            json_data.seek(0)
            video_json = json.load(json_data)["metadata"]

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
            mode=image_mode,
            magnification=50.0,
            position=position,
            camera_settings=camera_settings,
        )
        return info_obj

    def _read_video_info_txt(self) -> videoInfo:
        if not self.video_path.exists():
            print(f"Could not find {self.video_path}, skipping for now")
            return

        raw_data = pd.read_csv(
            self.video_path,
            sep=": ",
            engine="python",
            header=0,
            names=["Info"],
            index_col=0,
        )["Info"]
        # Ensure all entries are strings
        raw_data = raw_data.astype(str)
        # Drop all columns with no data
        raw_data = raw_data.dropna(how="all")
        for col in raw_data.index:
            raw_data[col] = raw_data[col].strip()
        time_info = " ".join(raw_data["DateTime"].split(", ")[1:])
        time_obj = datetime.strptime(time_info, "%d %B %Y %X")
        crossing_date = datetime.date.fromisoformat(raw_data["CrossDate"])

        plate_info_obj = plateInfo(
            plate_nr=raw_data["Plate"],
            root=raw_data["Root"],
            strain=raw_data["Strain"],
            treatment=raw_data["Treatment"],
            crossing_date=crossing_date,
        )

        camera_settings = cameraSettings(
            model=raw_data["Model"],
            exposure_us=float(raw_data["ExposureTime"].split(" ")[0]),
            frame_rate=float(raw_data["FrameRate"].split(" ")[0]),
            frame_size=(raw_data["FrameSize"].split(" ")[0].split("x")),
            binning=raw_data["Binning"].split("x")[0],
            gain=raw_data["Gain"],
            gamma=raw_data["Gamma"],
        )

        position = cameraPosition(
            x=raw_data["X"].split(" ")[0],
            y=raw_data["Y"].split(" ")[0],
            z=raw_data["Z"].split(" ")[0],
        )

        info_obj = videoInfo(
            plate_info=plate_info_obj,
            date_time=time_obj,
            storage_path=raw_data["StoragePath"],
            run_nr=raw_data["Run"],
            duration=timedelta(seconds=int(raw_data["Time"].strip().split(" ")[0])),
            frame_nr=int(raw_data["Frames Recorded"].strip().split("/")[0]),
            mode=raw_data["Operation"].strip().split(" ")[1].lower(),
            magnification=float(raw_data["Operation"].strip().split()[0][:-1]),
            camera_settings=camera_settings,
            position=position,
        )
        return info_obj

    def _load_tif_series_to_dask(self) -> npt.ArrayLike:
        """
        Loads a series of .tif images from a folder into a Dask array.

        Parameters:
            folder_path (str or Path): Path to the folder containing the .tif images.

        Returns:
            dask.array.Array: A Dask array representing the .tif series.
        """
        folder = Path(self.video_path).parent
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
