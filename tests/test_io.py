from pathlib import Path
import pytest
from unittest.mock import patch

import numpy as np

from flow_analysis_comps.io.video import videoIO


@pytest.fixture
def mock_video_folder(tmp_path: Path):
    # Create a fake .json file inside the folder
    json_path = tmp_path / "test.json"
    json_path.write_text(
        '{"metadata": {"camera": {"intensity": [0, 1], "model": "cam", "exposure_time": 0.01, "frame_rate": 10, "frame_size": [512, 512], "binning": "1x1", "gain": 1, "gamma": 1}, "video": {"location": [0,0,0], "duration": 1}}}'
    )
    return tmp_path


def test_read_video_info_json(mock_video_folder):
    # Patch cameraPosition, cameraSettings, videoInfo to simple mocks
    with (
        patch("flow_analysis_comps.io.video.cameraPosition", lambda x, y, z: (x, y, z)),
        patch("flow_analysis_comps.io.video.cameraSettings", lambda **kwargs: kwargs),
        patch("flow_analysis_comps.io.video.videoInfo", lambda **kwargs: kwargs),
        patch.object(
            videoIO, "_load_tif_series_to_dask", return_value=np.zeros((1, 2, 2))
        ),
    ):
        vio = videoIO(mock_video_folder)
        assert vio.metadata["mode"] == "fluorescence"
        assert vio.metadata["camera_settings"]["model"] == "cam"


def test_read_video_info_txt(tmp_path: Path):
    txt_path = tmp_path / "test.txt"
    txt_content = (
        "HEADER\n"
        "DateTime: Wednesday, 22 January 2025, 15:38:33\n"
        "CrossDate: 20250122\n"
        "Plate: 1\n"
        "Root: Carrot\n"
        "Strain: C2\n"
        "Treatment: 001P100N100C\n"
        "Model: cam\n"
        "ExposureTime: 1000 us\n"
        "FrameRate: 10 Hz\n"
        "FrameSize: 512x512\n"
        "Binning: 1x1\n"
        "Gain: 1\n"
        "Gamma: 1\n"
        "X: 0 um\n"
        "Y: 0 um\n"
        "Z: 0 um\n"
        "StoragePath: /tmp\n"
        "Run: 1\n"
        "Time: 60 s\n"
        "Frames Recorded: 60/60\n"
        "Operation: 10x brightfield\n"
    )
    txt_path.write_text(txt_content)
    # Pass the folder, not the file, to videoIO
    with (
        patch("flow_analysis_comps.io.video.plateInfo", lambda **kwargs: kwargs),
        patch("flow_analysis_comps.io.video.cameraPosition", lambda x, y, z: (x, y, z)),
        patch("flow_analysis_comps.io.video.cameraSettings", lambda **kwargs: kwargs),
        patch("flow_analysis_comps.io.video.videoInfo", lambda **kwargs: kwargs),
        patch.object(
            videoIO, "_load_tif_series_to_dask", return_value=np.zeros((1, 2, 2))
        ),
    ):
        vio = videoIO(tmp_path)
        assert vio.metadata["mode"] == "brightfield"
        assert vio.metadata["plate_info"]["plate_nr"] == "1"


def test_load_tif_series_to_dask(tmp_path: Path):
    # Create fake tif files
    import tifffile

    arr = np.ones((2, 2), dtype=np.uint8)
    # Create a directory structure
    (tmp_path / "Img").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        tifffile.imwrite(tmp_path / "Img" / f"img_{i}.tif", arr)
    # Also add a dummy metadata file so __init__ doesn't fail
    (tmp_path / "dummy.json").write_text(
        '{"metadata": {"camera": {"intensity": [0, 1], "model": "cam", "exposure_time": 0.01, "frame_rate": 10, "frame_size": [512, 512], "binning": "1x1", "gain": 1, "gamma": 1}, "video": {"location": [0,0,0], "duration": 1}}}'
    )
    # Patch metadata reading
    with patch.object(videoIO, "_read_video_metadata", return_value=None):
        vio = videoIO(tmp_path)
        dask_arr = vio._load_tif_series_to_dask()
        assert dask_arr.shape[0] == 3
        assert dask_arr.shape[1:] == arr.shape
