import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import dask.array as da

from flow_analysis_comps.visualizing.video import videoVisualizer

class DummyMetadata:
    def __init__(self):
        self.storage_path = "/tmp"
        self.camera_settings = MagicMock()
        self.camera_settings.frame_rate = 30

class DummyVideoIO:
    def __init__(self, video_path):
        self.metadata = DummyMetadata()
        # Create a fake video: 5 frames of 64x64 RGB
        self.video_array = da.from_array(
            np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8),
            chunks=(1, 64, 64, 3)
        )

@pytest.fixture
def patch_videoio(monkeypatch):
    monkeypatch.setattr(
        "flow_analysis_comps.visualizing.video.videoIO",
        DummyVideoIO
    )

@pytest.fixture
def patch_writer(monkeypatch):
    mock_writer = MagicMock()
    mock_get_writer = MagicMock(return_value=mock_writer)
    monkeypatch.setattr(
        "flow_analysis_comps.visualizing.video.imageio.get_writer",
        mock_get_writer
    )
    return mock_get_writer, mock_writer

def test_video_visualizer_init(patch_videoio):
    visualizer = videoVisualizer("dummy_path")
    assert isinstance(visualizer.metadata, DummyMetadata)
    assert isinstance(visualizer.array, da.Array)

def test_save_mp4_video_calls_writer(patch_videoio, patch_writer):
    mock_get_writer, mock_writer = patch_writer
    visualizer = videoVisualizer("dummy_path")
    visualizer.save_mp4_video()
    mock_get_writer.assert_called_once()
    assert mock_writer.append_data.call_count == 5
    mock_writer.close.assert_called_once()