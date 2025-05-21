import pytest
from unittest.mock import patch, MagicMock

from flow_analysis_comps.scripts import video_to_mp4


def test_process_missing_total_path():
    run_info = {"not_total_path": "/some/path"}
    with pytest.raises(KeyError, match="The key 'total_path' is missing"):
        video_to_mp4.process(run_info)