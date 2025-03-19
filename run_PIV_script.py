from pathlib import Path
from flow_analysis_comps.PIV.PIV_process import AMF_PIV
from flow_analysis_comps.PIV.definitions import PIV_params
# from flow_analysis_comps.data_structs.video_info import videoMode

data_adr = Path(r"U:\test_data\20250122_Plate017")
plate_id = r"20250122_Plate017"
video_id = r"043"
frame_id1 = r"Img0000.tif"
frame_id2 = r"Img0001.tif"
filter_mode = "Img"
# filter_mode = "aharm_thresh"

raw_img_adr = data_adr / video_id / "Img"

fps = 20
winsize = 10  # pixels, interrogation window size in frame A
searchsize = 12  # pixels, search area size in frame B
overlap = 4  # pixels
frame_ids = (0, 1)

STN_thresh = .9

piv_param_obj = PIV_params(
    video_path=str(data_adr / video_id / filter_mode),
    segment_mode="other",
    fps=fps,
    window_size=winsize,
    search_size=searchsize,
    overlap_size=overlap,
    stn_threshold=STN_thresh,
    px_per_mm=1500 / 2,
)

amf_piv_obj = AMF_PIV(piv_param_obj)
# amf_piv_obj.piv_process(frame_ids, FAKE_OUTLIERS=False)
amf_piv_obj.piv_process_windef(frame_ids)