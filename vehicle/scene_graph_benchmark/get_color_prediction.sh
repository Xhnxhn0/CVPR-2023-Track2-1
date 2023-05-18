img_file=../../CVPR_track2_DATA
save_file=../../CVPR_track2_DATA

# object-attribute detection 
python3 tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ${img_file} --save_file ${save_file} --visualize_attr MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False

# get color prediction result
python3 get_color_prediction.py
