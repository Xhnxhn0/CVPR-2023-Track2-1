# object-attribute detection 
train_img_file=../CVPR_track2_DATA/train/train_images
train_save_file=../CVPR_track2_DATA/train
python3 tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ${train_img_file} --save_file ${train_save_file} --visualize_attr MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False

# get color prediction result
train_box_save_file=../CVPR_track2_DATA/train/bbox
train_color_json_fn=../CVPR_track2_DATA/train/train_color_from_bbox.json 
python3 get_color_prediction.py --bbox_file ${train_box_save_file} --save_file ${train_color_json_fn}

# object-attribute detection 
val_img_file=../CVPR_track2_DATA/val/val_images
val_save_file=../CVPR_track2_DATA/val
python3 tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ${val_img_file} --save_file ${val_save_file} --visualize_attr MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False

# get color prediction result
val_box_save_file=../CVPR_track2_DATA/val/bbox
val_color_json_fn=../CVPR_track2_DATA/val/val_color_from_bbox.json 
python3 get_color_prediction.py --bbox_file ${val_box_save_file} --save_file ${val_color_json_fn}

# object-attribute detection 
test_img_file=../CVPR_track2_DATA/test/test_images
test_save_file=../CVPR_track2_DATA/test
python3 tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ${test_img_file} --save_file ${test_save_file} --visualize_attr MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False

# get color prediction result
test_box_save_file=../CVPR_track2_DATA/test/bbox
test_color_json_fn=../CVPR_track2_DATA/test/val_color_from_bbox.json 
python3 get_color_prediction.py --bbox_file ${test_box_save_file} --save_file ${test_color_json_fn}


