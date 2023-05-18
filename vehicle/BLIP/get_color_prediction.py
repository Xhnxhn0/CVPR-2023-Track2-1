import json
import os
import numpy as np
import re
from tqdm import tqdm

fn = "D:/code/project_code/CVPR2023/DATA/CVPR_track2_DATA/train/bbox"
file_list = os.listdir(fn)
color_set = set(['white', 'black', 'yellow', 'purple', 'pink', 'red', 'orange', 'blue', 'brown', 'green', 'grey', 'gray'])
ans = {}
class_set = []
for file_name in tqdm(file_list):
    bboxs = []
    with open(os.path.join(fn, file_name), "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = re.split(',| |\t', line)
            bboxs.append(line)
    deal_list = []
    for i in range(0, len(bboxs), 2):
        info_dict = {}
        class_type = bboxs[i][-1]
        class_type_prob = bboxs[i+1][-1]
        class_set.append(class_type)
        if class_type == "car" or class_type == "sedan" or class_type == "truck" or  class_type == "minivan" or \
            class_type == "bus" or class_type == "suv" or class_type == "vehicle" or class_type == "jeep":
#             print(bboxs[i])
            for j in range(0, len(bboxs[i])):
#                 print("j is ", j)
#                 print("bbox[i][j] is ", bboxs[i][j])
                if bboxs[i][j] in color_set:
                    color = bboxs[i][j]
            info_dict["class"] = class_type
            info_dict["class_prob"] = class_type_prob
            info_dict["color"] = color
            deal_list.append(info_dict)
    ans["{}.jpg".format(file_name.split(".")[0])] = deal_list

color_info_dict = {}
for img_name, info in tqdm(ans.items()):
    if len(info) > 0:
        color_info_dict[img_name] = {"color": info[0]['color']}

with open(os.path.join("D:/code/project_code/CVPR2023/DATA/CVPR_track2_DATA/train", "train_color_from_bbox.json"), "w") as f:
    f.write(json.dumps(color_info_dict, indent=4))