import json
import os
import numpy as np
import argparse
import re
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Save_color_prediction")
    parser.add_argument("--bbox_file", metavar="FILE", help="bbox path")
    parser.add_argument("--save_file", required=False, type=str, default=None,
                        help="filename to save the color prediction")

    args = parser.parse_args()
    file_list = os.listdir(args.bbox_file)
    print("file_list len is ", len(file_list))
    color_set = set(['white', 'black', 'yellow', 'purple', 'pink', 'red', 'orange', 'blue', 'brown', 'green', 'grey', 'gray'])
    ans = {}
    class_set = []
    for file_name in tqdm(file_list):
        bboxs = []
        with open(os.path.join(args.bbox_file, file_name), "r") as f:
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
            if class_type == "car" or class_type == "sedan" or class_type == "truck" or class_type == "minivan" or \
                class_type == "bus" or class_type == "suv" or class_type == "vehicle" or class_type == "jeep":
                for j in range(0, len(bboxs[i])):
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


    with open(args.save_file, "w") as f:
        f.write(json.dumps(color_info_dict, indent=4))


if __name__ == "__main__":
    main()
