import json
import os

import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption

promot_h = 30
promot_w = 30
color_padding_tensor = {"white": (255, 255, 255),
                        "black": (0, 0, 0),
                        "yellow": (255, 255, 0),
                        "purple": (160, 32, 240),
                        "pink": (255, 192, 203),
                        "red": (255, 0, 0),
                        "orange": (255, 97, 0),
                        "blue": (0, 0, 255),
                        "brown": (128, 42, 42),
                        "green": (0, 255, 0),
                        "grey": (192, 192, 192)}

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, color_info, max_words=30):

        with open(color_info, "r") as f:
            self.color_predictions = json.load(f)

        self.ann = []
        num_of_color_is_None = 0
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split('$')
                if len(line[1].split(',')) != 21:
                    ann = {}
                    ann["img_name"] = line[0]
                    ann["tag"] = line[1]
                    ann["caption"] = line[2]
                    ann["color"] = self.find_color(ann["tag"])
                    if ann["color"] is None and ann["img_name"] in self.color_predictions.keys():
                        ann["color"] = self.color_predictions[ann["img_name"]]["color"]
                    if ann["color"] is None:
                        num_of_color_is_None += 1
                    self.ann.append(ann)


        print("train ann len is ", len(self.ann))
        print("num_of_color_is_None is ", num_of_color_is_None)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.class_label2id = {}

        n = 0
        class_num = 0
        for ann in self.ann:
            img_id = ann["img_name"]
            tag = ann["tag"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            if tag not in self.class_label2id.keys():
                self.class_label2id[tag] = class_num
                class_num += 1
        print("self.class_label2id is ", self.class_label2id)
        print("self.class_label2id len is ", len(self.class_label2id))

    def find_color(self, tag):
        if "white" in tag:
            return "white"
        elif "black" in tag:
            return "black"
        elif "yellow" in tag:
            return "yellow"
        elif "purple" in tag:
            return "purple"
        elif "pink" in tag:
            return "pink"
        elif "red" in tag:
            return "red"
        elif "orange" in tag:
            return "orange"
        elif "blue" in tag:
            return "blue"
        elif "brown" in tag:
            return "brown"
        elif "green" in tag:
            return "green"
        elif "grey" in tag:
            return "grey"
        else:
            return None

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_name = ann["img_name"]

        if ann['color'] is not None:
            color = ann['color']
        elif image_name in self.color_predictions.keys():
            color = self.color_predictions[image_name]["color"]
        else:
            color = None
        if color == "gray":
            color = "grey"
        image_path = os.path.join(self.image_root, image_name)
        image = Image.open(image_path).convert('RGB')
        if color is not None:
            image.paste(color_padding_tensor[color], (0, 0, promot_h, promot_w))
        image = self.transform(image)

        caption = pre_caption(ann["caption"], self.max_words)
        return image, caption, self.class_label2id[ann["tag"]]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, color_info, max_words=30):

        with open(color_info, "r") as f:
            self.color_predictions = json.load(f)

        self.ann = []
        num_of_color_is_None = 0
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split('$')
                if len(line[1].split(',')) != 21:
                    ann = {}
                    ann["img_name"] = line[0]
                    ann["tag"] = line[1]
                    ann["caption"] = line[2]
                    ann["color"] = self.find_color(ann["tag"])
                    if ann["color"] is None and ann["img_name"] in self.color_predictions.keys():
                        ann["color"] = self.color_predictions[ann["img_name"]]["color"]
                    if ann["color"] is None:
                        num_of_color_is_None += 1
                    self.ann.append(ann)


        print("val ann len is ", len(self.ann))
        print("num_of_color_is_None is ", num_of_color_is_None)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.id2txt = {}
        self.id2img = {}
        self.class_label2id = {}

        txt_id = 0
        class_num = 0
        for img_id, ann in enumerate(self.ann):
            image_name = ann["img_name"]
            self.image.append(image_name)
            self.img2txt[img_id] = []
            caption = ann["caption"]
            self.text.append(pre_caption(caption, self.max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            self.id2txt[txt_id] = caption
            self.id2img[img_id] = image_name
            tag = ann["tag"]
            if tag not in self.class_label2id.keys():
                self.class_label2id[tag] = class_num
                class_num += 1
            txt_id += 1


    def find_color(self, tag):
        if "white" in tag:
            return "white"
        elif "black" in tag:
            return "black"
        elif "yellow" in tag:
            return "yellow"
        elif "purple" in tag:
            return "purple"
        elif "pink" in tag:
            return "pink"
        elif "red" in tag:
            return "red"
        elif "orange" in tag:
            return "orange"
        elif "blue" in tag:
            return "blue"
        elif "brown" in tag:
            return "brown"
        elif "green" in tag:
            return "green"
        elif "grey" in tag:
            return "grey"
        else:
            return None

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_name = ann["img_name"]
        if ann['color'] is not None:
            color = ann['color']
        elif image_name in self.color_predictions.keys():
            color = self.color_predictions[image_name]["color"]
        else:
            color = None
        if color == "gray":
            color = "grey"
        image_path = os.path.join(self.image_root, image_name)
        image = Image.open(image_path).convert('RGB')
        if color is not None:
            image.paste(color_padding_tensor[color], (0, 0, promot_h, promot_w))
        image = self.transform(image)
        tag = ann["tag"]

        return image, index, self.class_label2id[tag]


class re_test_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, color_info, max_words=30):

        self.ann = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.ann.append(line)

        # for vehicle
        self.ann = self.ann[0: 7611]

        with open(color_info, "r") as f:
            self.color_predictions = json.load(f)

        self.text = []
        self.text_ori = []
        self.max_words = max_words
        img_list = os.listdir(image_root)
        self.image = [image_name for image_name in img_list if "vehicle" in image_name]

        print("self.image len is ", len(self.image))
        for caption in self.ann:
            self.text.append(pre_caption(caption, self.max_words))
            self.text_ori.append(caption)

        self.transform = transform
        self.image_root = image_root


        self.id2txt = {}
        self.img2id = {}
        self.id2img = {}

        for text_id, t in enumerate(self.text_ori):
            self.id2txt[text_id] = t

        for img_id, image_name in enumerate(self.image):
            self.img2id[image_name] = img_id
            self.id2img[img_id] = image_name

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_name = self.image[index]
        color = self.color_predictions[image_name]["color"]
        if color == "gray":
            color = "grey"
        image_path = os.path.join(self.image_root, image_name)
        image = Image.open(image_path).convert('RGB')
        if color is not None:
            image.paste(color_padding_tensor[color], (0, 0, promot_h, promot_w))
        image = self.transform(image)

        return image, index
