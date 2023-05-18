import os.path as op
from typing import List
import torch

from utils.iotools import read_json
from .bases import BaseDataset


class MYdata(BaseDataset):
    dataset_dir = 'mydata'

    def __init__(self, root='', verbose=True):
        super(MYdata, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'DT/images')

        self.anno_path = op.join(self.dataset_dir, 'DT/data.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos, val = True)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False, val = False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                attribute_txts = []
                for att in anno['attribute']:
                    attribute_txt = att.split(',')
                    attribute_txt = list(map(int, attribute_txt))
                    attribute_txt = torch.tensor(attribute_txt)
                    attribute_txts.append(attribute_txt)
                dataset.append((pid, image_id, img_path, captions, attribute_txts))
                image_id += 1
            return dataset, pid_container
        elif val:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            attributes = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                attribute = anno['attribute'][0].split(',')
                attribute = list(map(int, attribute))
                attribute = torch.tensor(attribute)
                attributes.append(attribute)
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
                "attributes": attributes
            }
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
