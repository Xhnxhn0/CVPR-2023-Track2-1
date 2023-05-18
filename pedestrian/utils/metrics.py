from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import json
import os.path as op
import torch.nn as nn
from typing import List

from utils.iotools import read_json

def my_mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path) 

def test_rank(similarity, qfeats, gfeats, epoch, max_rank=10,get_mAP=True):
    text_list_now = []
    image_list_now = []
    test_annos = []
    annos = read_json("mydata/DT/data.json")
    for anno in annos:
        if anno['split'] == 'test':
            test_annos.append(anno)
    for anno in test_annos:
        for caption in anno['captions']:
            text_list_now.append(caption)
        image_list_now.append(anno['file_path'])
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)

        result_list = []
        pre = 3
        result_list = []
        for index in range(indices.shape[0]):
            dic = {'text': text_list_now[index], 'image_names': []}
            for j in range(pre):
                dic['image_names'].append(image_list_now[indices[index][j]])
            for j in range(10 - pre):
                dic['image_names'].append(image_list_now[indices[index][indices.shape[0] - j - 1]])
                
            result_list.append(dic)
        my_mkdir(f'infer_content/{epoch}')
        with open(f'infer_content/{epoch}/infer_json.json', 'w') as f:
            f.write(json.dumps({'results': result_list}, indent=4))
        return

def attributes_to_str(attributes):
    attributes = attributes.tolist()
    attributes = list(map(lambda x:str(x), attributes))
    attributes = ",".join(attributes)
    return attributes

def rank(similarity, q_pids, g_pids, attributes, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    #sim = torch.sort(similarity, dim=1, descending=True)
    n = attributes.shape[0]
    now_list = []
    for t in range(21):
        to_img = attributes[:,t:t+1].reshape(-1, 1)
        to_img = to_img[indices].reshape(n, n)
        to_txt = attributes[:,t:t+1].reshape(-1, 1)
        to_img -= to_txt
        now_list.append(to_img)
    matches_now = now_list[0]
    for t in range(1, 21):
        temp = attributes[:,t:t+1].reshape(-1)
        index = torch.nonzero(temp == torch.tensor(0)).reshape(-1)
        now_list[t].index_fill_(0, index, 0.)
        matches_now = (matches_now | now_list[t])
    matches_now = (matches_now == False)

    all_cmc = matches_now[:, :max_rank].cumsum(1) # cumulative sum

    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    #all_cmc = all_cmc[max_rank - 1]

    if not get_mAP:
        return all_cmc, indices
    mAP = []
    for k in range(10):
        matches = matches_now[:,0:(k+1)]
        num_rel = matches.sum(1)  # q
        tmp_cmc = matches.cumsum(1)  # q * k
        tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
        tmp_cmc = torch.stack(tmp_cmc, 1) * matches
        for i in range(tmp_cmc.shape[0]):
            if num_rel[i] == 0:
                num_rel[i] = int(1)
        AP = tmp_cmc.sum(1) / num_rel  # q
        mAP.append(AP.mean() * 100)
    return all_cmc, mAP, indices

class Evaluator():
    def __init__(self, img_loader, txt_loader,epoch = -1,test=False):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.now_epoch = epoch
        self.test = test
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        if self.test:
            qids, gids, qfeats, gfeats = [], [], [], []
            # text
            for pid, caption in self.txt_loader:
                caption = caption.to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(caption)
                qids.append(pid.view(-1)) # flatten 
                qfeats.append(text_feat)
            qids = torch.cat(qids, 0)
            qfeats = torch.cat(qfeats, 0)

            # image
            for pid, img in self.img_loader:
                img = img.to(device)
                with torch.no_grad():
                    img_feat = model.encode_image(img)
                gids.append(pid.view(-1)) # flatten 
                gfeats.append(img_feat)
            gids = torch.cat(gids, 0)
            gfeats = torch.cat(gfeats, 0)

            return qfeats, gfeats, qids, gids
        else:
            qids, gids, qfeats, gfeats = [], [], [], []
            attributes = []
            # text
            for pid, caption, attribute in self.txt_loader:
                caption = caption.to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(caption)
                qids.append(pid.view(-1)) # flatten 
                qfeats.append(text_feat)
                attributes.append(attribute)
            qids = torch.cat(qids, 0)
            qfeats = torch.cat(qfeats, 0)
            attributes = torch.cat(attributes, 0)
            
            # image
            for pid, img, attribute in self.img_loader:
                img = img.to(device)
                with torch.no_grad():
                    img_feat = model.encode_image(img)
                gids.append(pid.view(-1)) # flatten 
                gfeats.append(img_feat)
            gids = torch.cat(gids, 0)
            gfeats = torch.cat(gfeats, 0)

            return qfeats, gfeats, qids, gids, attributes
    
    def eval(self, model, i2t_metric=False):

        if self.test:
            qfeats, gfeats, qids, gids = self._compute_embedding(model)
        else:
            qfeats, gfeats, qids, gids, attributes = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        if self.test:
            test_rank(similarity=similarity, qfeats=qfeats, gfeats=gfeats, epoch = self.now_epoch,max_rank=10,get_mAP=True)
            return 0
        else:
            t2i_cmc, t2i_mAP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, attributes=attributes, max_rank=10, get_mAP=True)
            t2i_cmc = t2i_cmc.numpy()
            for i in range(10):
                t2i_mAP[i] = t2i_mAP[i].numpy()
            tmp_tabel = ["task"]
            tmp_tabel.append("Max")
            for i in range(10):
                tmp_tabel.append(f"mAP{i+1}")
            table = PrettyTable(tmp_tabel)

            tmp_tabel = ["t2i"]
            tmp_tabel.append(max(t2i_mAP))
            for i in range(10):
                tmp_tabel.append(t2i_mAP[i])
            table.add_row(tmp_tabel)

            table.custom_format["Max"] = lambda f, v: f"{v:.3f}"
            for i in range(10):
                table.custom_format[f"mAP{i+1}"] = lambda f, v: f"{v:.3f}"
            self.logger.info('\n' + str(table))
            
            return max(t2i_mAP)