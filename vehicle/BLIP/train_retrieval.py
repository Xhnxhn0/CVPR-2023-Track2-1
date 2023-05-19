'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from tqdm import tqdm
from data import create_dataset, create_sampler, create_loader
os.environ[ "CUDA_VISIBLE_DEVICES"] = "2,3,4"


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1, i/len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config, is_val=True):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35,
                                     return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    class_labels = []

    if is_val:
        for image, img_id, class_label in tqdm(data_loader):
            image = image.to(device)
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)
            class_labels.append(class_label)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)
        class_labels = torch.cat(class_labels, dim=0)
    else:
        for image, img_id in tqdm(data_loader):
            image = image.to(device)
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)
        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)


    sims_matrix = (image_embeds @ text_embeds.t()).t()

    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                    attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] += score + topk_sim

    if args.distributed:
        dist.barrier()
        # torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    if is_val:
        return score_matrix_t2i.cpu().numpy(), class_labels.numpy()
    else:
        return score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def fx_calc_map_label(orders, label, k=0):
    numcases = orders.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = orders[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)

@torch.no_grad()
def itm_eval(scores_t2i, txt2img, class_labels):
    #Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    inds_list = []

    for index, score in enumerate(tqdm(scores_t2i)):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]
        inds_list.append(inds)

    inds_list = np.stack(inds_list)
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    t2i_mAP_list = []
    for i in range(1, 11):
        t2i_mAP = fx_calc_map_label(inds_list, class_labels, k=i)
        t2i_mAP_list.append(t2i_mAP)

    t2i_mAP_dict = dict(zip([r'img_mAP@top{}'.format(i) for i in range(1, 11)], t2i_mAP_list))

    ir_mean = (ir1 + ir5 + ir10) / 3

    eval_result =  dict({
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean}, **t2i_mAP_dict)

    return eval_result

@torch.no_grad()
def t2i_infer(scores_t2i, id2txt, id2img):
    person_padding = ["090001.jpg", "090002.jpg", "090003.jpg", "090004.jpg", "090005.jpg", "090006.jpg", "090007.jpg",
                      "090008.jpg", "090009.jpg", "090010.jpg"]
    vehicle_padding = ["vehicle_0000002.jpg", "vehicle_0000003.jpg", "vehicle_0000004.jpg", "vehicle_0000005.jpg",
                "vehicle_0000006.jpg", "vehicle_0000007.jpg", "vehicle_0000008.jpg", "vehicle_0000009.jpg",
                "vehicle_0000010.jpg", "vehicle_0000011.jpg"]
    infer_dict = {}
    for i in range(1, 11):
        result_list = []
        for index, score in enumerate(tqdm(scores_t2i)):
            inds = np.argsort(score)[::-1]
            t2i_ans = {}
            t2i_ans["text"] = id2txt[index]
            t2i_ans["image_names"] = []
            for j in range(i):
                t2i_ans["image_names"].append(id2img[inds[j]])
            if "vehicle" in t2i_ans["image_names"][0]:
                t2i_ans["image_names"] = t2i_ans["image_names"] + person_padding[0: (10-i)]
            else:
                t2i_ans["image_names"] = t2i_ans["image_names"] + vehicle_padding[0: (10-i)]

            result_list.append(t2i_ans)
        infer_dict["infer_{}".format(i)] = {'results': result_list}
    return infer_dict


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None,None,None])


    #### Model ####
    print("Creating model")
    model = blip_retrieval(pretrained=args.pretrained, image_size=config['image_size'], vit=config['vit'],
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config)

        score_val_t2i, class_labels = evaluation(model_without_ddp, val_loader, device, config, is_val=True)

        if args.evaluate:
            score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, is_val=False)

        if utils.is_main_process():

            val_result = itm_eval(score_val_t2i, val_loader.dataset.txt2img, class_labels)
            print(val_result)

            if val_result['img_mAP@top10'] > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = val_result['img_mAP@top10']
                best_epoch = epoch

            if epoch > 9:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))


            if args.evaluate:
                val_infer_result = t2i_infer(score_val_t2i, val_loader.dataset.id2txt, val_loader.dataset.id2img)
                test_result = t2i_infer(score_test_t2i, test_loader.dataset.id2txt, test_loader.dataset.id2img)

                # write result
                for infer_name, infer_result in val_infer_result.items():
                    with open(os.path.join(args.output_dir,
                                           "blip_val_{}_{}_submit.json".format(args.output_dir.split('/')[-1], infer_name)),
                              "w") as f:
                        f.write(json.dumps(infer_result, indent=4))
                        print("推理完成 !!!!!!!!!!!!!!!!!!!")

                for infer_name, infer_result in test_result.items():
                    with open(os.path.join(args.output_dir,
                                           "blip_{}_{}_submit.json".format(args.output_dir.split('/')[-1], infer_name)),
                              "w") as f:
                        f.write(json.dumps(infer_result, indent=4))
                        print("推理完成 !!!!!!!!!!!!!!!!!!!")
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             # **{f'test_{k}': v for k, v in test_result.items()},
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             # **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--pretrained', default='./output/model_large.pth')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)