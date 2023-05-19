# 1. Extract object-attribute detection bounding box

We use the [Scene Graph Benchmark](https://github.com/microsoft/scene_graph_benchmark) project to detect color attributes for each vehicle and use them for  training. This project is based on Faster R-CNN. You can install the environment and download model weights based on the given link.

Once you have prepared the environment and model weights, you can automatically extract color attributes for vehicles using the following instructions.

```bash
# extract object-attribute detection bounding box and get color attribute for vehicles
$ cd scene_graph_benchmark
$ sh get_color_prediction.sh
```



# 2.Vehicle retrieve

## 2.1 Preparation of competition dataset

To facilitate reasoning, we will directly provide the predicted results of the vehicle's color. The data directory is organized according to the following structure.

```bash
|--vehicle
|  |--CVPR_track2_DATA
|     |--test 
|        |--test_images
|        |--test_text.txt
|        |--test_color_from_bbox.json
|     |--train  
|        |--train_images
|        |--train_label.txt
|        |--train_color_from_bbox.json
|     |--val 
|        |--val_images
|        |--val_label.txt
|        |--val_color_from_bbox.json
```

## 2.2 Environmental Setup

Our project is based on the [BLIP](https://github.com/salesforce/BLIP/tree/main) model. You can prepare the environment like that project, or you can use the following commands provided by us.

We have tested the implementation on the following environment:

  * Python 3.7.16 / PyTorch 1.13.1 / torchvision 0.14.1 / CUDA 11.6 / Ubuntu 18.04

```bash
$ cd ..
$ cd BLIP
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```



## 2.3 Prepare pre-trained model weights

We use BLIP with ViT-L (129M) as the backbone network, and you can download the model weights from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth). Place the downloaded model weights into the "output" directory.

```bash
# cd BLIP
ouput
│─ model_large.pth
```

## 2.4 Training

```bash
# for training
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_car.yaml \
--pretrained ./output/model_large.pth \
--output_dir output/retrieval_car_itm_add_mask_color_img_promot_node5_epoch15
```



## 2.5 Test

```bash
# for test
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_car.yaml \
--pretrained ./output/retrieval_car_itm_add_mask_color_img_promot_node5_epoch15/checkpoint_best.pth \
--output_dir output/retrieval_car_itm_add_mask_color_img_promot_node5_epoch15 \
--evaluate
```

The best score file name is "blip_retrieval_car_itm_add_mask_color_img_promot_node5_epoch15_infer_3_submit.json".

To conduct testing on the B-board, you need to modify the data directories in get_color_prediction.sh and BLIP/configs/retrieval_car.yaml again.



# 车辆方案概述

## 模型设计

车辆检索采用[BLIP (BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation)](https://github.com/salesforce/BLIP/tree/main)，的代码作为修改。损失函数包括两个ITC loss和ITM loss。

ITC:：图文对比损失，将成对的车辆图文拉近，将不成对的车辆图文拉远，此时引入车辆的tag标记(如 white Audi)，避免同类图文的表征错误的拉远。

ITM：图文匹配损失，将成对的图文标记为1，不成对标记为0，同样引入车辆的tag标记。

## 数据处理

同行人一样将车辆数据分离出来进行训练，为了显示的为图像引入颜色标记，我们使用基于带属性检测的Fasfer-Rcnn（[Scene Graph Benchmark](https://github.com/microsoft/scene_graph_benchmark)）的目标检测模型为车辆检测颜色属性，然后在车辆图片的左上角打上颜色块，作为视觉promot增强，其作为辅助信息帮助模型更好的感知颜色属性。

### 优化策略

我们将BLIP模型训练15个epoch，将官方划分的val集合的mAP@10作为线下评测指标，取val集合最高mAP@10作为我们的最优模型。