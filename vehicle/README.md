# 1. Extract object-attribute detection bounding box

We use the [Scene Graph Benchmark](https://github.com/microsoft/scene_graph_benchmark) project to detect color attributes for each vehicle and use them for  training. This project is based on Faster R-CNN. You can install the environment and download model weights based on the given link.

Once you have prepared the environment and model weights, you can automatically extract color attributes for vehicles using the following instructions.

```bash
# extract object-attribute detection bounding box and get color attribute for vehicles
$ cd scene_graph_benchmark
$ sh get_color_prediction.sh
```



# 2.Vehicle retrieve

## 2.1 Environmental Setup

Our project is based on the [BLIP](https://github.com/salesforce/BLIP/tree/main) model. You can prepare the environment like that project, or you can use the following commands provided by us.

We have tested the implementation on the following environment:

  * Python 3.7.16 / PyTorch 1.13.1 / torchvision 0.14.1 / CUDA 11.6 / Ubuntu 18.04

```bash
# download movienet data
$ cd ..
$ cd BLIP
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```



## 2.2 Prepare pre-trained model weights

We use BLIP with ViT-L (129M) as the backbone network, and you can download the model weights from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth). Place the downloaded model weights into the "output" directory.

```bash
# cd BLIP
ouput
│─ model_large.pth
```

## 2.3 Training

```bash
# for training
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_car.yaml \
--output_dir output/retrieval_car_itm_add_mask_color_img_promot_node5_epoch15 \
--pretrained ./output/model_large.pth
```



## 2.4 Test

```bash
# for test
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_car.yaml \
--output_dir output/retrieval_car_itm_add_mask_color_img_promot_node5_epoch15 \
--pretrained ./output/retrieval_car_itm_add_mask_color_img_promot_node5_epoch15/checkpoint_best.pth \
--evaluate
```

The best score file name is "blip_retrieval_car_itm_add_mask_color_img_promot_node5_epoch15_infer_3_submit.json"