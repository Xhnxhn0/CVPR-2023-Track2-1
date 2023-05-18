# 1. Pedestrian retrieve

## 1.1 General description
We use the [Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval](https://github.com/anosorae/IRRA) project to  identify pedestrian attribute features and complete text to image retrieval by aligning image text. This project is based on clip(Vit-B-16). You can install the environment and download model weights based on the given link. Note: We use a [pre-trained model](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view?usp=sharing) trained on the RSTPReid dataset as the initialization weights for our model.

## 1.2 Requirements
We use single RTX4090 24G GPU for training and evaluation. 
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```

## 1.3 Preparation of competition dataset
Firstly, we will organize the competition dataset into the following format.
Note: The folder test_person here only includes images of pedestrians to be retrieved.The folder mydata/DT/images here includes all images(train, val and test) of pedestrians.
```
|--pedestrian
|  |--contest_data
|     |--test 
|        |--test_person
|        |--test_text.txt
|     |--train  
|        |--train_images
|        |--train_label.txt
|     |--val 
|        |--val_images
|        |--val_label.txt
|   |--mydata
|      |--DT
|         |--images
```
Then we execute the following command to obtain the JSON form of the data.
```
python txt_to_json_train.py
cp data.json mydata/DT/
```

## 1.4 Prepare pre-trained model weights

We use CLIP with ViT-B-16 as the backbone network, and you can download the model weights from [here](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view?usp=sharing). Place the downloaded model weights into the "pedestrian" directory.

```
|--pedestrian
|  |--best.pth
```

## 1.5 Training

```bash
# for training
bash train.sh
```



## 1.6 Test

```bash
# for test
bash infer.sh
```
The best result is produced at the 18 epoch and the best score file name is "infer_content/18/infer_json.json".
The model is saved at "logs/MY-DATA/***/".