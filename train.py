import torch
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.transforms.functional as FT

from tqdm import tqdm

from torch.utils.data import DataLoader
from  model import YoloV1

import os.path as osp

from dataset import PascalVOC
# from utils import(
#     intersection_over_union,
#     non_max_suppression,
#     mean_average_precision,
#     cellboxes_to_boxes,
#     get_bboxes,
#     plot_image,
#     save_checkpoint,
#     load_checkpoint
# )

from loss import YoloV1Loss

import json

seed = 123
torch.manual_seed(seed=seed)

with open("hyperparameters.json", 'r') as hp_file:
    hp_dict = json.load(hp_file)


print(f"Setting device to {eval(hp_dict["device"])} as GPU is{"" if eval(hp_dict["device"]) == "cuda" else " not"} available")
hp_dict["device"] = eval(hp_dict["device"])
print("Hyperparameters")
print("-"*50)
for key,val in hp_dict.items():
    print(key,":", val)

dataset_dir = "/home/gokul/Downloads/pascal_voc_aladdin"
img_dir = osp.join(dataset_dir, "images")
label_dir = osp.join(dataset_dir, "labels")
train_csv = osp.join(dataset_dir, "train.csv")
test_csv = osp.join(dataset_dir, "test.csv")

train_data = PascalVOC(csv_file=train_csv, image_dir=img_dir, label_dir=label_dir )

for i in range(len(train_data)):
    a,b = train_data[i]
    print(f"got {i}th train data")
    if i == 10:
        break