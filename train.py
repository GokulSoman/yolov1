import torch
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.transforms.functional as FT

from tqdm import tqdm

from torch.utils.data import DataLoader
from  model import YoloV1

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
print(hp_dict)