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

train_data = PascalVOC(csv_file=train_csv, image_dir=img_dir, label_dir=label_dir)
print(f"Data size: {len(train_data)}")

train_dl = DataLoader(
    train_data,
    batch_size=hp_dict["batch_size"],
    shuffle=True,
    num_workers=hp_dict["num_workers"],   # parallel data loading
    pin_memory=True  # faster GPU transfer
)
total_steps = len(train_dl)
print(f"Number of steps in an epoch: {total_steps}")

model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(hp_dict["device"])
loss_fn = YoloV1Loss().to(hp_dict["device"])
optimizer = optim.SGD(model.parameters(), lr=hp_dict["lr"])

num_epochs = hp_dict["epochs"]
device = hp_dict["device"]
steps = len(train_data)

# log
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

global_step = 0

epoch_bars = []

if __name__=="__main__":
    for epoch in range(num_epochs):

        # create new progress bar

        pbar = tqdm(
            total=total_steps,
            desc=f"{epoch+1}/{num_epochs}",
            position=epoch,     # <-- stack downward
            leave=True          # <-- persist after completion
        )
        epoch_bars.append(pbar)
        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            # import pdb; pdb.set_trace()
            out = out.reshape(-1, 7,7,30)
            loss = loss_fn(out, batch_y)
            # print(f"Loss: {loss.item()}")
            writer.add_scalar("Loss/train", loss.item(), global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.update(1)
