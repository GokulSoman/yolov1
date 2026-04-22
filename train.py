import torch
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.transforms.functional as FT

from tqdm import tqdm

from torch.utils.data import DataLoader
from  model import YoloV1

import os.path as osp

import wandb

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

dataset_dir = "/home/gokul/Downloads/pascal_voc_aladdin"
img_dir = osp.join(dataset_dir, "images")
label_dir = osp.join(dataset_dir, "labels")
train_csv = osp.join(dataset_dir, "train.csv")
test_csv = osp.join(dataset_dir, "test.csv")


seed = 123
torch.manual_seed(seed=seed)

with open("hyperparameters.json", 'r') as hp_file:
    hp_dict = json.load(hp_file)


# print(f"Setting device to {eval(hp_dict["device"])} as GPU is{"" if eval(hp_dict["device"]) == "cuda" else " not"} available")
# hp_dict["device"] = eval(hp_dict["device"])
if hp_dict["device"] == "cuda"and not torch.cuda.is_available():
    print("Cuda not available: switching to cpu")
    hp_dict["device"] = "cpu"
    if hp_dict["batch_size"] > 16:
        print("[WARN]: Possible crash. running cpu with batchsize > 16")

# print("Hyperparameters")
# print("-"*50)
# for key,val in hp_dict.items():
#     print(key,":", val)


# set deterministic runs
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# set lower precision (tf32)

torch.set_float32_matmul_precision("high")


train_data = PascalVOC(csv_file=train_csv, image_dir=img_dir, label_dir=label_dir)
test_data = PascalVOC(csv_file=test_csv, image_dir=img_dir, label_dir=label_dir)
# print(f"Train data size: {len(train_data)}")
# print(f"Test data size: {len(test_data)}")


train_dl = DataLoader(
    train_data,
    batch_size=hp_dict["batch_size"],
    shuffle=True,
    num_workers=hp_dict["num_workers"],   # parallel data loading
    pin_memory=True  # faster GPU transfer
)
tr_total_steps = len(train_dl)
# print(f"Number of steps in an epoch: {tr_total_steps}")

test_dl = DataLoader(
    test_data,
    batch_size=hp_dict["batch_size"],
    shuffle=True,
    num_workers=hp_dict["num_workers"],   # parallel data loading
    pin_memory=True  # faster GPU transfer
)
test_total_steps = len(test_dl)
# print(f"Number of steps in an epoch: {test_total_steps}")

model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(hp_dict["device"])

loss_fn = YoloV1Loss().to(hp_dict["device"])

if hp_dict["compile"] == True:
    model = torch.compile(model)
    loss_fn = torch.compile(loss_fn)

optimizer = optim.AdamW(model.parameters(), lr=hp_dict["lr"],fused=hp_dict["fused"],
                        weight_decay=hp_dict["weight_decay"])

num_epochs = hp_dict["epochs"]
device = hp_dict["device"]
steps = len(train_data)

# log
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

global_step = 0

epoch_bars = []
import time
import math

def get_decay_lr(config, steps_per_epoch, step):
    min_lr = config["min_lr"]
    max_lr = config["max_lr"]
    warm_up_epochs = config["warm_up_epochs"]
    saturate_epochs = config["saturate_epochs"]
    warm_up_steps = steps_per_epoch * warm_up_epochs
    saturate_steps = steps_per_epoch * saturate_epochs
    if step < warm_up_steps:
        return max_lr * (step + 1) / warm_up_steps
    if step > saturate_steps:
        return min_lr
    
    decay_ratio = (step - warm_up_steps) / (saturate_steps - warm_up_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

if __name__=="__main__":

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="gokulsoman",
        # Set the wandb project where this run will be logged.
        project="yolov1",
        # Track hyperparameters and run metadata.
        config=hp_dict,
    )

    curr_test_loss = torch.inf

    # epoch_pbar = tqdm(
    #         total=num_epochs,
    #         desc=f"{"Training progress"}",
    #         position=0,     # <-- stack downward
    #         leave=True,         # <-- persist after completion
    #         mininterval=0.5
    #     )
    # epoch_bars.append(epoch_pbar)

    for epoch in range(num_epochs):

        # create new progress bar

        # pbar = tqdm(
        #     total=tr_total_steps,
        #     desc=f"{epoch+1}/{num_epochs}",
        #     position=epoch+1,     # <-- stack downward
        #     leave=True,          # <-- persist after completion
        #     mininterval=0.5
        # )
        # epoch_bars.append(pbar)

        for batch_x, batch_y in train_dl:
            t0 = time.time()

            # set learning rate
            lr = get_decay_lr(hp_dict, tr_total_steps, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype= torch.bfloat16):
                out = model(batch_x)
                loss = loss_fn(out, batch_y)
            # import pdb; pdb.set_trace()
            # print(f"Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1-t0)*1000
            im_sec = batch_y.shape[0]*1000/dt
            print(f"step: {global_step:4d}, lr: {lr:.5f}, loss: {loss.item():.4f}, norm: {norm:.4f}, dt: {dt:.3f}ms, im_sec: {im_sec}")
            log_train = {"loss/train": loss.item(), 
                        "lr": lr, 
                        "step/train" : global_step, 
                        "norm": norm, 
                        "iter_ms/train" : dt, 
                        "im_sec/train": im_sec}
            for k,v in log_train.items():
                writer.add_scalar(k, v, global_step)
            run.log(log_train)
            global_step += 1
            # pbar.set_postfix(loss=f"{loss.item():.4f}")
            # pbar.update(1)
            # if global_step == 50:
            #     break

        # test step
        if (epoch+1) % 10 == 0:
            test_loss = 0.0
            dt = 0.0
            im_sec = 0.0
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in test_dl:
                    t0 = time.time()
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        out = model(batch_x)
                        loss = loss_fn(out, batch_y)
                    test_loss += loss.item()
                    torch.cuda.synchronize()
                    t1 = time.time()
                    dt+= (t1-t0)*1000
                    im_sec += batch_y.shape[0]*1000/dt
            test_loss /= test_total_steps
            dt /= test_total_steps
            im_sec /= test_total_steps
            log_test = {"loss/test": test_loss, "step/test" : global_step-1, "iter_ms/test" : dt, "im_sec/test": im_sec}
            for k,v in log_test.items():
                writer.add_scalar(k, v, global_step-1)
            run.log(log_test)
            if test_loss < curr_test_loss:
                loss_underscored = "_".join(f"{test_loss:.3f}".split('.'))
                # torch.save(model.state_dict(), f"model_{loss_underscored}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": test_loss
                }, f"model_e{epoch}_loss_{loss_underscored}.pth")
                curr_test_loss = test_loss
            model.train()

    run.finish()
